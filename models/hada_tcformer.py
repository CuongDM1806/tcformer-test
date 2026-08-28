"""TCFormer with HADANet-style unsupervised domain adaptation.

Only source labels contribute to classification. Target batches contain EEG
samples only and are used by the adversarial and MK-MMD alignment objectives.
"""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchmetrics.functional import accuracy

from .classification_module import ClassificationModule
from .tcformer import TCFormerModule
from utils.latency import measure_latency


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, alpha: float) -> Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        return -ctx.alpha * grad_output, None


class GradientReversal(nn.Module):
    def forward(self, x: Tensor, alpha: float = 1.0) -> Tensor:
        return _GradientReversal.apply(x, alpha)


class ResidualFeatureAligner(nn.Module):
    """Learn a domain-shift correction while preserving TCFormer features."""

    def __init__(self, feature_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.correction = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
            nn.LayerNorm(feature_dim),
        )
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, features: Tensor) -> Tensor:
        return features + self.scale * self.correction(features)


class DomainDiscriminator(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.net(features)


class MultiKernelMMDLoss(nn.Module):
    def __init__(self, kernel_mul: float = 2.0, kernel_num: int = 5):
        super().__init__()
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num

    def forward(self, source: Tensor, target: Tensor) -> Tensor:
        if source.size(0) < 2 or target.size(0) < 2:
            return source.new_zeros(())

        source = F.normalize(source, p=2, dim=1)
        target = F.normalize(target, p=2, dim=1)
        total = torch.cat((source, target), dim=0)
        distances = torch.cdist(total, total, p=2).square()

        sample_count = total.size(0)
        bandwidth = distances.detach().sum()
        bandwidth = bandwidth / max(sample_count * (sample_count - 1), 1)
        bandwidth = bandwidth.clamp_min(1e-6)
        bandwidth = bandwidth / (self.kernel_mul ** (self.kernel_num // 2))

        kernels = sum(
            torch.exp(-distances / (bandwidth * (self.kernel_mul ** idx)))
            for idx in range(self.kernel_num)
        )
        source_count = source.size(0)
        target_count = target.size(0)
        k_ss = kernels[:source_count, :source_count]
        k_tt = kernels[source_count:, source_count:]
        k_st = kernels[:source_count, source_count:]

        source_term = (k_ss.sum() - k_ss.diagonal().sum()) / (
            source_count * (source_count - 1)
        )
        target_term = (k_tt.sum() - k_tt.diagonal().sum()) / (
            target_count * (target_count - 1)
        )
        return source_term + target_term - 2.0 * k_st.mean()


class HADATCFormer(ClassificationModule):
    """HADANet-style UDA applied to the TCFormer representation."""

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        F1: int = 16,
        temp_kernel_lengths: tuple = (16, 32, 64),
        pool_length_1: int = 8,
        pool_length_2: int = 7,
        D: int = 2,
        dropout_conv: float = 0.3,
        d_group: int = 16,
        tcn_depth: int = 2,
        kernel_length_tcn: int = 4,
        dropout_tcn: float = 0.3,
        use_group_attn: bool = True,
        q_heads: int = 8,
        kv_heads: int = 4,
        trans_depth: int = 5,
        trans_dropout: float = 0.4,
        aligner_hidden_dim: int = 128,
        domain_hidden_dim: int = 128,
        adaptation_dropout: float = 0.3,
        adversarial_weight: float = 1.0,
        mmd_weight: float = 0.5,
        **kwargs,
    ):
        model = TCFormerModule(
            n_channels=n_channels,
            n_classes=n_classes,
            F1=F1,
            temp_kernel_lengths=temp_kernel_lengths,
            pool_length_1=pool_length_1,
            pool_length_2=pool_length_2,
            D=D,
            dropout_conv=dropout_conv,
            d_group=d_group,
            tcn_depth=tcn_depth,
            kernel_length_tcn=kernel_length_tcn,
            dropout_tcn=dropout_tcn,
            use_group_attn=use_group_attn,
            q_heads=q_heads,
            kv_heads=kv_heads,
            trans_depth=trans_depth,
            trans_dropout=trans_dropout,
        )
        super().__init__(model, n_classes, **kwargs)
        self.aligner = ResidualFeatureAligner(
            model.feature_dim, aligner_hidden_dim, adaptation_dropout
        )
        self.grl = GradientReversal()
        self.domain_discriminator = DomainDiscriminator(
            model.feature_dim, domain_hidden_dim, adaptation_dropout
        )
        self.mmd_loss = MultiKernelMMDLoss()
        self.adversarial_weight = adversarial_weight
        self.mmd_weight = mmd_weight

    def forward(self, x: Tensor) -> Tensor:
        features = self.aligner(self.model.extract_features(x))
        return self.model.classify_features(features)

    def _grl_alpha(self) -> float:
        progress = self.current_epoch / max(int(self.hparams.max_epochs) - 1, 1)
        return 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0

    def training_step(self, batch, batch_idx):
        if not isinstance(batch, dict) or "source" not in batch or "target" not in batch:
            raise RuntimeError(
                "HADATCFormer requires LOSO UDA batches with 'source' and 'target'. "
                "Run it with --loso and a UDA-enabled config."
            )

        source_x, source_y = batch["source"]
        target_x = batch["target"]
        source_count = source_x.size(0)

        # A shared forward pass also gives BatchNorm both domains without ever
        # reading target labels.
        all_features = self.model.extract_features(torch.cat((source_x, target_x), dim=0))
        all_features = self.aligner(all_features)
        source_features = all_features[:source_count]
        target_features = all_features[source_count:]

        source_logits = self.model.classify_features(source_features)
        classification_loss = F.cross_entropy(source_logits, source_y)

        alpha = self._grl_alpha()
        domain_logits = self.domain_discriminator(self.grl(all_features, alpha))
        domain_targets = torch.cat(
            (
                torch.zeros(source_count, 1, device=all_features.device),
                torch.ones(target_features.size(0), 1, device=all_features.device),
            ),
            dim=0,
        )
        adversarial_loss = F.binary_cross_entropy_with_logits(
            domain_logits, domain_targets
        )
        mmd_loss = self.mmd_loss(source_features, target_features)
        loss = (
            classification_loss
            + self.adversarial_weight * adversarial_loss
            + self.mmd_weight * mmd_loss
        )

        acc = accuracy(
            source_logits, source_y, task="multiclass", num_classes=self.hparams.n_classes
        )
        batch_size = source_count
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log("train_acc", acc, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log("train_cls_loss", classification_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log("train_domain_loss", adversarial_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log("train_mmd_loss", mmd_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log("grl_alpha", alpha, on_step=False, on_epoch=True, batch_size=batch_size)
        return loss

    @staticmethod
    def benchmark(input_shape, device="cuda:0", warmup=100, runs=500):
        return measure_latency(HADATCFormer(22, 4), input_shape, device, warmup, runs)
