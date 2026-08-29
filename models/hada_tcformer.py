"""TCFormer with HADANet-style and GPL-style domain adaptation.

Only source labels contribute to classification. Target batches contain EEG
samples only. MK-MMD aligns both domains globally, while two prototype memory
banks add entropy-aware, class-conditional alignment without target labels.
"""

import math
import time

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


class PrototypeMemoryBank(nn.Module):
    """EMA class prototypes for the source and unlabeled target domains.

    The banks are buffers rather than parameters. Target entries are updated
    only with entropy-filtered pseudo-labels, so ground-truth target labels are
    never required or consumed during adaptation.
    """

    def __init__(self, n_classes: int, feature_dim: int, momentum: float = 0.9):
        super().__init__()
        if not 0.0 <= momentum < 1.0:
            raise ValueError("prototype_momentum must be in [0, 1)")
        self.n_classes = n_classes
        self.momentum = momentum
        self.register_buffer("source_prototypes", torch.zeros(n_classes, feature_dim))
        self.register_buffer("target_prototypes", torch.zeros(n_classes, feature_dim))
        self.register_buffer("source_initialized", torch.zeros(n_classes, dtype=torch.bool))
        self.register_buffer("target_initialized", torch.zeros(n_classes, dtype=torch.bool))

    @torch.no_grad()
    def update(
        self,
        domain: str,
        features: Tensor,
        labels: Tensor,
        weights: Tensor | None = None,
    ) -> None:
        if domain not in {"source", "target"}:
            raise ValueError("domain must be 'source' or 'target'")
        prototypes = getattr(self, f"{domain}_prototypes")
        initialized = getattr(self, f"{domain}_initialized")
        features = F.normalize(features.detach(), p=2, dim=1)

        for class_idx in labels.unique():
            class_id = int(class_idx.item())
            mask = labels == class_id
            class_features = features[mask]
            if class_features.numel() == 0:
                continue
            if weights is None:
                class_mean = class_features.mean(dim=0)
            else:
                class_weights = weights[mask].detach().clamp_min(1e-6)
                class_mean = (class_features * class_weights[:, None]).sum(dim=0)
                class_mean = class_mean / class_weights.sum()
            class_mean = F.normalize(class_mean, p=2, dim=0)
            if initialized[class_id]:
                class_mean = self.momentum * prototypes[class_id] + (
                    1.0 - self.momentum
                ) * class_mean
                class_mean = F.normalize(class_mean, p=2, dim=0)
            prototypes[class_id].copy_(class_mean)
            initialized[class_id] = True

    @staticmethod
    def contrastive_loss(
        features: Tensor,
        labels: Tensor,
        prototypes: Tensor,
        initialized: Tensor,
        temperature: float,
        sample_weights: Tensor | None = None,
    ) -> Tensor:
        """Classify normalized features against the available prototypes."""
        valid_samples = initialized[labels]
        if initialized.sum() < 2 or not valid_samples.any():
            return features.new_zeros(())

        features = F.normalize(features[valid_samples], p=2, dim=1)
        logits = features @ F.normalize(prototypes, p=2, dim=1).T
        logits = logits / temperature
        logits = logits.masked_fill(~initialized.unsqueeze(0), -1e4)
        per_sample = F.cross_entropy(logits, labels[valid_samples], reduction="none")
        if sample_weights is None:
            return per_sample.mean()
        selected_weights = sample_weights[valid_samples].clamp_min(1e-6)
        return (per_sample * selected_weights).sum() / selected_weights.sum()


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
        temporal_mmd_weight: float = 0.1,
        source_prototype_weight: float = 0.1,
        interactive_prototype_weight: float = 0.1,
        prototype_momentum: float = 0.9,
        prototype_temperature: float = 0.1,
        prototype_warmup_epochs: int = 5,
        target_entropy_threshold_start: float = 0.2,
        target_entropy_threshold_end: float = 0.6,
        log_every_n_batches: int = 5,
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
        self.prototype_bank = PrototypeMemoryBank(
            n_classes, model.feature_dim, prototype_momentum
        )
        self.adversarial_weight = adversarial_weight
        self.mmd_weight = mmd_weight
        self.temporal_mmd_weight = temporal_mmd_weight
        self.source_prototype_weight = source_prototype_weight
        self.interactive_prototype_weight = interactive_prototype_weight
        if prototype_temperature <= 0.0:
            raise ValueError("prototype_temperature must be positive")
        if not 0.0 <= target_entropy_threshold_start <= 1.0:
            raise ValueError("target_entropy_threshold_start must be in [0, 1]")
        if not 0.0 <= target_entropy_threshold_end <= 1.0:
            raise ValueError("target_entropy_threshold_end must be in [0, 1]")
        self.prototype_temperature = prototype_temperature
        self.prototype_warmup_epochs = max(0, int(prototype_warmup_epochs))
        self.target_entropy_threshold_start = target_entropy_threshold_start
        self.target_entropy_threshold_end = target_entropy_threshold_end
        self.log_every_n_batches = max(1, int(log_every_n_batches))
        self._epoch_started_at = None

    def forward(self, x: Tensor) -> Tensor:
        features = self.aligner(self.model.extract_features(x))
        return self.model.classify_features(features)

    def _grl_alpha(self) -> float:
        progress = self.current_epoch / max(int(self.hparams.max_epochs) - 1, 1)
        return 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0

    def _target_entropy_gate(self, target_logits: Tensor):
        """Return pseudo-labels, confidence weights, and GPL-style selection."""
        probabilities = target_logits.detach().softmax(dim=1)
        entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=1)
        normalized_entropy = entropy / math.log(self.hparams.n_classes)
        confidence = (1.0 - normalized_entropy).clamp(0.0, 1.0)

        adaptation_epochs = max(
            int(self.hparams.max_epochs) - self.prototype_warmup_epochs - 1, 1
        )
        progress = max(self.current_epoch - self.prototype_warmup_epochs, 0)
        progress = min(progress / adaptation_epochs, 1.0)
        threshold = self.target_entropy_threshold_start + progress * (
            self.target_entropy_threshold_end
            - self.target_entropy_threshold_start
        )
        selected = normalized_entropy <= threshold
        pseudo_labels = probabilities.argmax(dim=1)
        return pseudo_labels, confidence, selected, normalized_entropy.mean(), threshold

    def on_train_epoch_start(self):
        self._epoch_started_at = time.perf_counter()

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
        temporal_features = self.model.extract_temporal_features(
            torch.cat((source_x, target_x), dim=0)
        )
        pooled_features = self.model.tcn_head.pool_temporal_features(temporal_features)

        # Auxiliary alignment before temporal compression. It sees information
        # from every TCN time position and introduces no trainable parameters.
        temporal_mean_features = temporal_features.mean(dim=-1)
        source_temporal_mean = temporal_mean_features[:source_count]
        target_temporal_mean = temporal_mean_features[source_count:]
        temporal_mmd_loss = self.mmd_loss(
            source_temporal_mean, target_temporal_mean
        )

        all_features = pooled_features
        all_features = self.aligner(all_features)
        source_features = all_features[:source_count]
        target_features = all_features[source_count:]

        source_logits = self.model.classify_features(source_features)
        target_logits = self.model.classify_features(target_features)
        classification_loss = F.cross_entropy(source_logits, source_y)

        # GPL local alignment: the source bank is supervised by real source
        # labels. The target bank sees only entropy-filtered pseudo-labels.
        self.prototype_bank.update("source", source_features, source_y)
        source_prototype_loss = self.prototype_bank.contrastive_loss(
            source_features,
            source_y,
            self.prototype_bank.source_prototypes,
            self.prototype_bank.source_initialized,
            self.prototype_temperature,
        )
        pseudo_labels, target_confidence, target_selected, mean_target_entropy, entropy_threshold = (
            self._target_entropy_gate(target_logits)
        )
        prototype_adaptation_active = self.current_epoch >= self.prototype_warmup_epochs
        if not prototype_adaptation_active:
            target_selected = torch.zeros_like(target_selected)
        if prototype_adaptation_active and target_selected.any():
            self.prototype_bank.update(
                "target",
                target_features[target_selected],
                pseudo_labels[target_selected],
                target_confidence[target_selected],
            )
            target_to_source_loss = self.prototype_bank.contrastive_loss(
                target_features[target_selected],
                pseudo_labels[target_selected],
                self.prototype_bank.source_prototypes,
                self.prototype_bank.source_initialized,
                self.prototype_temperature,
                target_confidence[target_selected],
            )
            source_to_target_loss = self.prototype_bank.contrastive_loss(
                source_features,
                source_y,
                self.prototype_bank.target_prototypes,
                self.prototype_bank.target_initialized,
                self.prototype_temperature,
            )
            interactive_prototype_loss = 0.5 * (
                target_to_source_loss + source_to_target_loss
            )
        else:
            interactive_prototype_loss = source_features.new_zeros(())

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
            + self.temporal_mmd_weight * temporal_mmd_loss
            + self.source_prototype_weight * source_prototype_loss
            + self.interactive_prototype_weight * interactive_prototype_loss
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
        self.log("train_source_prototype_loss", source_prototype_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log("train_interactive_prototype_loss", interactive_prototype_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log("train_target_confident_ratio", target_selected.float().mean(), on_step=False, on_epoch=True, batch_size=batch_size)
        self.log("train_target_entropy", mean_target_entropy, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log(
            "train_temporal_mmd_loss",
            temporal_mmd_loss,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log("grl_alpha", alpha, on_step=False, on_epoch=True, batch_size=batch_size)

        total_batches = self.trainer.num_training_batches
        current_batch = batch_idx + 1
        should_print = (
            current_batch == 1
            or current_batch % self.log_every_n_batches == 0
            or current_batch == total_batches
        )
        if should_print:
            if self._epoch_started_at is None:
                self._epoch_started_at = time.perf_counter()
            elapsed = time.perf_counter() - self._epoch_started_at
            seconds_per_batch = elapsed / current_batch
            if isinstance(total_batches, int):
                eta_seconds = seconds_per_batch * max(total_batches - current_batch, 0)
                batch_progress = f"{current_batch}/{total_batches}"
                eta_text = f"{eta_seconds / 60:.1f}m"
            else:
                batch_progress = f"{current_batch}/?"
                eta_text = "?"
            self.print(
                f"Epoch {self.current_epoch + 1}/{self.hparams.max_epochs} | "
                f"Batch {batch_progress} | "
                f"loss={loss.detach().item():.4f} | "
                f"cls={classification_loss.detach().item():.4f} | "
                f"domain={adversarial_loss.detach().item():.4f} | "
                f"mmd={mmd_loss.detach().item():.4f} | "
                f"tmmd={temporal_mmd_loss.detach().item():.4f} | "
                f"sproto={source_prototype_loss.detach().item():.4f} | "
                f"xproto={interactive_prototype_loss.detach().item():.4f} | "
                f"tkeep={target_selected.float().mean().detach().item() * 100:.1f}% "
                f"(H<={entropy_threshold:.2f}) | "
                f"acc={acc.detach().item() * 100:.2f}% | "
                f"elapsed={elapsed / 60:.1f}m | ETA={eta_text}"
            )
        return loss

    @staticmethod
    def benchmark(input_shape, device="cuda:0", warmup=100, runs=500):
        return measure_latency(HADATCFormer(22, 4), input_shape, device, warmup, runs)
