"""TCFormer with gap-adaptive unsupervised domain adaptation.

Only source labels contribute to classification. Target batches contain EEG
samples only and are used by the normalization statistics, the adversarial
objective and MK-MMD.

The domain-adaptation head has three switchable components:

1. ``feature_alignment="group_dsbn"`` standardizes pooled features with
   separate source/target statistics and a shared affine transform. The
   correction therefore scales with the measured domain gap and has no
   domain-specific trainable parameters. ``"residual"`` restores the Lite-DA
   group-wise residual aligner for ablations.
2. ``cross_group_rank > 0`` adds a zero-gated low-rank mixer that lets the
   classifier exchange information between the temporal-kernel groups.
3. ``adaptive_adversary=True`` scales the reversed gradient by the EMA of the
   discriminator's balanced accuracy, so the feature extractor is pushed only
   while the two domains are still separable.
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
    """Apply a lightweight residual correction within each feature group."""

    def __init__(
        self,
        feature_dim: int,
        n_groups: int,
        hidden_dim: int,
        dropout: float,
    ):
        super().__init__()
        if feature_dim % n_groups != 0:
            raise ValueError("feature_dim must be divisible by n_groups.")
        self.n_groups = n_groups
        self.group_dim = feature_dim // n_groups
        self.corrections = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(self.group_dim),
                nn.Linear(self.group_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.group_dim),
                nn.LayerNorm(self.group_dim),
            )
            for _ in range(n_groups)
        )
        self.scales = nn.Parameter(torch.full((n_groups,), 0.1))

    def forward(self, features: Tensor) -> Tensor:
        groups = features.split(self.group_dim, dim=1)
        aligned_groups = [
            group + self.scales[index] * correction(group)
            for index, (group, correction) in enumerate(
                zip(groups, self.corrections)
            )
        ]
        return torch.cat(aligned_groups, dim=1)


class DomainSpecificNorm(nn.Module):
    """Standardize features with separate source and target statistics.

    Statistics are kept per feature channel, so no information crosses the
    grouped classifier's feature groups. The affine transform is shared by
    both domains: the domain correction itself is parameter-free and is as
    large as the measured difference between the two sets of statistics.
    """

    DOMAINS = ("source", "target")

    def __init__(self, feature_dim: int, momentum: float = 0.1, eps: float = 1e-5):
        super().__init__()
        if not 0.0 < momentum <= 1.0:
            raise ValueError("domain_norm_momentum must be in (0, 1].")
        self.momentum = momentum
        self.eps = eps
        for domain in self.DOMAINS:
            self.register_buffer(f"{domain}_mean", torch.zeros(feature_dim))
            self.register_buffer(f"{domain}_var", torch.ones(feature_dim))
        self.weight = nn.Parameter(torch.ones(feature_dim))
        self.bias = nn.Parameter(torch.zeros(feature_dim))

    def _buffers_for(self, domain: str) -> tuple[Tensor, Tensor]:
        if domain not in self.DOMAINS:
            raise ValueError(f"Unknown domain {domain!r}; expected {self.DOMAINS}.")
        return getattr(self, f"{domain}_mean"), getattr(self, f"{domain}_var")

    def standardize(self, features: Tensor, domain: str) -> Tensor:
        running_mean, running_var = self._buffers_for(domain)
        if self.training and features.size(0) > 1:
            mean = features.mean(dim=0)
            var = features.var(dim=0, unbiased=False)
            with torch.no_grad():
                running_mean.lerp_(mean.detach(), self.momentum)
                running_var.lerp_(
                    features.detach().var(dim=0, unbiased=True), self.momentum
                )
        else:
            mean, var = running_mean, running_var
        return (features - mean) / torch.sqrt(var + self.eps)

    def affine(self, standardized: Tensor) -> Tensor:
        return standardized * self.weight + self.bias

    def forward(self, features: Tensor, domain: str) -> Tensor:
        return self.affine(self.standardize(features, domain))

    @torch.no_grad()
    def set_statistics(self, domain: str, mean: Tensor, var: Tensor) -> None:
        running_mean, running_var = self._buffers_for(domain)
        running_mean.copy_(mean)
        running_var.copy_(var)


class LowRankCrossGroupMixer(nn.Module):
    """Zero-gated low-rank exchange of information between feature groups."""

    def __init__(self, feature_dim: int, rank: int):
        super().__init__()
        self.down = nn.Linear(feature_dim, rank, bias=False)
        self.up = nn.Linear(rank, feature_dim, bias=False)
        # The mixer starts as the identity; the gate opens only if the
        # source classification loss benefits from cross-group information.
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, features: Tensor) -> Tensor:
        return features + self.gate * self.up(self.down(features))


class DomainDiscriminator(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
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
    """Gap-adaptive UDA applied to the TCFormer representation."""

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
        sequence_block_types=None,
        mamba_d_state: int = 8,
        mamba_d_conv: int = 3,
        feature_alignment: str = "group_dsbn",
        domain_norm_momentum: float = 0.1,
        recalibrate_target_statistics: bool = True,
        cross_group_rank: int = 0,
        aligner_hidden_dim: int = 8,
        domain_hidden_dim: int = 32,
        adaptation_dropout: float = 0.3,
        adversarial_weight: float = 1.0,
        adaptive_adversary: bool = True,
        adversary_ema_decay: float = 0.95,
        mmd_weight: float = 0.5,
        im_tta_steps: int = 0,
        im_tta_lr: float = 1e-4,
        im_tta_diversity_weight: float = 1.0,
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
            sequence_block_types=sequence_block_types,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
        )
        super().__init__(model, n_classes, **kwargs)

        if feature_alignment == "group_dsbn":
            self.domain_norm = DomainSpecificNorm(
                model.feature_dim, momentum=domain_norm_momentum
            )
            self.aligner = None
        elif feature_alignment == "residual":
            self.domain_norm = None
            self.aligner = ResidualFeatureAligner(
                model.feature_dim,
                model.n_groups + 1,
                aligner_hidden_dim,
                adaptation_dropout,
            )
        else:
            raise ValueError(
                "feature_alignment must be 'group_dsbn' or 'residual', "
                f"got {feature_alignment!r}."
            )
        if cross_group_rank < 0:
            raise ValueError("cross_group_rank must be non-negative.")
        self.cross_group_mixer = (
            LowRankCrossGroupMixer(model.feature_dim, int(cross_group_rank))
            if cross_group_rank > 0
            else None
        )
        self.recalibrate_target_statistics = bool(recalibrate_target_statistics)

        self.grl = GradientReversal()
        self.domain_discriminator = DomainDiscriminator(
            model.feature_dim, domain_hidden_dim, adaptation_dropout
        )
        self.mmd_loss = MultiKernelMMDLoss()
        self.adversarial_weight = adversarial_weight
        self.mmd_weight = mmd_weight
        self.adaptive_adversary = bool(adaptive_adversary)
        if not 0.0 <= adversary_ema_decay < 1.0:
            raise ValueError("adversary_ema_decay must be in [0, 1).")
        self.adversary_ema_decay = float(adversary_ema_decay)

        if im_tta_steps < 0:
            raise ValueError("im_tta_steps must be non-negative.")
        if im_tta_lr <= 0.0:
            raise ValueError("im_tta_lr must be positive.")
        if im_tta_diversity_weight < 0.0:
            raise ValueError("im_tta_diversity_weight must be non-negative.")
        self.im_tta_steps = int(im_tta_steps)
        self.im_tta_lr = float(im_tta_lr)
        self.im_tta_diversity_weight = float(im_tta_diversity_weight)
        # 0 disables per-batch progress lines.
        self.log_every_n_batches = max(0, int(log_every_n_batches))
        self._epoch_started_at = None
        # Validation scores labeled source trials; every other inference path
        # (test, IM-TTA, prediction) sees the held-out target subject.
        self._inference_domain = "target"
        # One LOSO run has one target subject, so this EMA summarizes how
        # separable that subject still is from the source pool.
        self.register_buffer(
            "_domain_acc_ema", torch.tensor(float("nan")), persistent=False
        )

    # ------------------------------------------------------------------ #
    # Feature path
    def _align(self, features: Tensor, domain: str) -> Tensor:
        if self.domain_norm is not None:
            return self.domain_norm(features, domain)
        return self.aligner(features)

    def _classify(self, aligned_features: Tensor) -> Tensor:
        if self.cross_group_mixer is not None:
            aligned_features = self.cross_group_mixer(aligned_features)
        return self.model.classify_features(aligned_features)

    def forward(self, x: Tensor, domain: str | None = None) -> Tensor:
        domain = domain or self._inference_domain
        return self._classify(self._align(self.model.extract_features(x), domain))

    def on_validation_start(self):
        self._inference_domain = "source"

    def on_validation_end(self):
        self._inference_domain = "target"

    # ------------------------------------------------------------------ #
    # Adversary schedule
    def _grl_alpha(self) -> float:
        progress = self.current_epoch / max(int(self.hparams.max_epochs) - 1, 1)
        return 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0

    def _adversary_strength(self) -> float:
        """Map discriminator balanced accuracy 0.5 -> 0 and 1.0 -> 1."""
        if not self.adaptive_adversary:
            return 1.0
        if torch.isnan(self._domain_acc_ema):
            return 0.0
        return float(((self._domain_acc_ema - 0.5) / 0.5).clamp(0.0, 1.0))

    @torch.no_grad()
    def _update_domain_accuracy(self, domain_logits: Tensor, source_count: int) -> Tensor:
        predicted_target = domain_logits.squeeze(1) > 0
        source_acc = (~predicted_target[:source_count]).float().mean()
        target_acc = predicted_target[source_count:].float().mean()
        balanced_acc = 0.5 * (source_acc + target_acc)
        if torch.isnan(self._domain_acc_ema):
            self._domain_acc_ema.copy_(balanced_acc)
        else:
            self._domain_acc_ema.lerp_(balanced_acc, 1.0 - self.adversary_ema_decay)
        return balanced_acc

    def on_train_epoch_start(self):
        self._epoch_started_at = time.perf_counter()

    # ------------------------------------------------------------------ #
    # Target-side adaptation after training
    @staticmethod
    def _unpack_unlabeled_target(batch):
        if isinstance(batch, Tensor):
            return batch
        if isinstance(batch, (tuple, list)) and len(batch) == 1:
            return batch[0]
        raise RuntimeError(
            "Target adaptation requires an EEG-only loader; target labels must "
            "not be present."
        )

    def adapt_to_target(self, target_loader):
        """Label-free adaptation to the held-out target trials.

        IM-TTA (optional) updates BatchNorm affine parameters, then the target
        statistics of the domain-specific normalization are re-estimated in a
        single pass over the same unlabeled trials so they match the final
        backbone. The loader exposes EEG only and cannot carry target labels.
        """
        self._inference_domain = "target"
        stats = self._run_im_tta(target_loader) if self.im_tta_steps > 0 else None
        if self.domain_norm is not None and self.recalibrate_target_statistics:
            self._recalibrate_target_statistics(target_loader)
        self.eval()
        return stats

    @torch.no_grad()
    def _recalibrate_target_statistics(self, target_loader) -> None:
        # eval() keeps dropout off. After IM-TTA the backbone BatchNorm layers
        # have no running buffers and keep using target batch statistics,
        # exactly as they will during the final test pass.
        self.eval()
        device = next(self.parameters()).device
        feature_sum = None
        feature_square_sum = None
        sample_count = 0
        for batch in target_loader:
            target_x = self._unpack_unlabeled_target(batch).to(device, non_blocking=True)
            features = self.model.extract_features(target_x).double()
            batch_sum = features.sum(dim=0)
            batch_square_sum = features.square().sum(dim=0)
            if feature_sum is None:
                feature_sum, feature_square_sum = batch_sum, batch_square_sum
            else:
                feature_sum += batch_sum
                feature_square_sum += batch_square_sum
            sample_count += target_x.size(0)
        if sample_count < 2:
            raise RuntimeError("Target statistics need at least two target trials.")

        mean = feature_sum / sample_count
        var = (feature_square_sum - sample_count * mean.square()) / (sample_count - 1)
        previous_mean = self.domain_norm.target_mean.clone()
        self.domain_norm.set_statistics(
            "target", mean.float(), var.clamp_min(0.0).float()
        )
        shift = torch.linalg.vector_norm(mean.float() - previous_mean).item()
        source_target_gap = torch.linalg.vector_norm(
            self.domain_norm.source_mean - self.domain_norm.target_mean
        ).item()
        self.print(
            f"Target DSBN statistics recalibrated | samples={sample_count} | "
            f"mean_shift_vs_train={shift:.4f} | "
            f"source_target_mean_gap={source_target_gap:.4f}"
        )

    def _run_im_tta(self, target_loader):
        """Adapt BatchNorm affine parameters using unlabeled target trials.

        The information-maximization objective sharpens individual target
        predictions while maintaining a diverse batch-level class marginal.
        """
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

        batch_norm_modules = []
        adaptation_parameters = []
        for module in self.model.modules():
            if not isinstance(module, nn.modules.batchnorm._BatchNorm):
                continue
            if not module.affine:
                continue
            module.train()
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None
            module.weight.requires_grad_(True)
            module.bias.requires_grad_(True)
            batch_norm_modules.append(module)
            adaptation_parameters.extend((module.weight, module.bias))

        if not adaptation_parameters:
            raise RuntimeError(
                "IM-TTA requires at least one affine BatchNorm layer in TCFormer."
            )

        optimizer = torch.optim.Adam(adaptation_parameters, lr=self.im_tta_lr)
        parameter_count = sum(parameter.numel() for parameter in adaptation_parameters)
        self.print(
            f"IM-TTA start | steps={self.im_tta_steps} | lr={self.im_tta_lr:g} | "
            f"BN_layers={len(batch_norm_modules)} | trainable_params={parameter_count}"
        )

        device = next(self.parameters()).device
        final_stats = None
        with torch.enable_grad():
            for step in range(1, self.im_tta_steps + 1):
                # First pass estimates the class marginal over the complete
                # target subject. Dropout remains disabled and BatchNorm has no
                # running buffers, so this pass does not mutate model state.
                probability_sum = None
                sample_count = 0
                with torch.no_grad():
                    for batch in target_loader:
                        target_x = self._unpack_unlabeled_target(batch)
                        target_x = target_x.to(device, non_blocking=True)
                        probabilities = self(target_x).softmax(dim=1)
                        batch_sum = probabilities.sum(dim=0)
                        probability_sum = (
                            batch_sum
                            if probability_sum is None
                            else probability_sum + batch_sum
                        )
                        sample_count += target_x.size(0)

                if sample_count == 0:
                    raise RuntimeError("IM-TTA received an empty target loader.")
                mean_probability = probability_sum / sample_count
                marginal_entropy = -(
                    mean_probability * mean_probability.clamp_min(1e-6).log()
                ).sum()
                marginal_gradient = mean_probability.clamp_min(1e-6).log() + 1.0

                # Accumulate the exact full-target InfoMax gradient and update
                # BatchNorm affine parameters once per target pass.
                optimizer.zero_grad(set_to_none=True)
                conditional_sum = 0.0
                second_pass_count = 0
                for batch in target_loader:
                    target_x = self._unpack_unlabeled_target(batch)
                    target_x = target_x.to(device, non_blocking=True)
                    logits = self.forward(target_x)
                    probabilities = logits.softmax(dim=1)
                    log_probabilities = probabilities.clamp_min(1e-6).log()

                    conditional_entropy = -(
                        probabilities * log_probabilities
                    ).sum(dim=1).mean()
                    batch_size = target_x.size(0)
                    batch_fraction = batch_size / sample_count
                    diversity_surrogate = (
                        probabilities * marginal_gradient.unsqueeze(0)
                    ).sum(dim=1).mean()
                    loss = batch_fraction * (
                        conditional_entropy
                        + self.im_tta_diversity_weight * diversity_surrogate
                    )
                    loss.backward()
                    conditional_sum += conditional_entropy.detach().item() * batch_size
                    second_pass_count += batch_size

                if second_pass_count != sample_count:
                    raise RuntimeError("IM-TTA target loader changed between passes.")
                optimizer.step()
                conditional_entropy_value = conditional_sum / sample_count
                loss_value = (
                    conditional_entropy_value
                    - self.im_tta_diversity_weight * marginal_entropy.item()
                )
                final_stats = {
                    "loss": loss_value,
                    "conditional_entropy": conditional_entropy_value,
                    "marginal_entropy": marginal_entropy.item(),
                    "samples": sample_count,
                }
                self.print(
                    f"IM-TTA step {step}/{self.im_tta_steps} | "
                    f"loss={final_stats['loss']:.4f} | "
                    f"cond_entropy={final_stats['conditional_entropy']:.4f} | "
                    f"marg_entropy={final_stats['marginal_entropy']:.4f} | "
                    f"target_samples={sample_count}"
                )

        # Keep dropout disabled for evaluation. BatchNorm continues to use
        # target batch statistics because its running buffers are disabled.
        self.eval()
        return final_stats

    # ------------------------------------------------------------------ #
    # Training
    def training_step(self, batch, batch_idx):
        if not isinstance(batch, dict) or "source" not in batch or "target" not in batch:
            raise RuntimeError(
                "HADATCFormer requires LOSO UDA batches with 'source' and 'target'. "
                "Run it with --loso and a UDA-enabled config."
            )

        source_x, source_y = batch["source"]
        target_x = batch["target"]
        source_count = source_x.size(0)

        # A shared forward pass also gives the backbone BatchNorm both domains
        # without ever reading target labels.
        temporal_features = self.model.extract_temporal_features(
            torch.cat((source_x, target_x), dim=0)
        )
        pooled_features = self.model.tcn_head.pool_temporal_features(temporal_features)
        source_pooled = pooled_features[:source_count]
        target_pooled = pooled_features[source_count:]

        if self.domain_norm is not None:
            # Each domain is standardized with its own statistics. Alignment
            # losses see the parameter-free standardized features, so the
            # shared affine transform cannot shrink features to fool the
            # discriminator.
            source_standardized = self.domain_norm.standardize(source_pooled, "source")
            target_standardized = self.domain_norm.standardize(target_pooled, "target")
            source_aligned = self.domain_norm.affine(source_standardized)
            discriminator_features = torch.cat(
                (source_standardized, target_standardized), dim=0
            )
            source_mmd_features = source_standardized
            target_mmd_features = target_standardized
        else:
            # Lite-DA: residual aligner for classification/MMD, discriminator
            # on the pooled backbone features.
            source_aligned = self.aligner(source_pooled)
            source_mmd_features = source_aligned
            target_mmd_features = self.aligner(target_pooled)
            discriminator_features = pooled_features

        source_logits = self._classify(source_aligned)
        classification_loss = F.cross_entropy(source_logits, source_y)

        alpha = self._grl_alpha()
        adversary_strength = self._adversary_strength()
        # The discriminator always learns with full weight, so its accuracy
        # stays an honest separability estimate. Only the reversed gradient
        # that reaches the feature extractor is scaled.
        reversal_coefficient = alpha * adversary_strength
        domain_logits = self.domain_discriminator(
            self.grl(discriminator_features, reversal_coefficient)
        )
        domain_targets = torch.cat(
            (
                torch.zeros(source_count, 1, device=pooled_features.device),
                torch.ones(target_pooled.size(0), 1, device=pooled_features.device),
            ),
            dim=0,
        )
        adversarial_loss = F.binary_cross_entropy_with_logits(
            domain_logits, domain_targets
        )
        domain_acc = self._update_domain_accuracy(domain_logits, source_count)
        mmd_loss = self.mmd_loss(source_mmd_features, target_mmd_features)
        loss = (
            classification_loss
            + self.adversarial_weight * adversarial_loss
            + self.mmd_weight * mmd_loss
        )

        acc = accuracy(
            source_logits, source_y, task="multiclass", num_classes=self.hparams.n_classes
        )
        batch_size = source_count
        log = dict(on_step=False, on_epoch=True, batch_size=batch_size)
        self.log("train_loss", loss, prog_bar=True, **log)
        self.log("train_acc", acc, prog_bar=True, **log)
        self.log("train_cls_loss", classification_loss, **log)
        self.log("train_domain_loss", adversarial_loss, **log)
        self.log("train_mmd_loss", mmd_loss, **log)
        self.log("train_domain_acc", domain_acc, **log)
        self.log("train_adversary_strength", adversary_strength, **log)
        self.log("grl_alpha", alpha, **log)
        if self.cross_group_mixer is not None:
            self.log("train_mixer_gate", self.cross_group_mixer.gate.detach(), **log)

        if self.log_every_n_batches > 0:
            self._print_progress(
                batch_idx,
                loss=loss,
                classification_loss=classification_loss,
                adversarial_loss=adversarial_loss,
                mmd_loss=mmd_loss,
                domain_acc=domain_acc,
                adversary_strength=adversary_strength,
                acc=acc,
            )
        return loss

    def _print_progress(
        self,
        batch_idx,
        loss,
        classification_loss,
        adversarial_loss,
        mmd_loss,
        domain_acc,
        adversary_strength,
        acc,
    ):
        total_batches = self.trainer.num_training_batches
        current_batch = batch_idx + 1
        should_print = (
            current_batch == 1
            or current_batch % self.log_every_n_batches == 0
            or current_batch == total_batches
        )
        if not should_print:
            return
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
        mixer_text = (
            f"mix_gate={self.cross_group_mixer.gate.item():.3f} | "
            if self.cross_group_mixer is not None
            else ""
        )
        self.print(
            f"Epoch {self.current_epoch + 1}/{self.hparams.max_epochs} | "
            f"Batch {batch_progress} | "
            f"loss={loss.detach().item():.4f} | "
            f"cls={classification_loss.detach().item():.4f} | "
            f"domain={adversarial_loss.detach().item():.4f} | "
            f"mmd={mmd_loss.detach().item():.4f} | "
            f"D_acc={domain_acc.item():.2f} | "
            f"adv={adversary_strength:.2f} | "
            f"{mixer_text}"
            f"acc={acc.detach().item() * 100:.2f}% | "
            f"elapsed={elapsed / 60:.1f}m | ETA={eta_text}"
        )

    @staticmethod
    def benchmark(input_shape, device="cuda:0", warmup=100, runs=500):
        return measure_latency(HADATCFormer(22, 4), input_shape, device, warmup, runs)
