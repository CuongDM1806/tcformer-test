"""TCFormer with HADANet-style unsupervised domain adaptation.

Only source labels contribute to classification. Target batches contain EEG
samples only and are used by the adversarial and MK-MMD alignment objectives.
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


class AutoDIALFeatureNorm(nn.Module):
    """Domain alignment layer with a learned mixing weight (AutoDIAL).

    Source features are standardized with source statistics. Target features
    are standardized with the moments of the mixture
    ``alpha * target + (1 - alpha) * source``, where ``alpha`` in [0, 1] is
    learned separately for every feature group. ``alpha -> 1`` recovers fully
    domain-specific normalization (DSBN); ``alpha -> 0`` normalizes the target
    with source statistics. A shared affine transform follows, so no
    parameters are domain specific apart from the mixing weights.
    """

    def __init__(
        self,
        feature_dim: int,
        n_groups: int,
        momentum: float = 0.1,
        eps: float = 1e-5,
        alpha_init: float = 0.5,
    ):
        super().__init__()
        if feature_dim % n_groups != 0:
            raise ValueError("feature_dim must be divisible by n_groups.")
        if not 0.0 < alpha_init < 1.0:
            raise ValueError("autodial_alpha_init must be in (0, 1).")
        self.group_size = feature_dim // n_groups
        self.momentum = momentum
        self.eps = eps
        self.alpha_logit = nn.Parameter(
            torch.full((n_groups,), math.log(alpha_init / (1.0 - alpha_init)))
        )
        self.weight = nn.Parameter(torch.ones(feature_dim))
        self.bias = nn.Parameter(torch.zeros(feature_dim))
        for domain in ("source", "target"):
            self.register_buffer(f"{domain}_mean", torch.zeros(feature_dim))
            self.register_buffer(f"{domain}_var", torch.ones(feature_dim))

    def alpha(self, detach: bool = False) -> Tensor:
        logit = self.alpha_logit.detach() if detach else self.alpha_logit
        return torch.sigmoid(logit)

    def _mix(self, alpha: Tensor, source_mean, source_var, target_mean, target_var):
        a = alpha.repeat_interleave(self.group_size)
        mean = a * target_mean + (1.0 - a) * source_mean
        # Exact second moment of the two-component mixture.
        var = (
            a * target_var
            + (1.0 - a) * source_var
            + a * (1.0 - a) * (target_mean - source_mean).square()
        )
        return mean, var

    def _normalize(self, features: Tensor, mean: Tensor, var: Tensor) -> Tensor:
        standardized = (features - mean) / torch.sqrt(var + self.eps)
        return standardized * self.weight + self.bias

    @torch.no_grad()
    def _update_running(self, domain: str, mean: Tensor, var: Tensor, count: int):
        unbiased = var * count / max(count - 1, 1)
        getattr(self, f"{domain}_mean").lerp_(mean, self.momentum)
        getattr(self, f"{domain}_var").lerp_(unbiased, self.momentum)

    def forward_pair(
        self,
        source: Tensor,
        target: Tensor,
        detach_alpha: bool = False,
        update_stats: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """Training path: batch statistics of both domains, as in BatchNorm."""
        source_mean = source.mean(dim=0)
        source_var = source.var(dim=0, unbiased=False)
        target_mean = target.mean(dim=0)
        target_var = target.var(dim=0, unbiased=False)
        if update_stats:
            self._update_running("source", source_mean, source_var, source.size(0))
            self._update_running("target", target_mean, target_var, target.size(0))
        mixed_mean, mixed_var = self._mix(
            self.alpha(detach_alpha), source_mean, source_var, target_mean, target_var
        )
        return (
            self._normalize(source, source_mean, source_var),
            self._normalize(target, mixed_mean, mixed_var),
        )

    def forward(self, features: Tensor, domain: str) -> Tensor:
        """Inference path: running statistics of the requested domain."""
        if domain == "source":
            return self._normalize(features, self.source_mean, self.source_var)
        mean, var = self._mix(
            self.alpha(detach=True),
            self.source_mean,
            self.source_var,
            self.target_mean,
            self.target_var,
        )
        return self._normalize(features, mean, var)

    @torch.no_grad()
    def set_target_statistics(self, mean: Tensor, var: Tensor) -> None:
        self.target_mean.copy_(mean)
        self.target_var.copy_(var)


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
        sequence_block_types=None,
        mamba_d_state: int = 8,
        mamba_d_conv: int = 3,
        feature_alignment: str = "none",
        autodial_alpha_init: float = 0.5,
        domain_norm_momentum: float = 0.1,
        target_im_weight: float = 0.0,
        recalibrate_target_statistics: bool = True,
        aligner_hidden_dim: int = 128,
        domain_hidden_dim: int = 128,
        adaptation_dropout: float = 0.3,
        adversarial_weight: float = 1.0,
        mmd_weight: float = 0.5,
        temporal_mmd_weight: float = 0.1,
        light_adaptation_factor: float = 0.25,
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
        if feature_alignment not in ("none", "autodial"):
            raise ValueError("feature_alignment must be 'none' or 'autodial'.")
        if target_im_weight < 0.0:
            raise ValueError("target_im_weight must be non-negative.")
        # AutoDIAL normalizes the pooled features before the residual aligner.
        # One mixing weight per feature group: three temporal-kernel groups
        # from the CNN shortcut plus the selective-SSM group.
        self.domain_norm = (
            AutoDIALFeatureNorm(
                model.feature_dim,
                n_groups=model.n_groups + 1,
                momentum=domain_norm_momentum,
                alpha_init=autodial_alpha_init,
            )
            if feature_alignment == "autodial"
            else None
        )
        self.target_im_weight = float(target_im_weight)
        self.recalibrate_target_statistics = bool(recalibrate_target_statistics)
        # Validation scores labeled source trials; every other pass sees target.
        self._inference_domain = "target"
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
        self.temporal_mmd_weight = temporal_mmd_weight
        if not 0.0 < light_adaptation_factor <= 1.0:
            raise ValueError("light_adaptation_factor must be in (0, 1].")
        self.light_adaptation_factor = light_adaptation_factor
        if im_tta_steps < 0:
            raise ValueError("im_tta_steps must be non-negative.")
        if im_tta_lr <= 0.0:
            raise ValueError("im_tta_lr must be positive.")
        if im_tta_diversity_weight < 0.0:
            raise ValueError("im_tta_diversity_weight must be non-negative.")
        self.im_tta_steps = int(im_tta_steps)
        self.im_tta_lr = float(im_tta_lr)
        self.im_tta_diversity_weight = float(im_tta_diversity_weight)
        if log_every_n_batches < 0:
            raise ValueError("log_every_n_batches must be non-negative.")
        self.log_every_n_batches = int(log_every_n_batches)
        self._epoch_started_at = None
        # One LOSO run has one target subject. These EMAs therefore summarize
        # target-level transferability instead of reacting to a single batch.
        self.register_buffer(
            "_target_gap_ema", torch.tensor(float("nan")), persistent=False
        )
        self.register_buffer(
            "_target_confidence_ema", torch.tensor(float("nan")), persistent=False
        )

    def _normalize(self, features: Tensor, domain: str) -> Tensor:
        if self.domain_norm is None:
            return features
        return self.domain_norm(features, domain)

    def forward(self, x: Tensor, domain: str | None = None) -> Tensor:
        features = self.model.extract_features(x)
        features = self._normalize(features, domain or self._inference_domain)
        return self.model.classify_features(self.aligner(features))

    def on_validation_start(self):
        self._inference_domain = "source"

    def on_validation_end(self):
        self._inference_domain = "target"

    def autodial_alpha(self) -> list[float] | None:
        if self.domain_norm is None:
            return None
        return [round(float(a), 4) for a in self.domain_norm.alpha(detach=True)]

    @staticmethod
    def _information_maximization(logits: Tensor) -> Tensor:
        """Mean conditional entropy minus the entropy of the batch marginal."""
        probabilities = logits.softmax(dim=1)
        log_probabilities = probabilities.clamp_min(1e-6).log()
        conditional = -(probabilities * log_probabilities).sum(dim=1).mean()
        marginal = probabilities.mean(dim=0)
        marginal_entropy = -(marginal * marginal.clamp_min(1e-6).log()).sum()
        return conditional - marginal_entropy

    def _grl_alpha(self) -> float:
        progress = self.current_epoch / max(int(self.hparams.max_epochs) - 1, 1)
        return 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0

    @torch.no_grad()
    def _target_adaptation_gate(
        self,
        source_features: Tensor,
        target_features: Tensor,
        target_logits: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return a target-level normal/light DA gate and its diagnostics.

        A target is treated as high-risk when its normalized domain gap is
        larger than the within-domain feature spread while its prediction
        confidence is in the lower half of the range between random guessing
        and certainty. Fixed, dimensionless cutoffs keep tuning to a minimum.
        """
        source = F.normalize(source_features.detach(), p=2, dim=1)
        target = F.normalize(target_features.detach(), p=2, dim=1)
        source_center = source.mean(dim=0)
        target_center = target.mean(dim=0)

        center_distance = torch.linalg.vector_norm(source_center - target_center)
        source_spread = torch.linalg.vector_norm(
            source - source_center, dim=1
        ).mean()
        target_spread = torch.linalg.vector_norm(
            target - target_center, dim=1
        ).mean()
        domain_gap = center_distance / (
            0.5 * (source_spread + target_spread)
        ).clamp_min(1e-6)

        mean_confidence = target_logits.detach().softmax(dim=1).amax(dim=1).mean()
        random_confidence = 1.0 / self.hparams.n_classes
        normalized_confidence = (
            (mean_confidence - random_confidence) / (1.0 - random_confidence)
        ).clamp(0.0, 1.0)

        ema_decay = 0.9
        if torch.isnan(self._target_gap_ema):
            self._target_gap_ema.copy_(domain_gap)
            self._target_confidence_ema.copy_(normalized_confidence)
        else:
            self._target_gap_ema.lerp_(domain_gap, 1.0 - ema_decay)
            self._target_confidence_ema.lerp_(
                normalized_confidence, 1.0 - ema_decay
            )

        use_light_adaptation = (self._target_gap_ema > 1.0) & (
            self._target_confidence_ema < 0.5
        )
        normal_gate = source_features.new_ones(())
        light_gate = source_features.new_tensor(self.light_adaptation_factor)
        gate = torch.where(use_light_adaptation, light_gate, normal_gate)
        return gate, self._target_gap_ema.clone(), self._target_confidence_ema.clone()

    def on_train_epoch_start(self):
        self._epoch_started_at = time.perf_counter()

    def on_train_epoch_end(self):
        epoch = self.current_epoch + 1
        if self.domain_norm is not None and (
            epoch % 10 == 0 or epoch == int(self.hparams.max_epochs)
        ):
            self.print(f"Epoch {epoch} | AutoDIAL alpha: {self.autodial_alpha()}")

    def adapt_to_target(self, target_loader):
        """Label-free adaptation to the target trials after training.

        IM-TTA (optional) updates the backbone BatchNorm affine parameters.
        The AutoDIAL target statistics are then re-estimated in one pass over
        the same unlabeled trials so they match the adapted backbone. Target
        labels may be present in the loader but are never read.
        """
        self._inference_domain = "target"
        stats = self._run_im_tta(target_loader) if self.im_tta_steps > 0 else None
        if self.domain_norm is not None:
            if self.recalibrate_target_statistics:
                self._recalibrate_target_statistics(target_loader)
            self.print(f"AutoDIAL alpha per feature group: {self.autodial_alpha()}")
        self.eval()
        return stats

    @torch.no_grad()
    def _recalibrate_target_statistics(self, target_loader) -> None:
        # eval() keeps dropout off. After IM-TTA the backbone BatchNorm layers
        # use target batch statistics, exactly as in the final test pass.
        self.eval()
        device = next(self.parameters()).device
        feature_sum, square_sum, count = None, None, 0
        for batch in target_loader:
            target_x = batch[0] if isinstance(batch, (tuple, list)) else batch
            features = self.model.extract_features(
                target_x.to(device, non_blocking=True)
            ).double()
            if feature_sum is None:
                feature_sum = features.sum(dim=0)
                square_sum = features.square().sum(dim=0)
            else:
                feature_sum += features.sum(dim=0)
                square_sum += features.square().sum(dim=0)
            count += features.size(0)
        if count < 2:
            raise RuntimeError("AutoDIAL target statistics need two or more trials.")
        mean = feature_sum / count
        var = ((square_sum - count * mean.square()) / (count - 1)).clamp_min(0.0)
        self.domain_norm.set_target_statistics(mean.float(), var.float())
        self.print(f"AutoDIAL target statistics recalibrated | samples={count}")

    def _run_im_tta(self, target_loader):
        """Adapt BatchNorm affine parameters using unlabeled target trials.

        The information-maximization objective sharpens individual target
        predictions while maintaining a diverse batch-level class marginal.
        Target labels may be present in the evaluation loader but are ignored.
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
                        target_x = (
                            batch[0] if isinstance(batch, (tuple, list)) else batch
                        )
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
                    target_x = batch[0] if isinstance(batch, (tuple, list)) else batch
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

        source_pooled = pooled_features[:source_count]
        target_pooled = pooled_features[source_count:]
        target_task_features = None
        if self.domain_norm is not None:
            # Alignment losses use a detached mixing weight: they could always
            # be lowered by alpha -> 1 (full DSBN), which would make alpha a
            # trivial domain-gap minimizer. Alpha is therefore learned only
            # from the task signals below (source CE and target InfoMax), as
            # in AutoDIAL.
            source_pooled, target_aligned_input = self.domain_norm.forward_pair(
                source_pooled, target_pooled, detach_alpha=True
            )
            if self.target_im_weight > 0.0:
                _, target_task_input = self.domain_norm.forward_pair(
                    pooled_features[:source_count],
                    target_pooled,
                    update_stats=False,
                )
                target_task_features = self.aligner(target_task_input)
            target_pooled = target_aligned_input

        all_features = self.aligner(torch.cat((source_pooled, target_pooled), dim=0))
        source_features = all_features[:source_count]
        target_features = all_features[source_count:]

        source_logits = self.model.classify_features(source_features)
        with torch.no_grad():
            target_logits = self.model.classify_features(target_features)
        classification_loss = F.cross_entropy(source_logits, source_y)
        target_im_loss = source_logits.new_zeros(())
        if target_task_features is not None:
            target_im_loss = self._information_maximization(
                self.model.classify_features(target_task_features)
            )

        adaptation_gate, target_gap, target_confidence = (
            self._target_adaptation_gate(
                source_features, target_features, target_logits
            )
        )

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
            + self.target_im_weight * target_im_loss
            + adaptation_gate
            * (
                self.adversarial_weight * adversarial_loss
                + self.mmd_weight * mmd_loss
                + self.temporal_mmd_weight * temporal_mmd_loss
            )
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
        self.log(
            "train_da_gate",
            adaptation_gate,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "train_target_gap",
            target_gap,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "train_target_confidence",
            target_confidence,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "train_temporal_mmd_loss",
            temporal_mmd_loss,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log("grl_alpha", alpha, on_step=False, on_epoch=True, batch_size=batch_size)
        if self.domain_norm is not None:
            self.log("train_target_im_loss", target_im_loss, on_step=False, on_epoch=True, batch_size=batch_size)
            for group, value in enumerate(self.domain_norm.alpha(detach=True)):
                self.log(f"autodial_alpha_g{group}", value, on_step=False, on_epoch=True, batch_size=batch_size)

        total_batches = self.trainer.num_training_batches
        current_batch = batch_idx + 1
        should_print = self.log_every_n_batches > 0 and (
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
            adaptation_mode = (
                "light" if adaptation_gate.detach().item() < 1.0 else "normal"
            )
            self.print(
                f"Epoch {self.current_epoch + 1}/{self.hparams.max_epochs} | "
                f"Batch {batch_progress} | "
                f"loss={loss.detach().item():.4f} | "
                f"cls={classification_loss.detach().item():.4f} | "
                f"domain={adversarial_loss.detach().item():.4f} | "
                f"mmd={mmd_loss.detach().item():.4f} | "
                f"tmmd={temporal_mmd_loss.detach().item():.4f} | "
                f"DA={adaptation_mode}({adaptation_gate.detach().item():.2f}) | "
                f"gap={target_gap.detach().item():.2f} | "
                f"conf={target_confidence.detach().item():.2f} | "
                f"acc={acc.detach().item() * 100:.2f}% | "
                f"elapsed={elapsed / 60:.1f}m | ETA={eta_text}"
            )
        return loss

    @staticmethod
    def benchmark(input_shape, device="cuda:0", warmup=100, runs=500):
        return measure_latency(HADATCFormer(22, 4), input_shape, device, warmup, runs)
