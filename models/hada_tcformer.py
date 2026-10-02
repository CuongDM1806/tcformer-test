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


class _GradientScale(torch.autograd.Function):
    """Identity in the forward pass; scales the gradient by ``scale``."""

    @staticmethod
    def forward(ctx, x: Tensor, scale: float) -> Tensor:
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        return ctx.scale * grad_output, None


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

    def correction_term(self, features: Tensor) -> Tensor:
        return self.scale * self.correction(features)

    def forward(self, features: Tensor) -> Tensor:
        return features + self.correction_term(features)


class GroupResidualAligner(nn.Module):
    """Group-preserving residual correction (Lite-DA style).

    Each feature group gets its own small residual MLP, so this path cannot
    mix frequency bands. The per-group scales start at zero: the model starts
    exactly at the Full-Mamba aligner and opens this path only if useful.
    """

    def __init__(self, feature_dim: int, n_groups: int, hidden_dim: int, dropout: float):
        super().__init__()
        if feature_dim % n_groups != 0:
            raise ValueError("feature_dim must be divisible by n_groups.")
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
        self.scales = nn.Parameter(torch.zeros(n_groups))

    def correction_term(self, features: Tensor) -> Tensor:
        groups = features.split(self.group_dim, dim=1)
        return torch.cat(
            [
                self.scales[index] * correction(group)
                for index, (group, correction) in enumerate(
                    zip(groups, self.corrections)
                )
            ],
            dim=1,
        )


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
        aligner_hidden_dim: int = 128,
        domain_hidden_dim: int = 128,
        adaptation_dropout: float = 0.3,
        adversarial_weight: float = 1.0,
        mmd_weight: float = 0.5,
        temporal_mmd_weight: float = 0.1,
        light_adaptation_factor: float = 0.25,
        grl_routing: str = "none",
        grl_aligner_rho: float = 1.0,
        grl_routing_tau: float = 1.0,
        group_aligner: bool = False,
        group_aligner_hidden_dim: int = 8,
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
        self.aligner = ResidualFeatureAligner(
            model.feature_dim, aligner_hidden_dim, adaptation_dropout
        )
        self.group_aligner = (
            GroupResidualAligner(
                model.feature_dim,
                model.n_groups + 1,
                group_aligner_hidden_dim,
                adaptation_dropout,
            )
            if group_aligner
            else None
        )
        if grl_routing not in ("none", "fixed", "gap"):
            raise ValueError("grl_routing must be 'none', 'fixed' or 'gap'.")
        if not 0.0 <= grl_aligner_rho <= 1.0:
            raise ValueError("grl_aligner_rho must be in [0, 1].")
        if grl_routing_tau <= 0.0:
            raise ValueError("grl_routing_tau must be positive.")
        self.grl_routing = grl_routing
        self.grl_aligner_rho = float(grl_aligner_rho)
        self.grl_routing_tau = float(grl_routing_tau)
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
        self.register_buffer(
            "_backbone_gap_ema", torch.tensor(float("nan")), persistent=False
        )
        self._epoch_diagnostics = {}

    def _align(self, features: Tensor) -> tuple[Tensor, Tensor, Tensor | None]:
        """Return aligned features and the global/group correction terms."""
        global_correction = self.aligner.correction_term(features)
        group_correction = (
            self.group_aligner.correction_term(features)
            if self.group_aligner is not None
            else None
        )
        aligned = features + global_correction
        if group_correction is not None:
            aligned = aligned + group_correction
        return aligned, global_correction, group_correction

    def forward(self, x: Tensor) -> Tensor:
        features, _, _ = self._align(self.model.extract_features(x))
        return self.model.classify_features(features)

    @staticmethod
    def _normalized_gap(source_features: Tensor, target_features: Tensor) -> Tensor:
        """Center distance over mean within-domain spread, on L2-normalized features."""
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
        return center_distance / (
            0.5 * (source_spread + target_spread)
        ).clamp_min(1e-6)

    @torch.no_grad()
    def _aligner_grl_share(
        self, source_pooled: Tensor, target_pooled: Tensor
    ) -> float:
        """Share rho of the reversed gradient that reaches the global aligner.

        The gap is measured on the backbone features, before the aligner, so
        the aligner cannot lower its own adversarial pressure by aligning.
        """
        gap = self._normalized_gap(source_pooled, target_pooled)
        if torch.isnan(self._backbone_gap_ema):
            self._backbone_gap_ema.copy_(gap)
        else:
            self._backbone_gap_ema.lerp_(gap, 0.1)
        if self.grl_routing == "none":
            return 1.0
        if self.grl_routing == "fixed":
            return self.grl_aligner_rho
        return float(
            (self._backbone_gap_ema / self.grl_routing_tau).clamp(0.0, 1.0)
        )

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
        domain_gap = self._normalized_gap(source_features, target_features)

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
        self._epoch_diagnostics = {}

    def _accumulate_diagnostics(self, **values: float) -> None:
        for name, value in values.items():
            total, count = self._epoch_diagnostics.get(name, (0.0, 0))
            self._epoch_diagnostics[name] = (total + float(value), count + 1)

    def on_train_epoch_end(self):
        # One compact line per epoch, independent of log_every_n_batches, so
        # the routing behaviour can be checked on BCI 2a vs Zhou2016.
        means = {
            name: total / max(count, 1)
            for name, (total, count) in self._epoch_diagnostics.items()
        }
        if not means:
            return
        group_scales = (
            [round(float(v), 4) for v in self.group_aligner.scales.detach()]
            if self.group_aligner is not None
            else None
        )
        self.print(
            f"[DA] epoch {self.current_epoch + 1}/{self.hparams.max_epochs} | "
            f"routing={self.grl_routing} | rho={means['rho']:.3f} | "
            f"backbone_gap={means['backbone_gap']:.3f} | "
            f"aligned_gap={means['aligned_gap']:.3f} | "
            f"conf={means['confidence']:.3f} | gate={means['gate']:.2f} | "
            f"grl={means['grl_alpha']:.3f} | D_acc={means['domain_acc']:.3f} | "
            f"global_scale={float(self.aligner.scale.detach()):.4f} | "
            f"group_scales={group_scales}"
        )

    def adapt_to_target(self, target_loader):
        """Adapt BatchNorm affine parameters using unlabeled target trials.

        The information-maximization objective sharpens individual target
        predictions while maintaining a diverse batch-level class marginal.
        Target labels may be present in the evaluation loader but are ignored.
        """
        if self.im_tta_steps == 0:
            return None

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

        all_features, global_correction, group_correction = self._align(
            pooled_features
        )
        source_features = all_features[:source_count]
        target_features = all_features[source_count:]

        source_logits = self.model.classify_features(source_features)
        with torch.no_grad():
            target_logits = self.model.classify_features(target_features)
        classification_loss = F.cross_entropy(source_logits, source_y)

        adaptation_gate, target_gap, target_confidence = (
            self._target_adaptation_gate(
                source_features, target_features, target_logits
            )
        )

        alpha = self._grl_alpha()
        rho = self._aligner_grl_share(
            pooled_features[:source_count], pooled_features[source_count:]
        )
        # The discriminator sees exactly the aligned features (same forward
        # value as Full-Mamba). Only the backward pass is routed: the backbone
        # gets the full reversed gradient, the global aligner a share rho, and
        # the group-preserving path none at all (CE + MMD only).
        discriminator_input = pooled_features + (
            global_correction
            if rho == 1.0
            else _GradientScale.apply(global_correction, rho)
        )
        if group_correction is not None:
            discriminator_input = discriminator_input + group_correction.detach()
        domain_logits = self.domain_discriminator(
            self.grl(discriminator_input, alpha)
        )
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
        with torch.no_grad():
            domain_acc = (
                (domain_logits > 0).float() == domain_targets
            ).float().mean()
        self._accumulate_diagnostics(
            rho=rho,
            backbone_gap=self._backbone_gap_ema.item(),
            aligned_gap=target_gap.item(),
            confidence=target_confidence.item(),
            gate=adaptation_gate.item(),
            grl_alpha=alpha,
            domain_acc=domain_acc.item(),
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
        self.log("train_grl_aligner_rho", rho, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log("train_domain_acc", domain_acc, on_step=False, on_epoch=True, batch_size=batch_size)

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
