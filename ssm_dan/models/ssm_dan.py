"""SSM-DAN: the SSM backbone with a domain-adaptation head and IM-TTA.

Stage 1 trains on labeled source and unlabeled target batches with
- AutoDIAL feature normalization (learned per-group source/target mixing alpha),
- a residual feature aligner,
- adversarial alignment (gradient reversal + domain discriminator),
- multi-kernel MMD on pooled features and on time-averaged TCN features,
- target information maximization (InfoMax), the only signal that updates alpha.
Stage 2 (IM-TTA) adapts only the backbone BatchNorm affine parameters on the
unlabeled evaluation trials, then re-estimates the AutoDIAL target statistics.

Setting ``domain_adaptation: false`` in the run config feeds source batches
only; with ``feature_alignment: none`` and zero loss weights this gives the
"without domain adaptation" variant.
"""
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchmetrics.functional import accuracy

from .backbone import SSMBackbone, split_backbone_kwargs
from .lightning import ClassificationModule


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
    """Learned residual correction of the pooled features."""

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
    """Domain alignment layer with a learned mixing weight (Carlucci et al., ICCV 2017).

    Source features are standardized with source statistics; target features
    with the moments of the mixture ``alpha * target + (1 - alpha) * source``,
    with one ``alpha`` in [0, 1] per feature group. ``alpha -> 1`` is fully
    domain-specific normalization, ``alpha -> 0`` normalizes the target with
    source statistics. A shared affine transform follows.
    """

    def __init__(self, feature_dim: int, n_groups: int, momentum: float = 0.1,
                 eps: float = 1e-5, alpha_init: float = 0.5):
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
        var = (a * target_var + (1.0 - a) * source_var
               + a * (1.0 - a) * (target_mean - source_mean).square())
        return mean, var

    def _normalize(self, features: Tensor, mean: Tensor, var: Tensor) -> Tensor:
        standardized = (features - mean) / torch.sqrt(var + self.eps)
        return standardized * self.weight + self.bias

    @torch.no_grad()
    def _update_running(self, domain: str, mean: Tensor, var: Tensor, count: int):
        unbiased = var * count / max(count - 1, 1)
        getattr(self, f"{domain}_mean").lerp_(mean, self.momentum)
        getattr(self, f"{domain}_var").lerp_(unbiased, self.momentum)

    def forward_pair(self, source: Tensor, target: Tensor, detach_alpha: bool = False,
                     update_stats: bool = True) -> tuple[Tensor, Tensor]:
        """Training path: batch statistics of both domains, as in BatchNorm."""
        source_mean = source.mean(dim=0)
        source_var = source.var(dim=0, unbiased=False)
        target_mean = target.mean(dim=0)
        target_var = target.var(dim=0, unbiased=False)
        if update_stats:
            self._update_running("source", source_mean, source_var, source.size(0))
            self._update_running("target", target_mean, target_var, target.size(0))
        mixed_mean, mixed_var = self._mix(
            self.alpha(detach_alpha), source_mean, source_var, target_mean, target_var)
        return (self._normalize(source, source_mean, source_var),
                self._normalize(target, mixed_mean, mixed_var))

    def forward(self, features: Tensor, domain: str) -> Tensor:
        """Inference path: running statistics of the requested domain."""
        if domain == "source":
            return self._normalize(features, self.source_mean, self.source_var)
        mean, var = self._mix(self.alpha(detach=True), self.source_mean, self.source_var,
                              self.target_mean, self.target_var)
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
    """Multi-kernel (Gaussian) MMD on L2-normalized features."""

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
        kernels = sum(torch.exp(-distances / (bandwidth * (self.kernel_mul ** idx)))
                      for idx in range(self.kernel_num))

        source_count, target_count = source.size(0), target.size(0)
        k_ss = kernels[:source_count, :source_count]
        k_tt = kernels[source_count:, source_count:]
        k_st = kernels[:source_count, source_count:]
        source_term = (k_ss.sum() - k_ss.diagonal().sum()) / (source_count * (source_count - 1))
        target_term = (k_tt.sum() - k_tt.diagonal().sum()) / (target_count * (target_count - 1))
        return source_term + target_term - 2.0 * k_st.mean()


def information_maximization(logits: Tensor) -> Tensor:
    """Mean conditional entropy minus the entropy of the batch marginal."""
    probabilities = logits.softmax(dim=1)
    log_probabilities = probabilities.clamp_min(1e-6).log()
    conditional = -(probabilities * log_probabilities).sum(dim=1).mean()
    marginal = probabilities.mean(dim=0)
    marginal_entropy = -(marginal * marginal.clamp_min(1e-6).log()).sum()
    return conditional - marginal_entropy


class SSMDAN(ClassificationModule):
    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        feature_alignment: str = "autodial",
        autodial_alpha_init: float = 0.5,
        domain_norm_momentum: float = 0.1,
        target_im_weight: float = 0.1,
        recalibrate_target_statistics: bool = True,
        aligner_hidden_dim: int = 128,
        domain_hidden_dim: int = 128,
        adaptation_dropout: float = 0.3,
        adversarial_weight: float = 1.0,
        mmd_weight: float = 0.5,
        temporal_mmd_weight: float = 0.1,
        im_tta_steps: int = 5,
        im_tta_lr: float = 1e-4,
        im_tta_diversity_weight: float = 1.0,
        **kwargs,
    ):
        backbone_kwargs, kwargs = split_backbone_kwargs(dict(kwargs))
        model = SSMBackbone(n_channels=n_channels, n_classes=n_classes, **backbone_kwargs)
        super().__init__(model, n_classes, **backbone_kwargs, **kwargs)
        if feature_alignment not in ("none", "autodial"):
            raise ValueError("feature_alignment must be 'none' or 'autodial'.")
        if min(target_im_weight, im_tta_steps, im_tta_diversity_weight) < 0 or im_tta_lr <= 0:
            raise ValueError("Adaptation weights and IM-TTA steps must be non-negative.")
        # One AutoDIAL mixing weight per feature group: the three temporal-kernel
        # groups of the convolutional shortcut and the selective-SSM group.
        self.domain_norm = (
            AutoDIALFeatureNorm(model.feature_dim, n_groups=model.n_groups + 1,
                                momentum=domain_norm_momentum, alpha_init=autodial_alpha_init)
            if feature_alignment == "autodial" else None
        )
        self.target_im_weight = float(target_im_weight)
        self.recalibrate_target_statistics = bool(recalibrate_target_statistics)
        # Validation scores labeled source trials; every other pass sees the target.
        self._inference_domain = "target"
        self.aligner = ResidualFeatureAligner(model.feature_dim, aligner_hidden_dim, adaptation_dropout)
        self.grl = GradientReversal()
        self.domain_discriminator = DomainDiscriminator(model.feature_dim, domain_hidden_dim,
                                                        adaptation_dropout)
        self.mmd_loss = MultiKernelMMDLoss()
        self.adversarial_weight = adversarial_weight
        self.mmd_weight = mmd_weight
        self.temporal_mmd_weight = temporal_mmd_weight
        self.im_tta_steps = int(im_tta_steps)
        self.im_tta_lr = float(im_tta_lr)
        self.im_tta_diversity_weight = float(im_tta_diversity_weight)

    # ----------------------------------------------------------------- inference
    def forward(self, x: Tensor, domain: str | None = None) -> Tensor:
        features = self.model.extract_features(x)
        if self.domain_norm is not None:
            features = self.domain_norm(features, domain or self._inference_domain)
        return self.model.classify_features(self.aligner(features))

    def on_validation_start(self):
        self._inference_domain = "source"

    def on_validation_end(self):
        self._inference_domain = "target"

    def autodial_alpha(self) -> list[float] | None:
        if self.domain_norm is None:
            return None
        return [round(float(a), 4) for a in self.domain_norm.alpha(detach=True)]

    def on_train_epoch_end(self):
        epoch = self.current_epoch + 1
        if self.domain_norm is not None and (epoch % 10 == 0 or epoch == int(self.hparams.max_epochs)):
            self.print(f"Epoch {epoch} | AutoDIAL alpha: {self.autodial_alpha()}")

    # ------------------------------------------------------------------ Stage 1
    def _grl_alpha(self) -> float:
        progress = self.current_epoch / max(int(self.hparams.max_epochs) - 1, 1)
        return 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0

    def _log(self, name, value, batch_size, prog_bar=False):
        self.log(name, value, prog_bar=prog_bar, on_step=False, on_epoch=True, batch_size=batch_size)

    def training_step(self, batch, batch_idx):
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            return self._source_only_step(batch)
        if not isinstance(batch, dict) or "source" not in batch or "target" not in batch:
            raise RuntimeError("Expected a labeled source batch or {'source', 'target'} UDA batches.")

        source_x, source_y = batch["source"]
        target_x = batch["target"]
        source_count = source_x.size(0)

        # One forward pass over both domains, so BatchNorm sees both.
        temporal_features = self.model.extract_temporal_features(torch.cat((source_x, target_x), dim=0))
        pooled_features = self.model.tcn_head.pool_temporal_features(temporal_features)

        # Temporal MMD on the time-averaged TCN output (before pooling).
        temporal_mean = temporal_features.mean(dim=-1)
        temporal_mmd_loss = self.mmd_loss(temporal_mean[:source_count], temporal_mean[source_count:])

        source_pooled = pooled_features[:source_count]
        target_pooled = pooled_features[source_count:]
        target_task_features = None
        if self.domain_norm is not None:
            # The alignment losses see a detached alpha (they could always be
            # lowered by alpha -> 1); alpha is learned from the task signals
            # only (source cross-entropy and target InfoMax).
            source_pooled, target_aligned_input = self.domain_norm.forward_pair(
                source_pooled, target_pooled, detach_alpha=True)
            if self.target_im_weight > 0.0:
                _, target_task_input = self.domain_norm.forward_pair(
                    pooled_features[:source_count], target_pooled, update_stats=False)
                target_task_features = self.aligner(target_task_input)
            target_pooled = target_aligned_input

        all_features = self.aligner(torch.cat((source_pooled, target_pooled), dim=0))
        source_features = all_features[:source_count]
        target_features = all_features[source_count:]

        source_logits = self.model.classify_features(source_features)
        with torch.no_grad():
            # Unused output. Every classifier call re-applies the max-norm
            # constraint to its weights in place, so this call keeps the
            # sequence of weight updates of the runs reported in the paper.
            self.model.classify_features(target_features)
        classification_loss = F.cross_entropy(source_logits, source_y)
        target_im_loss = source_logits.new_zeros(())
        if target_task_features is not None:
            target_im_loss = information_maximization(self.model.classify_features(target_task_features))

        alpha = self._grl_alpha()
        domain_logits = self.domain_discriminator(self.grl(all_features, alpha))
        domain_targets = torch.cat((
            torch.zeros(source_count, 1, device=all_features.device),
            torch.ones(target_features.size(0), 1, device=all_features.device),
        ), dim=0)
        adversarial_loss = F.binary_cross_entropy_with_logits(domain_logits, domain_targets)
        mmd_loss = self.mmd_loss(source_features, target_features)
        loss = (
            classification_loss
            + self.target_im_weight * target_im_loss
            + (self.adversarial_weight * adversarial_loss
               + self.mmd_weight * mmd_loss
               + self.temporal_mmd_weight * temporal_mmd_loss)
        )

        acc = accuracy(source_logits, source_y, task="multiclass", num_classes=self.hparams.n_classes)
        self._log("train_loss", loss, source_count, prog_bar=True)
        self._log("train_acc", acc, source_count, prog_bar=True)
        self._log("train_cls_loss", classification_loss, source_count)
        self._log("train_domain_loss", adversarial_loss, source_count)
        self._log("train_mmd_loss", mmd_loss, source_count)
        self._log("train_temporal_mmd_loss", temporal_mmd_loss, source_count)
        self._log("grl_alpha", alpha, source_count)
        if self.domain_norm is not None:
            self._log("train_target_im_loss", target_im_loss, source_count)
            for group, value in enumerate(self.domain_norm.alpha(detach=True)):
                self._log(f"autodial_alpha_g{group}", value, source_count)
        return loss

    def _source_only_step(self, batch):
        """Labeled source batch only (``domain_adaptation: false``)."""
        if self.domain_norm is not None:
            raise RuntimeError("Training without target batches needs feature_alignment: none.")
        source_x, source_y = batch
        source_logits = self.model.classify_features(self.aligner(self.model.extract_features(source_x)))
        loss = F.cross_entropy(source_logits, source_y)
        acc = accuracy(source_logits, source_y, task="multiclass", num_classes=self.hparams.n_classes)
        batch_size = source_x.size(0)
        self._log("train_loss", loss, batch_size, prog_bar=True)
        self._log("train_acc", acc, batch_size, prog_bar=True)
        self._log("train_cls_loss", loss, batch_size)
        return loss

    # ------------------------------------------------------------------ Stage 2
    def adapt_to_target(self, target_loader):
        """Label-free adaptation to the evaluation trials after training.

        IM-TTA (if ``im_tta_steps > 0``) updates the backbone BatchNorm affine
        parameters; the AutoDIAL target statistics are then re-estimated on
        the same trials. Labels in the loader, if any, are never read.
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
        # eval() keeps dropout off; after IM-TTA the BatchNorm layers keep
        # using target batch statistics, exactly as in the test pass.
        self.eval()
        device = next(self.parameters()).device
        feature_sum, square_sum, count = None, None, 0
        for batch in target_loader:
            target_x = batch[0] if isinstance(batch, (tuple, list)) else batch
            features = self.model.extract_features(target_x.to(device, non_blocking=True)).double()
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
        """InfoMax on the whole target set; one Adam step on BN affine parameters per pass."""
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

        batch_norm_modules, adaptation_parameters = [], []
        for module in self.model.modules():
            if not isinstance(module, nn.modules.batchnorm._BatchNorm) or not module.affine:
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
            raise RuntimeError("IM-TTA requires at least one affine BatchNorm layer.")

        optimizer = torch.optim.Adam(adaptation_parameters, lr=self.im_tta_lr)
        self.print(f"IM-TTA start | steps={self.im_tta_steps} | lr={self.im_tta_lr:g} | "
                   f"BN_layers={len(batch_norm_modules)} | "
                   f"trainable_params={sum(p.numel() for p in adaptation_parameters)}")
        device = next(self.parameters()).device
        final_stats = None
        with torch.enable_grad():
            for step in range(1, self.im_tta_steps + 1):
                # Pass 1: class marginal over the whole target set. Dropout is
                # off and BatchNorm has no running buffers, so no state changes.
                probability_sum, sample_count = None, 0
                with torch.no_grad():
                    for batch in target_loader:
                        target_x = batch[0] if isinstance(batch, (tuple, list)) else batch
                        target_x = target_x.to(device, non_blocking=True)
                        batch_sum = self(target_x).softmax(dim=1).sum(dim=0)
                        probability_sum = batch_sum if probability_sum is None else probability_sum + batch_sum
                        sample_count += target_x.size(0)
                if sample_count == 0:
                    raise RuntimeError("IM-TTA received an empty target loader.")
                mean_probability = probability_sum / sample_count
                marginal_entropy = -(mean_probability * mean_probability.clamp_min(1e-6).log()).sum()
                marginal_gradient = mean_probability.clamp_min(1e-6).log() + 1.0

                # Pass 2: accumulate the exact full-set InfoMax gradient.
                optimizer.zero_grad(set_to_none=True)
                conditional_sum, second_pass_count = 0.0, 0
                for batch in target_loader:
                    target_x = batch[0] if isinstance(batch, (tuple, list)) else batch
                    target_x = target_x.to(device, non_blocking=True)
                    probabilities = self.forward(target_x).softmax(dim=1)
                    log_probabilities = probabilities.clamp_min(1e-6).log()
                    conditional_entropy = -(probabilities * log_probabilities).sum(dim=1).mean()
                    batch_size = target_x.size(0)
                    diversity_surrogate = (probabilities * marginal_gradient.unsqueeze(0)).sum(dim=1).mean()
                    loss = (batch_size / sample_count) * (
                        conditional_entropy + self.im_tta_diversity_weight * diversity_surrogate)
                    loss.backward()
                    conditional_sum += conditional_entropy.detach().item() * batch_size
                    second_pass_count += batch_size
                if second_pass_count != sample_count:
                    raise RuntimeError("IM-TTA target loader changed between passes.")
                optimizer.step()
                conditional_value = conditional_sum / sample_count
                final_stats = {
                    "loss": conditional_value - self.im_tta_diversity_weight * marginal_entropy.item(),
                    "conditional_entropy": conditional_value,
                    "marginal_entropy": marginal_entropy.item(),
                    "samples": sample_count,
                }
                self.print(f"IM-TTA step {step}/{self.im_tta_steps} | loss={final_stats['loss']:.4f} | "
                           f"cond_entropy={conditional_value:.4f} | "
                           f"marg_entropy={final_stats['marginal_entropy']:.4f} | "
                           f"target_samples={sample_count}")
        # Dropout stays off; BatchNorm keeps using target batch statistics.
        self.eval()
        return final_stats
