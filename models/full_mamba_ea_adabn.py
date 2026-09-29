"""Full-Mamba with target-statistics adaptation and optional IM-TTA.

Training is supervised on source subjects only. Domain adaptation happens
without labels through session-wise Euclidean Alignment in the data modules,
target-statistics BatchNorm here, and information-maximization TTA on the
BatchNorm affine parameters. There is no discriminator, gradient reversal,
MMD, or trainable feature aligner.
"""

import torch
from torch import Tensor, nn

from .classification_module import ClassificationModule
from .tcformer import TCFormerModule


class FullMambaEAdaBN(ClassificationModule):
    """Full-Mamba classifier using EA + adaptive BN + IM-TTA."""

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
        im_tta_steps: int = 5,
        im_tta_lr: float = 1e-4,
        im_tta_diversity_weight: float = 1.0,
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
        if im_tta_steps < 0:
            raise ValueError("im_tta_steps must be non-negative.")
        if im_tta_lr <= 0.0:
            raise ValueError("im_tta_lr must be positive.")
        if im_tta_diversity_weight < 0.0:
            raise ValueError("im_tta_diversity_weight must be non-negative.")
        self.im_tta_steps = int(im_tta_steps)
        self.im_tta_lr = float(im_tta_lr)
        self.im_tta_diversity_weight = float(im_tta_diversity_weight)

    @staticmethod
    def _target_x(batch):
        return batch[0] if isinstance(batch, (tuple, list)) else batch

    def _batch_norm_modules(self):
        return [
            module
            for module in self.model.modules()
            if isinstance(module, nn.modules.batchnorm._BatchNorm)
        ]

    @torch.no_grad()
    def _adapt_batch_norm_statistics(self, target_loader, device):
        """Replace source BN buffers with statistics from the unlabeled target."""
        target_batches = [self._target_x(batch).cpu() for batch in target_loader]
        if not target_batches:
            raise RuntimeError("Adaptive BatchNorm received an empty target loader.")
        target_x = torch.cat(target_batches, dim=0).to(device, non_blocking=True)

        self.eval()
        batch_norm_modules = self._batch_norm_modules()
        if not batch_norm_modules:
            raise RuntimeError("Adaptive BatchNorm requires BatchNorm layers.")
        for module in batch_norm_modules:
            if not module.track_running_stats:
                raise RuntimeError("Adaptive BatchNorm requires running statistics.")
            module.reset_running_stats()
            module.momentum = None
            module.train()

        # Dropout and every non-BN module remain in evaluation mode. A single
        # full-target pass reproduces the offline target-statistics assumption.
        self(target_x)
        self.eval()
        self.print(
            f"Adaptive BN complete | layers={len(batch_norm_modules)} | "
            f"unlabeled_target_samples={target_x.size(0)}"
        )
        return batch_norm_modules, target_x.size(0)

    def adapt_to_target(self, target_loader):
        """Run target Adaptive BN, then retain the existing IM-TTA mechanism."""
        device = next(self.parameters()).device
        for parameter in self.parameters():
            parameter.requires_grad_(False)

        batch_norm_modules, sample_count = self._adapt_batch_norm_statistics(
            target_loader, device
        )
        if self.im_tta_steps == 0:
            return {
                "loss": None,
                "conditional_entropy": None,
                "marginal_entropy": None,
                "samples": sample_count,
            }

        adaptation_parameters = []
        for module in batch_norm_modules:
            if not module.affine:
                continue
            module.weight.requires_grad_(True)
            module.bias.requires_grad_(True)
            adaptation_parameters.extend((module.weight, module.bias))
        if not adaptation_parameters:
            raise RuntimeError("IM-TTA requires affine BatchNorm parameters.")

        optimizer = torch.optim.Adam(adaptation_parameters, lr=self.im_tta_lr)
        parameter_count = sum(p.numel() for p in adaptation_parameters)
        self.print(
            f"IM-TTA start | steps={self.im_tta_steps} | lr={self.im_tta_lr:g} | "
            f"trainable_BN_affine_params={parameter_count}"
        )

        final_stats = None
        with torch.enable_grad():
            for step in range(1, self.im_tta_steps + 1):
                probability_sum = None
                observed = 0
                with torch.no_grad():
                    for batch in target_loader:
                        target_x = self._target_x(batch).to(device, non_blocking=True)
                        probabilities = self(target_x).softmax(dim=1)
                        probability_sum = (
                            probabilities.sum(dim=0)
                            if probability_sum is None
                            else probability_sum + probabilities.sum(dim=0)
                        )
                        observed += target_x.size(0)
                if observed == 0:
                    raise RuntimeError("IM-TTA received an empty target loader.")

                mean_probability = probability_sum / observed
                marginal_entropy = -(
                    mean_probability * mean_probability.clamp_min(1e-6).log()
                ).sum()
                marginal_gradient = mean_probability.clamp_min(1e-6).log() + 1.0

                optimizer.zero_grad(set_to_none=True)
                conditional_sum = 0.0
                second_pass_count = 0
                for batch in target_loader:
                    target_x = self._target_x(batch).to(device, non_blocking=True)
                    probabilities = self(target_x).softmax(dim=1)
                    log_probabilities = probabilities.clamp_min(1e-6).log()
                    conditional_entropy = -(
                        probabilities * log_probabilities
                    ).sum(dim=1).mean()
                    batch_size = target_x.size(0)
                    diversity_surrogate = (
                        probabilities * marginal_gradient.unsqueeze(0)
                    ).sum(dim=1).mean()
                    loss = (batch_size / observed) * (
                        conditional_entropy
                        + self.im_tta_diversity_weight * diversity_surrogate
                    )
                    loss.backward()
                    conditional_sum += conditional_entropy.detach().item() * batch_size
                    second_pass_count += batch_size
                if second_pass_count != observed:
                    raise RuntimeError("IM-TTA target loader changed between passes.")
                optimizer.step()
