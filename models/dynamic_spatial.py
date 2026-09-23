"""Input-conditioned spatial filtering for multi-scale EEG features."""

import torch
from torch import Tensor, nn


class ScaleConditionedDynamicSpatialConv(nn.Module):
    """Replace a static depthwise EEG spatial convolution.

    The input channels are arranged as contiguous temporal-scale groups.  Each
    temporal feature keeps its own static spatial filters, matching the
    Full-Mamba baseline.  A small scale-specific network additionally predicts
    a trial-dependent spatial correction from the mean and standard deviation
    of the current feature tensor.

    The dynamic residual is shared by temporal features within a scale.  This
    keeps the conditioning network small while allowing kernels 20, 32 and 64
    to use different trial-specific electrode combinations.
    """

    def __init__(
        self,
        in_channels: int,
        num_scales: int,
        n_electrodes: int,
        depth_multiplier: int = 2,
        reduction: int = 4,
    ) -> None:
        super().__init__()
        if in_channels % num_scales != 0:
            raise ValueError("in_channels must be divisible by num_scales")
        if n_electrodes < 1 or depth_multiplier < 1:
            raise ValueError("n_electrodes and depth_multiplier must be positive")

        self.in_channels = in_channels
        self.num_scales = num_scales
        self.channels_per_scale = in_channels // num_scales
        self.n_electrodes = n_electrodes
        self.depth_multiplier = depth_multiplier
        self.out_channels = in_channels * depth_multiplier

        # Static baseline: one spatial vector for every temporal feature and
        # every depth-multiplier head, exactly as in a depthwise Conv2d(C, C*D).
        # Keep the same weight layout as the replaced depthwise Conv2d so a
        # Full-Mamba spatial checkpoint can be reused directly.
        self.weight = nn.Parameter(
            torch.empty(self.out_channels, 1, n_electrodes, 1)
        )
        nn.init.xavier_uniform_(self.weight)

        descriptor_dim = 2 * n_electrodes  # electrode-wise mean and std
        hidden_dim = max(8, descriptor_dim // reduction)
        self.conditioners = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(descriptor_dim, hidden_dim),
                    nn.ELU(),
                    nn.Linear(hidden_dim, depth_multiplier * n_electrodes),
                    nn.Tanh(),
                )
                for _ in range(num_scales)
            ]
        )

        # tanh(dynamic_scale) bounds the residual strength. Starting from zero
        # makes the first forward pass identical to the static spatial filter.
        self.dynamic_scale = nn.Parameter(torch.zeros(num_scales))

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [B, F, C, T], got shape {tuple(x.shape)}")
        batch, features, electrodes, timepoints = x.shape
        if features != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} feature channels, got {features}"
            )
        if electrodes != self.n_electrodes:
            raise ValueError(
                f"Expected {self.n_electrodes} EEG electrodes, got {electrodes}"
            )

        grouped = x.reshape(
            batch,
            self.num_scales,
            self.channels_per_scale,
            electrodes,
            timepoints,
        )

        # Summarize each electrode within each temporal scale for this trial.
        mean = grouped.mean(dim=(2, 4))
        std = grouped.var(dim=(2, 4), unbiased=False).add(1e-6).sqrt()
        descriptor = torch.cat((mean, std), dim=-1)

        dynamic = torch.stack(
            [
                conditioner(descriptor[:, scale_idx])
                for scale_idx, conditioner in enumerate(self.conditioners)
            ],
            dim=1,
        ).reshape(
            batch,
            self.num_scales,
            self.depth_multiplier,
            self.n_electrodes,
        )

        residual_strength = self.dynamic_scale.tanh().view(
            1, self.num_scales, 1, 1, 1
        )
        static_weight = self.weight.reshape(
            self.num_scales,
            self.channels_per_scale,
            self.depth_multiplier,
            self.n_electrodes,
        )
        effective_weight = (
            static_weight.unsqueeze(0)
            + residual_strength * dynamic.unsqueeze(2)
        )

        # [B,S,F,C,T] x [B,S,F,D,C] -> [B,S,F,D,T]
        output = torch.einsum("bsfct,bsfdc->bsfdt", grouped, effective_weight)
        return output.reshape(batch, self.out_channels, 1, timepoints)
