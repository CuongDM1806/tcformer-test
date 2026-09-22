"""Large-kernel depthwise temporal blocks for TCFormer.

The blocks operate on ``[B, C, 1, T]`` tensors after the first temporal pool.
Channels remain grouped by temporal scale, and BatchNorm is retained so the
existing IM-TTA routine can adapt the new stage.
"""

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        if not 0.0 <= drop_prob < 1.0:
            raise ValueError("drop_prob must be in [0, 1).")
        self.drop_prob = float(drop_prob)

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep


class LKTemporalBlock(nn.Module):
    """Residual large-kernel temporal mixer with a grouped pointwise FFN."""

    def __init__(
        self,
        channels: int,
        n_groups: int = 3,
        kernel_size: int = 31,
        small_kernel: int = 5,
        expand: int = 2,
        dropout: float = 0.3,
        drop_path: float = 0.1,
        layer_scale_init: float = 1.0,
    ):
        super().__init__()
        if kernel_size % 2 == 0 or small_kernel % 2 == 0:
            raise ValueError("kernel_size and small_kernel must be odd.")
        if small_kernel > kernel_size:
            raise ValueError("small_kernel must be <= kernel_size.")
        if channels % n_groups != 0:
            raise ValueError("channels must be divisible by n_groups.")
        if expand < 1:
            raise ValueError("expand must be positive.")

        self.channels = channels
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        self.dw_large = nn.Conv2d(
            channels,
            channels,
            (1, kernel_size),
            padding=(0, kernel_size // 2),
            groups=channels,
            bias=False,
        )
        self.bn_large = nn.BatchNorm2d(channels)
        self.dw_small = nn.Conv2d(
            channels,
            channels,
            (1, small_kernel),
            padding=(0, small_kernel // 2),
            groups=channels,
            bias=False,
        )
        self.bn_small = nn.BatchNorm2d(channels)

        hidden = channels * expand
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, groups=n_groups, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden, channels, 1, groups=n_groups, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.gamma = nn.Parameter(
            torch.full((1, channels, 1, 1), float(layer_scale_init))
        )
        self.drop_path = DropPath(drop_path)
        self.reparameterized = False

    def _token_mixer(self, x: Tensor) -> Tensor:
        if self.reparameterized:
            return self.dw_merged(x)
        return self.bn_large(self.dw_large(x)) + self.bn_small(self.dw_small(x))

    def forward(self, x: Tensor) -> Tensor:
        update = self.ffn(self._token_mixer(x))
        return x + self.drop_path(self.gamma * update)

    @torch.no_grad()
    def reparameterize(self) -> None:
        """Fuse parallel depthwise convolutions for inference/latency only.

        Do not call this before or during IM-TTA because fusion removes the two
        BatchNorm layers and consumes their running statistics.
        """
        if self.reparameterized:
            return

        def fuse(conv: nn.Conv2d, bn: nn.BatchNorm2d):
            if bn.running_var is None:
                raise RuntimeError("BatchNorm has no running stats; cannot fuse.")
            std = (bn.running_var + bn.eps).sqrt()
            weight = conv.weight * (bn.weight / std).reshape(-1, 1, 1, 1)
            bias = bn.bias - bn.running_mean * bn.weight / std
            return weight, bias

        large_weight, large_bias = fuse(self.dw_large, self.bn_large)
        small_weight, small_bias = fuse(self.dw_small, self.bn_small)
        pad = (self.kernel_size - self.small_kernel) // 2
        small_weight = F.pad(small_weight, (pad, pad))

        merged = nn.Conv2d(
            self.channels,
            self.channels,
            (1, self.kernel_size),
            padding=(0, self.kernel_size // 2),
            groups=self.channels,
            bias=True,
        ).to(device=large_weight.device, dtype=large_weight.dtype)
        merged.weight.copy_(large_weight + small_weight)
        merged.bias.copy_(large_bias + small_bias)
        self.dw_merged = merged
        del self.dw_large, self.bn_large, self.dw_small, self.bn_small
        self.reparameterized = True


class LKTemporalStage(nn.Sequential):
    """Stack LK blocks with linearly increasing stochastic-depth rates."""

    def __init__(
        self,
        channels: int,
        n_groups: int,
        depth: int = 1,
        drop_path_max: float = 0.1,
        **block_kwargs,
    ):
        if depth < 1:
            raise ValueError("depth must be positive.")
        rates = (
            [drop_path_max * i / (depth - 1) for i in range(depth)]
            if depth > 1
            else [drop_path_max]
        )
        super().__init__(
            *[
                LKTemporalBlock(channels, n_groups, drop_path=rate, **block_kwargs)
                for rate in rates
            ]
        )

    def reparameterize(self) -> None:
        for block in self:
            block.reparameterize()


if __name__ == "__main__":
    torch.manual_seed(0)
    batch, channels, groups, length = 8, 48, 3, 125
    stage = LKTemporalStage(channels, groups, depth=2, kernel_size=31)
    x = torch.randn(batch, channels, 1, length)

    stage.train()
    for _ in range(5):
        stage(torch.randn(batch, channels, 1, length))

    stage.eval()
    reference = stage(x)
    print("output shape:", tuple(reference.shape))
    print("params:", sum(parameter.numel() for parameter in stage.parameters()))

    perturbed = x.clone()
    perturbed[:, :16] += 1.0
    difference = (stage(perturbed) - reference).abs().amax(dim=(0, 2, 3))
    print("other groups unaffected:", bool(difference[16:].max() < 1e-6))

    stage.reparameterize()
    reparameterized = stage(x)
    print("reparam max abs error:", (reparameterized - reference).abs().max().item())
