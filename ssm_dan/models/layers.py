"""Building blocks shared by SSM-DAN and the baselines."""
import torch
from torch import nn
from torch.nn import functional as F


class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1,
                 groups=1, bias=True):
        super(CausalConv1d, self).__init__(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride,
            dilation=dilation, groups=groups, bias=bias)
        self.__padding = (kernel_size - 1) * dilation

    def forward(self, x):
        return super(CausalConv1d, self).forward(F.pad(x, (self.__padding, 0)))


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=None, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.max_norm is not None:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)

class Conv1dWithConstraint(nn.Conv1d):
    def __init__(self, *args, max_norm=None, **kwargs):
        self.max_norm = max_norm
        super().__init__(*args, **kwargs)

    def forward(self, x):
        if self.max_norm is not None:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super().forward(x)


class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, max_norm=None, **kwargs):
        self.max_norm = max_norm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.max_norm is not None:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)


class ChannelGroupAttention(nn.Module):
    """
    Implements Channel Group Attention.

    Applies attention mechanism over groups of channels rather than individual channels.

    Args:
        in_channels (int): Number of input channels (C). Must be divisible by num_groups.
        num_groups (int): Number of groups (G) to divide channels into.
    """
    def __init__(self, in_channels, num_groups, reduction=4):
        
        super().__init__()

        assert in_channels % num_groups == 0, "in_channels must be divisible by num_groups"

        self.in_channels = in_channels
        self.num_groups = num_groups
        self.group_size = in_channels // num_groups

        # x → GlobalAvgPool → att_fc → [ReLU?] → [Expand and Sigmoid] → multiply with input
        """
            1. Global Avg Pool          → [B, C, 1, 1] ← SQUEEZE
            2. Grouped Conv (att_fc)    → [B, C/r, 1, 1] grouped 1×1 conv
            3. ReLU (activation)        → [B, C/r, 1, 1]
            4. Grouped Conv (expand)    → [B, G, 1, 1] grouped 1×1 conv
            5. Sigmoid                  → [B, G, 1, 1]  [0..1] scale factors per group
            6. Multiply with input      → [B, C, H, W]
        """
        
        # Global Average Pooling to [B, C, 1, 1]
        self.pool = nn.AdaptiveAvgPool2d(1)
        # Project to 1 value per group (using grouped conv!)
        # Using Conv2d with kernel size 1 acts like a Linear layer on (B, G, 1, 1) tensors
        # For (B, G, 1, 1) tensors, nn.Conv2d(C, G, kernel_size=1) is equivalent to nn.Linear(C, G)
        self.att_fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, groups=num_groups, bias=False)
        self.att_fc2 = nn.Conv2d(in_channels // reduction, num_groups, kernel_size=1, groups=num_groups, bias=False)
        self.relu = nn.ReLU()           
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W) after applying group attention.
        """
        B, C, H, W = x.shape

        if C != self.in_channels:
             raise ValueError(f"Input channel dimension ({C}) does not match "
                              f"configured in_channels ({self.in_channels})")

        # 1. Global Average Pool
        pooled = self.pool(x)   # [B, C, 1, 1]  ← SQUEEZE

        # 2. Compress to per-group attention: [B, num_groups, 1, 1]
        group_att = self.att_fc1(pooled) # [B, C/r, 1, 1] grouped 1×1 conv
        att_weights = self.att_fc2(self.relu(group_att)) # [B, G, 1, 1] grouped 1×1 conv
          
        # 3. Apply sigmoid to get attention values
        att_weights = self.sigmoid(att_weights)

        # 4. Apply attention
        # Efficient expansion using view (instead of repeat_interleave)
        if att_weights.shape[1] != C:
            # 1. Reshape input to group structure
            x_reshaped = x.view(B, self.num_groups, self.group_size, H, W)

            # 2. Reshape attention weights: [B, G, 1, 1, 1]
            att_weights = att_weights.view(B, self.num_groups, 1, 1, 1)

            # 3. Element-wise multiplication with broadcasting
            out = x_reshaped * att_weights

            # 4. Reshape back to [B, C, H, W]
            out = out.view(B, C, H, W)
        else:
            # Broadcasting handles the H and W dimensions: (B, C, H, W) * (B, C, 1, 1) -> (B, C, H, W)
            out = x * att_weights

        return out


def glorot_weight_zero_bias(model):
    for module in model.modules():
        if hasattr(module, "weight"):
            if "norm" not in module.__class__.__name__.lower():
                nn.init.xavier_uniform_(module.weight)
        if hasattr(module, "bias"):
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
