"""SSM-DAN backbone: multi-scale convolution, bidirectional selective SSM, grouped TCN.

The convolutional stem and the grouped TCN head follow TCFormer
(Altaheri et al., Sci. Rep. 2025, https://github.com/altaheri/TCFormer, MIT
license); the attention encoder of TCFormer is replaced by a stack of
lightweight selective state-space (Mamba-style) blocks with alternating scan
direction.
"""
from typing import List, Sequence

import torch
from torch import Tensor, nn
from einops.layers.torch import Rearrange

from .layers import CausalConv1d, ChannelGroupAttention, Conv1dWithConstraint, glorot_weight_zero_bias
from .lightning import ClassificationModule

SSM_DIRECTIONS = ("forward", "backward", "forward", "backward", "forward")


class MultiKernelConvBlock(nn.Module):
    """Parallel temporal convolutions, depth-wise spatial convolution, grouped temporal conv."""

    def __init__(
        self,
        n_channels: int,
        temp_kernel_lengths: Sequence[int] = (20, 32, 64),
        F1: int = 32,
        D: int = 2,
        pool_length_1: int = 8,
        pool_length_2: int = 7,
        dropout: float = 0.4,
        d_group: int = 16,
        use_group_attn: bool = True,
    ):
        super().__init__()
        self.rearrange = Rearrange("b c seq -> b 1 c seq")
        self.temporal_convs = nn.ModuleList([
            nn.Sequential(
                nn.ConstantPad2d((k // 2 - 1, k // 2, 0, 0) if k % 2 == 0 else (k // 2, k // 2, 0, 0), 0),
                nn.Conv2d(1, F1, (1, k), bias=False),
                nn.BatchNorm2d(F1),
            )
            for k in temp_kernel_lengths
        ])
        n_groups = len(temp_kernel_lengths)
        self.d_model = d_group * n_groups

        # Depth-wise convolution across EEG channels.
        F2 = F1 * n_groups * D
        self.channel_DW_conv = nn.Sequential(
            nn.Conv2d(F1 * n_groups, F2, (n_channels, 1), bias=False, groups=F1 * n_groups),
            nn.BatchNorm2d(F2),
            nn.ELU(),
        )
        self.pool1 = nn.AvgPool2d((1, pool_length_1))
        self.drop1 = nn.Dropout(dropout)

        # Grouped point-wise reduction F2 -> d_model.
        self.use_channel_reduction_2 = self.d_model != F2
        if self.use_channel_reduction_2:
            self.channel_reduction_2 = nn.Sequential(
                nn.Conv2d(F2, self.d_model, (1, 1), bias=False, groups=n_groups),
                nn.BatchNorm2d(self.d_model),
            )

        # Grouped temporal convolution (1 x 16) per kernel group.
        self.temporal_conv_2 = nn.Sequential(
            nn.Conv2d(self.d_model, self.d_model, (1, 16), padding="same", bias=False, groups=n_groups),
            nn.BatchNorm2d(self.d_model),
            nn.ELU(),
        )
        self.use_group_attn = False if n_groups == 1 else use_group_attn
        if self.use_group_attn:
            self.group_attn = ChannelGroupAttention(in_channels=self.d_model, num_groups=n_groups)
        self.pool2 = nn.AvgPool2d((1, pool_length_2))
        self.drop2 = nn.Dropout(dropout)

        glorot_weight_zero_bias(self)

    def forward(self, x):
        x = self.rearrange(x)                                     # [B, 1, C, T]
        x = torch.cat([conv(x) for conv in self.temporal_convs], dim=1)
        x = self.drop1(self.pool1(self.channel_DW_conv(x)))
        if self.use_channel_reduction_2:
            x = self.channel_reduction_2(x)
        x = self.temporal_conv_2(x)
        if self.use_group_attn:
            x = x + self.group_attn(x)
        x = self.drop2(self.pool2(x))
        return x.squeeze(2)                                       # [B, d_model, T']


class TCNBlock(nn.Module):
    def __init__(self, kernel_length: int = 4, n_filters: int = 32, dilation: int = 1,
                 n_groups: int = 1, dropout: float = 0.3):
        super().__init__()
        self.conv1 = CausalConv1d(n_filters, n_filters, kernel_size=kernel_length,
                                  dilation=dilation, groups=n_groups)
        self.bn1 = nn.BatchNorm1d(n_filters)
        self.nonlinearity1 = nn.ELU()
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = CausalConv1d(n_filters, n_filters, kernel_size=kernel_length,
                                  dilation=dilation, groups=n_groups)
        self.bn2 = nn.BatchNorm1d(n_filters)
        self.nonlinearity2 = nn.ELU()
        self.drop2 = nn.Dropout(dropout)
        self.nonlinearity3 = nn.ELU()
        nn.init.constant_(self.conv1.bias, 0.0)
        nn.init.constant_(self.conv2.bias, 0.0)

    def forward(self, input):
        x = self.drop1(self.nonlinearity1(self.bn1(self.conv1(input))))
        x = self.drop2(self.nonlinearity2(self.bn2(self.conv2(x))))
        return self.nonlinearity3(input + x)


class TCN(nn.Module):
    def __init__(self, depth: int = 2, kernel_length: int = 4, n_filters: int = 32,
                 n_groups: int = 1, dropout: float = 0.3):
        super().__init__()
        self.blocks = nn.ModuleList(
            TCNBlock(kernel_length, n_filters, 2 ** i, n_groups, dropout) for i in range(depth)
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class ClassificationHead(nn.Module):
    """Grouped 1x1 projection to class logits and a learned softmax-weighted group sum."""

    def __init__(self, d_features: int, n_groups: int, n_classes: int,
                 kernel_size: int = 1, max_norm: float = 0.25):
        super().__init__()
        self.n_groups = n_groups
        self.n_classes = n_classes
        self.linear = Conv1dWithConstraint(
            in_channels=d_features, out_channels=n_classes * n_groups,
            kernel_size=kernel_size, groups=n_groups, max_norm=max_norm,
        )
        # Zero init: equal group weights at the start of training.
        self.group_weights = nn.Parameter(torch.zeros(n_groups))

    def forward(self, x: Tensor) -> Tensor:                      # [B, d, 1] -> [B, n_classes]
        x = self.linear(x).squeeze(-1)
        x = x.view(x.size(0), self.n_groups, self.n_classes)
        weights = torch.softmax(self.group_weights, dim=0)
        return (x * weights.view(1, self.n_groups, 1)).sum(dim=1)


class TCNHead(nn.Module):
    def __init__(self, d_features: int = 64, n_groups: int = 1, tcn_depth: int = 2,
                 kernel_length: int = 4, dropout_tcn: float = 0.3, n_classes: int = 4):
        super().__init__()
        self.n_groups = n_groups
        self.n_classes = n_classes
        self.tcn = TCN(tcn_depth, kernel_length, d_features, n_groups, dropout_tcn)
        self.classifier = ClassificationHead(d_features=d_features, n_groups=n_groups,
                                             n_classes=n_classes)

    def extract_temporal_features(self, x):
        """Complete TCN output sequence before temporal pooling."""
        return self.tcn(x)

    @staticmethod
    def pool_temporal_features(x):
        """Average of the last causal state and the whole-trial mean."""
        return 0.5 * (x[:, :, -1] + x.mean(dim=-1))

    def extract_features(self, x):
        return self.pool_temporal_features(self.extract_temporal_features(x))

    def classify_features(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        return self.classifier(x)

    def forward(self, x):
        return self.classify_features(self.extract_features(x))


class DropPath(nn.Module):
    """Stochastic depth per sample (timm implementation)."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


@torch.jit.script
def _hoisted_selective_scan(values: Tensor, delta: Tensor, A: Tensor,
                            B: Tensor, C: Tensor, D: Tensor) -> Tensor:
    """Selective-scan recurrence with all state-independent work outside the time loop.

    h_t = exp(dt_t A) h_{t-1} + dt_t B_t x_t,  y_t = C_t h_t + D x_t.
    Same element-wise operation order as the step-by-step reference.
    """
    dt = delta.unsqueeze(-1)                                    # [b, L, d, 1]
    decay = torch.exp(dt * A)                                   # [b, L, d, n]
    drive = dt * B.unsqueeze(2) * values.unsqueeze(-1)          # [b, L, d, n]
    state = torch.zeros_like(drive[:, 0])
    states = torch.jit.annotate(List[Tensor], [])
    for step in range(values.size(1)):
        state = decay[:, step] * state + drive[:, step]
        states.append(state)
    readout = (torch.stack(states, dim=1) * C.unsqueeze(2)).sum(dim=-1)
    return readout + D * values


@torch.jit.script
def _segsum_selective_scan(values: Tensor, delta: Tensor, A: Tensor, B: Tensor, C: Tensor,
                           D: Tensor, chunk_size: int = 32) -> Tensor:
    """Selective-scan recurrence evaluated in parallel over time (closed form per chunk).

    Same recurrence as ``_hoisted_selective_scan``:
    h_t = exp(dt_t A) h_{t-1} + dt_t B_t x_t,  y_t = C_t h_t + D x_t.
    Unrolled inside a chunk that starts from the carried state h_0,
    h_t = exp(S_t) h_0 + sum_{s<=t} exp(S_t - S_s) drive_s,  with S_t = cumsum_{r<=t}(dt_r A),
    so every step of the chunk is computed at once ("segment sum", as in Mamba-2).
    All exponents are <= 0 (dt > 0, A < 0): nothing overflows, and an entry that underflows to 0
    is a state that has decayed away. Only the chunks run one after the other; with the
    ~17 pooled time steps of BCI IV-2a a single chunk holds the whole sequence.
    The sums run in a different order than the step-by-step loop, so the outputs agree
    with the hoisted scan up to float rounding, not bit for bit.
    """
    dt = delta.unsqueeze(-1)                                    # [b, L, d, 1]
    log_decay = dt * A                                          # [b, L, d, n], <= 0
    drive = dt * B.unsqueeze(2) * values.unsqueeze(-1)          # [b, L, d, n]
    length = values.size(1)
    state = torch.zeros_like(drive[:, 0])                       # [b, d, n]
    chunks = torch.jit.annotate(List[Tensor], [])
    for start in range(0, length, chunk_size):
        stop = min(start + chunk_size, length)
        cum = torch.cumsum(log_decay[:, start:stop], dim=1)     # S_t, [b, l, d, n]
        size = stop - start
        causal = torch.ones(size, size, dtype=torch.bool, device=values.device).tril()
        segment = cum.unsqueeze(2) - cum.unsqueeze(1)           # S_t - S_s, [b, t, s, d, n]
        segment = segment.masked_fill(~causal.view(1, size, size, 1, 1), float("-inf"))
        states = (torch.exp(segment) * drive[:, start:stop].unsqueeze(1)).sum(dim=2)
        states = states + torch.exp(cum) * state.unsqueeze(1)   # carried state, decayed to t
        state = states[:, -1]
        chunks.append(states)
    readout = (torch.cat(chunks, dim=1) * C.unsqueeze(2)).sum(dim=-1)
    return readout + D * values


SSM_SCANS = ("hoisted", "segsum", "sequential")


class SelectiveSSMMixer(nn.Module):
    """Dependency-free Mamba-style selective state-space mixer.

    Input-dependent step sizes and B/C projections select what is written to
    and read from a per-channel state; the diagonal negative A keeps the
    recurrence stable. ``scan="hoisted"`` (used for every experiment) evaluates
    the causal depth-wise convolution as shifted products and hoists the scan;
    ``scan="segsum"`` computes the same recurrence in parallel over time (closed form per chunk,
    see ``_segsum_selective_scan``); ``scan="sequential"`` is the step-by-step reference implementation.
    """

    def __init__(self, d_model: int, d_state: int = 8, d_conv: int = 3, scan: str = "hoisted"):
        super().__init__()
        if d_state < 1 or d_conv < 1:
            raise ValueError("ssm_d_state and ssm_d_conv must be positive.")
        if scan not in SSM_SCANS:
            raise ValueError(f"ssm_scan must be one of {SSM_SCANS}, got {scan!r}.")
        self.d_model = d_model
        self.d_state = d_state
        self.scan = scan
        self.in_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.local_conv = nn.Conv1d(d_model, d_model, d_conv, groups=d_model, padding=d_conv - 1)
        self.x_proj = nn.Linear(d_model, d_model + 2 * d_state, bias=False)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).unsqueeze(0).repeat(d_model, 1)
        )
        self.D = nn.Parameter(torch.ones(d_model))
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        length = x.size(1)
        values, gate = self.in_proj(x).chunk(2, dim=-1)
        if self.scan in ("hoisted", "segsum"):
            values = self._shifted_causal_conv(values)
        else:
            values = self.local_conv(values.transpose(1, 2))[..., :length].transpose(1, 2)
        values = torch.nn.functional.silu(values)
        delta_raw, B, C = torch.split(self.x_proj(values), [self.d_model, self.d_state, self.d_state], dim=-1)
        delta = torch.nn.functional.softplus(delta_raw).clamp(max=1.0)
        A = -torch.exp(self.A_log).to(dtype=x.dtype)
        if self.scan == "hoisted":
            y = _hoisted_selective_scan(values, delta, A, B, C, self.D)
        elif self.scan == "segsum":
            y = _segsum_selective_scan(values, delta, A, B, C, self.D)
        else:
            y = self._sequential_scan(values, delta, A, B, C)
        y = y * torch.nn.functional.silu(gate)
        return self.out_proj(y)

    def _shifted_causal_conv(self, values: Tensor) -> Tensor:
        """local_conv(values.transpose(1, 2))[..., :L].transpose(1, 2) as shifted products."""
        length = values.size(1)
        weight = self.local_conv.weight[:, 0, :]                 # [d, k]
        kernel_size = weight.size(1)
        padded = torch.nn.functional.pad(values, (0, 0, kernel_size - 1, 0))
        out = self.local_conv.bias + padded[:, :length] * weight[:, 0]
        for offset in range(1, kernel_size):
            out = out + padded[:, offset:offset + length] * weight[:, offset]
        return out

    def _sequential_scan(self, values: Tensor, delta: Tensor, A: Tensor, B: Tensor, C: Tensor) -> Tensor:
        batch, length, _ = values.shape
        state = values.new_zeros(batch, self.d_model, self.d_state)
        outputs = []
        for step in range(length):
            dt = delta[:, step, :].unsqueeze(-1)
            decay = torch.exp(dt * A.unsqueeze(0))
            state = decay * state + dt * B[:, step, :].unsqueeze(1) * values[:, step, :].unsqueeze(-1)
            readout = (state * C[:, step, :].unsqueeze(1)).sum(dim=-1)
            outputs.append(readout + self.D * values[:, step, :])
        return torch.stack(outputs, dim=1)


class SelectiveSSMBlock(nn.Module):
    """Pre-norm residual selective SSM, optionally scanning time backwards."""

    def __init__(self, d_model: int, d_state: int = 8, d_conv: int = 3, dropout: float = 0.4,
                 drop_path_rate: float = 0.0, reverse: bool = False, scan: str = "hoisted"):
        super().__init__()
        self.reverse = reverse
        self.norm = nn.LayerNorm(d_model)
        self.mixer = SelectiveSSMMixer(d_model, d_state, d_conv, scan)
        self.dropout = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path_rate)

    def forward(self, x: Tensor) -> Tensor:
        mixed = self.norm(x)
        if self.reverse:
            mixed = torch.flip(mixed, dims=(1,))
        mixed = self.mixer(mixed)
        if self.reverse:
            mixed = torch.flip(mixed, dims=(1,))
        return x + self.drop_path(self.dropout(mixed))


class SSMBackbone(nn.Module):
    """Convolutional stem -> selective SSM encoder -> grouped TCN head.

    ``extract_features`` returns the pooled feature vector of dimension
    ``d_group * (n_kernel_groups + 1)`` used by the adaptation head.
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        F1: int = 32,
        temp_kernel_lengths: Sequence[int] = (20, 32, 64),
        pool_length_1: int = 8,
        pool_length_2: int = 7,
        D: int = 2,
        dropout_conv: float = 0.4,
        d_group: int = 16,
        tcn_depth: int = 2,
        kernel_length_tcn: int = 4,
        dropout_tcn: float = 0.3,
        use_group_attn: bool = True,
        ssm_directions: Sequence[str] = SSM_DIRECTIONS,
        ssm_dropout: float = 0.4,
        drop_path_max: float = 0.25,
        ssm_d_state: int = 8,
        ssm_d_conv: int = 3,
        ssm_scan: str = "hoisted",
    ):
        super().__init__()
        unknown = set(ssm_directions) - {"forward", "backward"}
        if unknown or not ssm_directions:
            raise ValueError(f"ssm_directions must be a non-empty list of forward/backward, got {ssm_directions}")
        self.n_classes = n_classes
        self.n_groups = len(temp_kernel_lengths)
        self.d_model = d_group * self.n_groups
        self.feature_dim = d_group * (self.n_groups + 1)

        self.rearrange = Rearrange("b c seq -> b seq c")
        self.conv_block = MultiKernelConvBlock(n_channels, temp_kernel_lengths, F1, D, pool_length_1,
                                               pool_length_2, dropout_conv, d_group, use_group_attn)
        self.mix = nn.Sequential(
            nn.Conv1d(self.d_model, self.d_model, kernel_size=1, groups=1, bias=False),
            nn.BatchNorm1d(self.d_model),
            nn.SiLU(),
        )
        # Drop-path rate grows quadratically from 0 to drop_path_max with depth.
        drop_rates = torch.linspace(0, 1, len(ssm_directions)) ** 2 * drop_path_max
        self.ssm_blocks = nn.ModuleList(
            SelectiveSSMBlock(self.d_model, d_state=ssm_d_state, d_conv=ssm_d_conv, scan=ssm_scan,
                              dropout=ssm_dropout, drop_path_rate=drop_rates[i].item(),
                              reverse=direction == "backward")
            for i, direction in enumerate(ssm_directions)
        )
        self.reduce = nn.Sequential(
            Rearrange("b t c -> b c t"),
            nn.Conv1d(self.d_model, d_group, kernel_size=1, groups=1, bias=False),
            nn.BatchNorm1d(d_group),
            nn.SiLU(),
        )
        self.tcn_head = TCNHead(self.feature_dim, self.n_groups + 1, tcn_depth,
                                kernel_length_tcn, dropout_tcn, n_classes)

    def extract_temporal_features(self, x):                     # x: [B, C, T]
        conv_features = self.conv_block(x)
        tokens = self.rearrange(self.mix(conv_features))
        for block in self.ssm_blocks:
            tokens = block(tokens)
        ssm_features = self.reduce(tokens)
        features = torch.cat((conv_features, ssm_features), dim=1)
        return self.tcn_head.extract_temporal_features(features)

    def extract_features(self, x):
        return self.tcn_head.pool_temporal_features(self.extract_temporal_features(x))

    def classify_features(self, features):
        return self.tcn_head.classify_features(features)

    def forward(self, x):
        return self.classify_features(self.extract_features(x))


BACKBONE_KWARGS = (
    "F1", "temp_kernel_lengths", "pool_length_1", "pool_length_2", "D", "dropout_conv", "d_group",
    "tcn_depth", "kernel_length_tcn", "dropout_tcn", "use_group_attn", "ssm_directions",
    "ssm_dropout", "drop_path_max", "ssm_d_state", "ssm_d_conv", "ssm_scan",
)


def split_backbone_kwargs(kwargs: dict):
    """(backbone kwargs, remaining kwargs)."""
    backbone = {k: kwargs.pop(k) for k in BACKBONE_KWARGS if k in kwargs}
    return backbone, kwargs


class SSMSourceOnly(ClassificationModule):
    """The SSM-DAN backbone trained on labeled source data only (no adaptation head)."""

    def __init__(self, n_channels: int, n_classes: int, **kwargs):
        backbone, kwargs = split_backbone_kwargs(dict(kwargs))
        super().__init__(SSMBackbone(n_channels=n_channels, n_classes=n_classes, **backbone),
                         n_classes, **backbone, **kwargs)


def rename_legacy_keys(state_dict: dict) -> dict:
    """Map checkpoints of the development branches (``transformer.*``) to ``ssm_blocks.*``."""
    return {key.replace("model.transformer.", "model.ssm_blocks."): value
            for key, value in state_dict.items()}
