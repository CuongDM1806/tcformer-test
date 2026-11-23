"""SSM-DAN backbone: multi-scale convolution, bidirectional selective SSM, grouped TCN.

The convolutional stem and the grouped TCN head follow TCFormer
(Altaheri et al., Sci. Rep. 2025, https://github.com/altaheri/TCFormer, MIT
license); the attention encoder of TCFormer is replaced by a stack of
lightweight selective state-space (Mamba-style) blocks with alternating scan
direction.
"""
import math
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


class SelectiveSSMMixer(nn.Module):
    """Dependency-free Mamba-style selective state-space mixer.

    Input-dependent step sizes and B/C projections select what is written to
    and read from a per-channel state; the diagonal negative A keeps the
    recurrence stable. ``scan="hoisted"`` (used for every experiment) evaluates
    the causal depth-wise convolution as shifted products and hoists the scan;
    ``scan="sequential"`` is the step-by-step reference implementation.
    """

    def __init__(self, d_model: int, d_state: int = 8, d_conv: int = 3, scan: str = "hoisted"):
        super().__init__()
        if d_state < 1 or d_conv < 1:
            raise ValueError("ssm_d_state and ssm_d_conv must be positive.")
        if scan not in ("hoisted", "sequential"):
            raise ValueError("ssm_scan must be 'hoisted' or 'sequential'.")
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
        if self.scan == "hoisted":
            values = self._shifted_causal_conv(values)
        else:
            values = self.local_conv(values.transpose(1, 2))[..., :length].transpose(1, 2)
        values = torch.nn.functional.silu(values)
        delta_raw, B, C = torch.split(self.x_proj(values), [self.d_model, self.d_state, self.d_state], dim=-1)
        delta = torch.nn.functional.softplus(delta_raw).clamp(max=1.0)
        A = -torch.exp(self.A_log).to(dtype=x.dtype)
        if self.scan == "hoisted":
            y = _hoisted_selective_scan(values, delta, A, B, C, self.D)
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


def _complex_linear_scan(decay_re: Tensor, decay_im: Tensor, drive: Tensor):
    """h_t = a_t h_{t-1} + x_t over the last axis, h_{-1} = 0, with complex a_t and real x_t.

    Hillis-Steele parallel scan (log2 T levels of element-wise operations): element t
    holds the affine map h -> a h + x; each level composes it with the map d steps
    earlier. Only products of |a| <= 1 are formed, so the scan is numerically stable.
    Complex numbers are kept as (real, imaginary) pairs. Returns (Re h, Im h).
    """
    pad = torch.nn.functional.pad
    ar, ai = decay_re, decay_im
    xr, xi = drive, torch.zeros_like(drive)
    length, step = drive.size(-1), 1
    while step < length:
        sxr, sxi = pad(xr[..., :-step], (step, 0)), pad(xi[..., :-step], (step, 0))
        xr, xi = xr + ar * sxr - ai * sxi, xi + ar * sxi + ai * sxr
        if step * 2 < length:                                   # the last level needs no new a
            sar = pad(ar[..., :-step], (step, 0), value=1.0)
            sai = pad(ai[..., :-step], (step, 0))
            ar, ai = ar * sar - ai * sai, ar * sai + ai * sar
        step *= 2
    return xr, xi


class SyncProbes(nn.Module):
    """Phase synchrony between virtual channels within each band.

    For band k and probe j, with unit phasors p_{s,k}(t) = h_{s,k}(t) / sqrt(|h_{s,k}(t)|^2 + eps):

        R_{j,k}(t) = |sum_s w_{j,s,k} p_{s,k}(t)| / (sum_s |w_{j,s,k}| + eps)  in [0, 1],
        w_{j,s,k} = softplus(a_{j,s,k}) exp(i phi_{j,s,k}).

    R ignores amplitudes. The learned phases phi absorb the arbitrary sign of a spatial filter
    and fixed phase lags between sources, so R measures how stable the phase relation is,
    not whether the phases coincide. Complex numbers are (real, imaginary) pairs.
    """

    def __init__(self, n_virtual: int, n_modes: int, n_probes: int = 3, hemi_split: bool = False,
                 eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        if hemi_split and n_probes >= 2:
            a = torch.full((n_probes, n_virtual, n_modes), -3.0)
            half = n_virtual // 2
            a[0, :half] = 2.0                                   # left-hemisphere virtual channels
            a[1, half:] = 2.0                                   # right-hemisphere virtual channels
            if n_probes >= 3:
                a[2:] = 2.0                                     # global probes
        else:
            a = 0.5 * torch.randn(n_probes, n_virtual, n_modes)
        self.amp_logit = nn.Parameter(a)
        self.phase = nn.Parameter(torch.zeros(n_probes, n_virtual, n_modes))

    def amplitudes(self) -> Tensor:                             # [J, S, K]
        return torch.nn.functional.softplus(self.amp_logit)

    def forward(self, h_re: Tensor, h_im: Tensor) -> Tensor:    # [B, S, K, T] -> [B, J*K, T]
        mag = torch.sqrt(h_re ** 2 + h_im ** 2 + self.eps)
        pr, pi = h_re / mag, h_im / mag
        amp = self.amplitudes()
        wr, wi = amp * torch.cos(self.phase), amp * torch.sin(self.phase)
        sr = torch.einsum('jsk,bskt->bjkt', wr, pr) - torch.einsum('jsk,bskt->bjkt', wi, pi)
        si = torch.einsum('jsk,bskt->bjkt', wr, pi) + torch.einsum('jsk,bskt->bjkt', wi, pr)
        R = torch.sqrt(sr ** 2 + si ** 2 + self.eps) / (amp.sum(dim=1)[None, :, :, None] + self.eps)
        return R.flatten(1, 2)                                  # channel order j*K + k


def channel_sides(names: Sequence[str], midline_m: float = 0.005) -> List[int]:
    """-1 (left), 0 (midline), +1 (right) for each channel, from the x coordinate of the
    standard_1020 montage (case-insensitive names)."""
    import mne
    positions = mne.channels.make_standard_montage("standard_1020").get_positions()["ch_pos"]
    by_name = {name.lower(): xyz for name, xyz in positions.items()}
    missing = [n for n in names if n.lower() not in by_name]
    if missing:
        raise KeyError(f"Channels without a standard_1020 position: {missing}")
    sides = []
    for name in names:
        x = float(by_name[name.lower()][0])
        sides.append(0 if abs(x) < midline_m else (-1 if x < 0 else 1))
    return sides


class OscillatorySSM(nn.Module):
    """Selective oscillatory state-space branch that runs at the sampling rate of the signal.

    Spatial filters W project the C electrodes onto S virtual channels u_t = W x_t.
    Each virtual channel drives K complex states, damped oscillators with a learned
    frequency f_k and bandwidth b_k (in Hz):

        h_t = r_t exp(i 2 pi f_k / fs) h_{t-1} + (1 - r_t) u_t,   r_t = exp(-Delta_t pi b_k / fs).

    The rotation, i.e. the frequency of a state, never depends on the input; only the
    step Delta_t does (selection, as in Mamba): Delta_t > 1 forgets faster and widens the
    band, Delta_t < 1 integrates longer. Delta_t is computed from the virtual channel's own
    recent signal and starts at exactly 1 (a fixed resonator bank). The gain 1 - r_t gives
    a complex exponential at the centre frequency a unit response (no phase shift, whatever
    the bandwidth), so |h_t|^2 tracks the instantaneous power of u in band k.

    Readout (``sync_mode``): ``"none"`` window means of log power (S*K features); ``"concat"``
    adds window means of the phase synchrony R of ``SyncProbes`` (J*K features); ``"only"``
    keeps R alone. With synchrony the centre frequencies are shared by all virtual channels
    (``tie_freq``, automatic), since synchrony compares states of the same frequency.
    ``channel_side`` (-1/0/+1 per electrode) restricts the first S/2 virtual channels to the
    left and midline electrodes and the others to the midline and right ones; the probes then
    start as left, right and global probes.

    Every parameter that shapes the features can be read in physical units
    (``interpretation``).
    """

    def __init__(self, n_channels: int, d_out: int, sfreq: float, n_virtual: int = 16, n_modes: int = 8,
                 f_min: float = 4.0, f_max: float = 40.0, bandwidth_ratio: float = 0.4,
                 delta_kernel: int = 32, pool_lengths: Sequence[int] = (8, 7), dropout: float = 0.4,
                 sync_mode: str = "none", n_probes: int = 3, tie_freq: bool = None,
                 channel_side: Sequence[int] = None):
        super().__init__()
        if sfreq is None or sfreq <= 0:
            raise ValueError("OscillatorySSM needs the sampling rate (osc_sfreq).")
        if sync_mode not in ("none", "concat", "only"):
            raise ValueError(f"osc_sync must be none | concat | only, got {sync_mode!r}")
        nyquist = sfreq / 2
        self.sfreq = float(sfreq)
        self.n_virtual, self.n_modes = n_virtual, n_modes
        self.sync_mode = sync_mode
        self.tie_freq = (sync_mode != "none") if tie_freq is None else bool(tie_freq)
        self.f_low, self.f_high = 1.0, min(45.0, 0.9 * nyquist)     # frequency range of a state
        self.b_low, self.b_high = 0.5, 0.5 * nyquist                  # bandwidth range
        f_max = min(f_max, 0.95 * self.f_high)
        self.spatial = nn.Conv1d(n_channels, n_virtual, 1, bias=False)
        # Centre frequencies log-spaced in [f_min, f_max]: one grid shared by every virtual channel
        # (tie_freq) or one copy of it per virtual channel.
        grid = torch.logspace(math.log10(f_min), math.log10(f_max), n_modes)
        freqs = grid.unsqueeze(0) if self.tie_freq else grid.repeat(n_virtual, 1)
        self.freq_logit = nn.Parameter(self._logit((freqs - self.f_low) / (self.f_high - self.f_low)))
        bandwidths = (bandwidth_ratio * grid.repeat(n_virtual, 1)).clamp(min=1.0)
        self.bandwidth_logit = nn.Parameter(self._logit((bandwidths - self.b_low) / (self.b_high - self.b_low)))
        # Selection: Delta_t from the virtual channel's own causal neighbourhood; zero-initialized
        # read-out weights and a bias of softplus^-1(1) give Delta_t = 1 at the start.
        self.delta_kernel = delta_kernel
        self.delta_conv = nn.Conv1d(n_virtual, n_virtual, delta_kernel, groups=n_virtual)
        self.delta_proj = nn.Conv1d(n_virtual, n_virtual * n_modes, 1, groups=n_virtual)
        nn.init.zeros_(self.delta_proj.weight)
        nn.init.constant_(self.delta_proj.bias, math.log(math.e - 1.0))
        self.pool = nn.Sequential(*[nn.AvgPool1d(p) for p in pool_lengths])
        self.n_power = 0 if sync_mode == "only" else n_virtual * n_modes
        self.n_sync = 0 if sync_mode == "none" else n_probes * n_modes
        n_features = self.n_power + self.n_sync
        self.power_norm = nn.BatchNorm1d(n_features)
        self.readout = nn.Sequential(
            nn.Conv1d(n_features, d_out, 1, bias=False),
            nn.BatchNorm1d(d_out),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.hemispheric = channel_side is not None
        if self.hemispheric:
            side = torch.as_tensor(list(channel_side), dtype=torch.float32)
            if side.numel() != n_channels or not set(side.tolist()) <= {-1.0, 0.0, 1.0}:
                raise ValueError("osc_channel_side needs one value -1, 0 or +1 per channel.")
            half = n_virtual // 2
            mask = torch.zeros(n_virtual, n_channels)
            mask[:half] = (side <= 0).float()                   # left + midline
            mask[half:] = (side >= 0).float()                   # midline + right
            self.register_buffer("spatial_mask", mask)
        if sync_mode != "none":                                 # absent otherwise: old state_dict kept
            self.sync = SyncProbes(n_virtual, n_modes, n_probes, hemi_split=self.hemispheric)

    @staticmethod
    def _logit(p: Tensor) -> Tensor:
        p = p.clamp(1e-4, 1 - 1e-4)
        return torch.log(p) - torch.log1p(-p)

    def frequencies(self) -> Tensor:                            # [S, K] in Hz
        freqs = self.f_low + (self.f_high - self.f_low) * torch.sigmoid(self.freq_logit)
        return freqs.expand(self.n_virtual, self.n_modes)

    def bandwidths(self) -> Tensor:                             # [S, K] in Hz
        return self.b_low + (self.b_high - self.b_low) * torch.sigmoid(self.bandwidth_logit)

    def spatial_weight(self) -> Tensor:                         # [S, C, 1], masked if hemispheric
        weight = self.spatial.weight
        return weight * self.spatial_mask[:, :, None] if self.hemispheric else weight

    def delta(self, u: Tensor) -> Tensor:                       # u: [B, S, T] -> [B, S*K, T]
        context = torch.nn.functional.pad(u, (self.delta_kernel - 1, 0))     # causal
        z = self.delta_proj(torch.nn.functional.silu(self.delta_conv(context)))
        return torch.nn.functional.softplus(z).clamp(0.05, 4.0)

    def band_state(self, x: Tensor):
        """(Re h, Im h), each [B, S*K, T]; channel order s*K + k (as repeat_interleave and
        frequencies().reshape)."""
        u = torch.nn.functional.conv1d(x, self.spatial_weight()) if self.hemispheric else self.spatial(x)
        omega = (2 * math.pi / self.sfreq) * self.frequencies().reshape(1, -1, 1)
        damping = (math.pi / self.sfreq) * self.bandwidths().reshape(1, -1, 1)
        radius = torch.exp(-self.delta(u) * damping)            # r_t in (0, 1)
        drive = (1 - radius) * u.repeat_interleave(self.n_modes, dim=1)
        return _complex_linear_scan(radius * torch.cos(omega), radius * torch.sin(omega), drive)

    def band_power(self, x: Tensor) -> Tensor:
        """Instantaneous power |h_t|^2 of every state: [B, S*K, T]."""
        h_re, h_im = self.band_state(x)
        return h_re ** 2 + h_im ** 2

    def synchrony(self, h_re: Tensor, h_im: Tensor) -> Tensor:  # [B, S*K, T] -> [B, J*K, T]
        shape = (h_re.size(0), self.n_virtual, self.n_modes, h_re.size(-1))
        return self.sync(h_re.reshape(shape), h_im.reshape(shape))

    def forward(self, x: Tensor) -> Tensor:                     # x: [B, C, T] -> [B, d_out, T']
        h_re, h_im = self.band_state(x)
        features = []
        if self.sync_mode != "only":
            features.append(torch.log(self.pool(h_re ** 2 + h_im ** 2) + 1e-6))   # [B, S*K, T']
        if self.sync_mode != "none":
            features.append(self.pool(self.synchrony(h_re, h_im)))               # [B, J*K, T'], linear mean
        return self.readout(self.power_norm(torch.cat(features, dim=1)))

    @torch.no_grad()
    def interpretation(self) -> dict:
        """Physical read-out of the branch (lists, for the run's JSON record)."""
        conv, out_norm = self.readout[0], self.power_norm
        # Weight of each feature in the readout, through the batch normalization before it.
        scale = out_norm.weight.abs() / torch.sqrt(out_norm.running_var + out_norm.eps)
        weight = conv.weight[:, :, 0].norm(dim=0) * scale
        result = {
            "sfreq": self.sfreq,
            "sync_mode": self.sync_mode,
            "tie_freq": self.tie_freq,
            "frequency_hz": self.frequencies().cpu().tolist(),
            "bandwidth_hz": self.bandwidths().cpu().tolist(),
            "spatial_filters": self.spatial_weight()[:, :, 0].cpu().tolist(),     # [S, C]
        }
        if self.n_power:
            result["readout_weight"] = weight[:self.n_power].reshape(self.n_virtual, self.n_modes).cpu().tolist()
        if self.n_sync:
            n_probes = self.n_sync // self.n_modes
            result["sync_readout_weight"] = weight[self.n_power:].reshape(n_probes, self.n_modes).cpu().tolist()
            result["sync_probe_amplitude"] = self.sync.amplitudes().cpu().tolist()          # [J, S, K]
            result["sync_probe_phase_rad"] = self.sync.phase.cpu().tolist()                 # [J, S, K]
        if self.hemispheric:
            result["spatial_mask"] = self.spatial_mask.cpu().tolist()
        return result


class SSMBackbone(nn.Module):
    """Convolutional stem -> selective SSM encoder -> grouped TCN head.

    ``extract_features`` returns the pooled feature vector of dimension
    ``d_group * (n_kernel_groups + 1)`` used by the adaptation head.

    ``osc_branch`` adds the signal-rate ``OscillatorySSM`` (default ``"none"``: the paper's backbone,
    built exactly as before):

    * ``"feed"``: two time scales. The oscillatory band powers join the convolutional tokens at the
      input of the token-rate selective SSM, whose output forms the last feature group as before.
    * ``"replace"``: the oscillatory branch forms the last feature group itself; no token SSM.
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
        osc_branch: str = "none",
        osc_sfreq: float = None,
        osc_virtual_channels: int = 16,
        osc_modes: int = 8,
        osc_f_min: float = 4.0,
        osc_f_max: float = 40.0,
        osc_bandwidth_ratio: float = 0.4,
        osc_delta_kernel: int = 32,
        osc_dropout: float = 0.4,
        osc_sync: str = "none",
        osc_sync_probes: int = 3,
        osc_tie_freq: bool = None,
        osc_channel_side: Sequence[int] = None,
    ):
        super().__init__()
        if osc_branch not in ("none", "feed", "replace"):
            raise ValueError(f"osc_branch must be none | feed | replace, got {osc_branch!r}")
        self.osc_branch = osc_branch
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
        if osc_branch != "replace":
            mix_in = self.d_model + (d_group if osc_branch == "feed" else 0)
            self.mix = nn.Sequential(
                nn.Conv1d(mix_in, self.d_model, kernel_size=1, groups=1, bias=False),
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
        if osc_branch != "none":
            self.osc = OscillatorySSM(n_channels, d_group, osc_sfreq, n_virtual=osc_virtual_channels,
                                      n_modes=osc_modes, f_min=osc_f_min, f_max=osc_f_max,
                                      bandwidth_ratio=osc_bandwidth_ratio, delta_kernel=osc_delta_kernel,
                                      pool_lengths=(pool_length_1, pool_length_2), dropout=osc_dropout,
                                      sync_mode=osc_sync, n_probes=osc_sync_probes, tie_freq=osc_tie_freq,
                                      channel_side=osc_channel_side)

    def extract_temporal_features(self, x):                     # x: [B, C, T]
        conv_features = self.conv_block(x)
        if self.osc_branch == "replace":
            ssm_features = self.osc(x)
        else:
            tokens = conv_features
            if self.osc_branch == "feed":
                tokens = torch.cat((tokens, self.osc(x)), dim=1)
            tokens = self.rearrange(self.mix(tokens))
            for block in self.ssm_blocks:
                tokens = block(tokens)
            ssm_features = self.reduce(tokens)
        features = torch.cat((conv_features, ssm_features), dim=1)
        return self.tcn_head.extract_temporal_features(features)

    def interpretation(self):
        """Learned frequencies, bandwidths, readout weights and spatial filters of the
        oscillatory branch (None without it)."""
        return self.osc.interpretation() if self.osc_branch != "none" else None

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
    "osc_branch", "osc_sfreq", "osc_virtual_channels", "osc_modes", "osc_f_min", "osc_f_max",
    "osc_bandwidth_ratio", "osc_delta_kernel", "osc_dropout",
    "osc_sync", "osc_sync_probes", "osc_tie_freq", "osc_channel_side",
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
