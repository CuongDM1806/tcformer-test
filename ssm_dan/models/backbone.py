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


def _complex_linear_scan(decay_re: Tensor, decay_im: Tensor, drive: Tensor, drive_im: Tensor = None):
    """h_t = a_t h_{t-1} + x_t over the last axis, h_{-1} = 0, with complex a_t and x_t
    (x_t real unless ``drive_im`` is given).

    Hillis-Steele parallel scan (log2 T levels of element-wise operations): element t
    holds the affine map h -> a h + x; each level composes it with the map d steps
    earlier. Only products of |a| <= 1 are formed, so the scan is numerically stable.
    Complex numbers are kept as (real, imaginary) pairs. Returns (Re h, Im h).
    """
    pad = torch.nn.functional.pad
    ar, ai = decay_re, decay_im
    xr, xi = drive, (torch.zeros_like(drive) if drive_im is None else drive_im)
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


def _oscillator_scan(log_decay: Tensor, omega: Tensor, drive: Tensor, chunk: int = 16):
    """Same recurrence as ``_complex_linear_scan`` for a_t = exp(-z_t + i omega), i.e. a fixed
    rotation per state and an input-dependent decay z_t >= 0 (``log_decay``), in O(T) work.

    Within a chunk of length K starting at c, with l_t = sum_{c<=tau<=t} z_tau and k_t = t - c + 1,
        h_t = exp(-l_t + i omega k_t) [h_{c-1} + sum_{c<=s<=t} exp(l_s - i omega k_s) x_s],
    i.e. one cumulative sum per chunk. exp(l_s) stays below exp(K max z), finite in float32 for
    K * max z < 88 (K = 16 and z <= 4 * pi / 4 here). The chunk-end states are then linked by the
    Hillis-Steele scan over T / K chunks.

    log_decay, drive: [B, N, T]; omega: [1, N, 1] (rad/sample). Returns (Re h, Im h), [B, N, T].
    """
    batch, states, length = drive.shape
    pad = (-length) % chunk
    if pad:                                                     # trailing padding never reaches earlier steps
        log_decay = torch.nn.functional.pad(log_decay, (0, pad))
        drive = torch.nn.functional.pad(drive, (0, pad))
    n_chunks = (length + pad) // chunk
    z = log_decay.reshape(batch, states, n_chunks, chunk)
    x = drive.reshape(batch, states, n_chunks, chunk)
    ell = z.cumsum(dim=-1)                                      # l_t
    k = torch.arange(1, chunk + 1, device=drive.device, dtype=drive.dtype)
    angle = omega.unsqueeze(-1) * k                             # [1, N, 1, K]
    cos_k, sin_k = torch.cos(angle), torch.sin(angle)
    grow = torch.exp(ell) * x
    sum_re, sum_im = (grow * cos_k).cumsum(dim=-1), (-grow * sin_k).cumsum(dim=-1)
    shrink = torch.exp(-ell)
    prop_re, prop_im = shrink * cos_k, shrink * sin_k           # exp(-l_t + i omega k_t)
    local_re = prop_re * sum_re - prop_im * sum_im              # state with h_{c-1} = 0
    local_im = prop_re * sum_im + prop_im * sum_re
    # Chunk-end states: H_c = A_c H_{c-1} + Y_c.
    carry_re, carry_im = _complex_linear_scan(prop_re[..., -1], prop_im[..., -1],
                                              local_re[..., -1], local_im[..., -1])
    pad1 = torch.nn.functional.pad
    prev_re = pad1(carry_re[..., :-1], (1, 0)).unsqueeze(-1)    # h_{c-1}
    prev_im = pad1(carry_im[..., :-1], (1, 0)).unsqueeze(-1)
    h_re = local_re + prop_re * prev_re - prop_im * prev_im
    h_im = local_im + prop_re * prev_im + prop_im * prev_re
    return (h_re.reshape(batch, states, -1)[..., :length],
            h_im.reshape(batch, states, -1)[..., :length])


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
    a complex exponential at the centre frequency a unit response, so |h_t|^2 tracks the
    instantaneous power of u in band k. Window means of this power (the windows of the convolutional stem) are
    log-compressed and read out to ``d_out`` features per time step.

    Every parameter that shapes the features can be read in physical units
    (``interpretation``): spatial filters per virtual channel, centre frequency and
    bandwidth per state, and the weight of each state in the readout.
    """

    def __init__(self, n_channels: int, d_out: int, sfreq: float, n_virtual: int = 16, n_modes: int = 8,
                 f_min: float = 4.0, f_max: float = 40.0, bandwidth_ratio: float = 0.4,
                 delta_kernel: int = 32, pool_lengths: Sequence[int] = (8, 7), dropout: float = 0.4,
                 scan: str = "parallel", chunk: int = 16):
        super().__init__()
        if sfreq is None or sfreq <= 0:
            raise ValueError("OscillatorySSM needs the sampling rate (osc_sfreq).")
        if scan not in ("parallel", "chunked"):
            raise ValueError(f"osc_scan must be parallel | chunked, got {scan!r}")
        self.scan, self.chunk = scan, int(chunk)
        nyquist = sfreq / 2
        self.sfreq = float(sfreq)
        self.n_virtual, self.n_modes = n_virtual, n_modes
        self.f_low, self.f_high = 1.0, min(45.0, 0.9 * nyquist)     # frequency range of a state
        self.b_low, self.b_high = 0.5, 0.5 * nyquist                  # bandwidth range
        f_max = min(f_max, 0.95 * self.f_high)
        self.spatial = nn.Conv1d(n_channels, n_virtual, 1, bias=False)
        # Centre frequencies log-spaced in [f_min, f_max], the same grid for every virtual channel.
        freqs = torch.logspace(math.log10(f_min), math.log10(f_max), n_modes).repeat(n_virtual, 1)
        self.freq_logit = nn.Parameter(self._logit((freqs - self.f_low) / (self.f_high - self.f_low)))
        bandwidths = (bandwidth_ratio * freqs).clamp(min=1.0)
        self.bandwidth_logit = nn.Parameter(self._logit((bandwidths - self.b_low) / (self.b_high - self.b_low)))
        # Selection: Delta_t from the virtual channel's own causal neighbourhood; zero-initialized
        # read-out weights and a bias of softplus^-1(1) give Delta_t = 1 at the start.
        self.delta_kernel = delta_kernel
        self.delta_conv = nn.Conv1d(n_virtual, n_virtual, delta_kernel, groups=n_virtual)
        self.delta_proj = nn.Conv1d(n_virtual, n_virtual * n_modes, 1, groups=n_virtual)
        nn.init.zeros_(self.delta_proj.weight)
        nn.init.constant_(self.delta_proj.bias, math.log(math.e - 1.0))
        self.pool = nn.Sequential(*[nn.AvgPool1d(p) for p in pool_lengths])
        self.power_norm = nn.BatchNorm1d(n_virtual * n_modes)
        self.readout = nn.Sequential(
            nn.Conv1d(n_virtual * n_modes, d_out, 1, bias=False),
            nn.BatchNorm1d(d_out),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

    @staticmethod
    def _logit(p: Tensor) -> Tensor:
        p = p.clamp(1e-4, 1 - 1e-4)
        return torch.log(p) - torch.log1p(-p)

    def frequencies(self) -> Tensor:                            # [S, K] in Hz
        return self.f_low + (self.f_high - self.f_low) * torch.sigmoid(self.freq_logit)

    def bandwidths(self) -> Tensor:                             # [S, K] in Hz
        return self.b_low + (self.b_high - self.b_low) * torch.sigmoid(self.bandwidth_logit)

    def delta(self, u: Tensor) -> Tensor:                       # u: [B, S, T] -> [B, S*K, T]
        context = torch.nn.functional.pad(u, (self.delta_kernel - 1, 0))     # causal
        z = self.delta_proj(torch.nn.functional.silu(self.delta_conv(context)))
        return torch.nn.functional.softplus(z).clamp(0.05, 4.0)

    def band_power(self, x: Tensor) -> Tensor:
        """Instantaneous power |h_t|^2 of every state: [B, S*K, T]."""
        u = self.spatial(x)                                     # [B, S, T]
        batch, _, length = u.shape
        omega = (2 * math.pi / self.sfreq) * self.frequencies().reshape(1, -1, 1)
        damping = (math.pi / self.sfreq) * self.bandwidths().reshape(1, -1, 1)
        log_decay = self.delta(u) * damping                     # -log r_t >= 0
        radius = torch.exp(-log_decay)                          # r_t in (0, 1)
        drive = (1 - radius) * u.repeat_interleave(self.n_modes, dim=1)
        if self.scan == "chunked":
            h_re, h_im = _oscillator_scan(log_decay, omega, drive, self.chunk)
        else:
            h_re, h_im = _complex_linear_scan(radius * torch.cos(omega), radius * torch.sin(omega), drive)
        return h_re ** 2 + h_im ** 2

    def forward(self, x: Tensor) -> Tensor:                     # x: [B, C, T] -> [B, d_out, T']
        log_power = torch.log(self.pool(self.band_power(x)) + 1e-6)
        return self.readout(self.power_norm(log_power))

    @torch.no_grad()
    def interpretation(self) -> dict:
        """Physical read-out of the branch (lists, for the run's JSON record)."""
        conv, out_norm = self.readout[0], self.power_norm
        # Weight of each state in the readout, through the batch normalization before it.
        scale = out_norm.weight.abs() / torch.sqrt(out_norm.running_var + out_norm.eps)
        weight = conv.weight[:, :, 0].norm(dim=0) * scale
        shape = (self.n_virtual, self.n_modes)
        return {
            "sfreq": self.sfreq,
            "frequency_hz": self.frequencies().cpu().tolist(),
            "bandwidth_hz": self.bandwidths().cpu().tolist(),
            "readout_weight": weight.reshape(shape).cpu().tolist(),
            "spatial_filters": self.spatial.weight[:, :, 0].cpu().tolist(),       # [S, C]
        }


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
        osc_scan: str = "parallel",
        osc_chunk: int = 16,
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
                                      scan=osc_scan, chunk=osc_chunk)

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
    "osc_bandwidth_ratio", "osc_delta_kernel", "osc_dropout", "osc_scan", "osc_chunk",
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
