"""OscillatorySSM as committed in 5430ead1 (before phase synchrony): the reference for the
backward-compatibility test. Do not edit."""
import math
from typing import Sequence

import torch
from torch import Tensor, nn

from ssm_dan.models.backbone import _complex_linear_scan


class ReferenceOscillatorySSM(nn.Module):
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
                 delta_kernel: int = 32, pool_lengths: Sequence[int] = (8, 7), dropout: float = 0.4):
        super().__init__()
        if sfreq is None or sfreq <= 0:
            raise ValueError("OscillatorySSM needs the sampling rate (osc_sfreq).")
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
        radius = torch.exp(-self.delta(u) * damping)            # r_t in (0, 1)
        drive = (1 - radius) * u.repeat_interleave(self.n_modes, dim=1)
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


