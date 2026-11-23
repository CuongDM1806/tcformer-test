"""Phase-synchrony readout of OscillatorySSM (spec items 1-8).

Run from the repository root:  python -m pytest -q tests/test_osc_sync.py
The learning sanity check trains small models on synthetic data (about a minute on a GPU).
"""
import math

import numpy as np
import pytest
import torch
from torch import nn

from ssm_dan.models import backbone as bb
from ssm_dan.models.backbone import OscillatorySSM, SSMBackbone, SyncProbes, channel_sides

from reference_osc_5430ead1 import ReferenceOscillatorySSM

FS = 250.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _nearest_mode(osc, hz):
    return int(torch.argmin((osc.frequencies()[0] - hz).abs()))


# ---------------------------------------------------------------- 1. backward compatibility
def test_none_matches_reference_branch():
    torch.manual_seed(0)
    old = ReferenceOscillatorySSM(22, 16, FS)
    torch.manual_seed(0)
    new = OscillatorySSM(22, 16, FS, sync_mode="none")
    assert sum(p.numel() for p in old.parameters()) == sum(p.numel() for p in new.parameters())
    old_state, new_state = old.state_dict(), new.state_dict()
    assert list(old_state) == list(new_state)
    for key in old_state:
        assert old_state[key].shape == new_state[key].shape, key
        assert torch.equal(old_state[key], new_state[key]), key
    new.load_state_dict(old_state, strict=True)
    x = torch.randn(4, 22, 448)
    old.eval(), new.eval()
    assert torch.allclose(old(x), new(x), atol=0, rtol=0)
    assert set(old.interpretation()) <= set(new.interpretation())


@pytest.mark.parametrize("branch", ["feed", "replace"])
def test_none_matches_reference_backbone(branch, monkeypatch):
    kwargs = dict(n_channels=22, n_classes=4, osc_branch=branch, osc_sfreq=FS)
    torch.manual_seed(0)
    new = SSMBackbone(**kwargs)

    def reference(*args, sync_mode="none", n_probes=3, tie_freq=None, channel_side=None, **kw):
        return ReferenceOscillatorySSM(*args, **kw)

    monkeypatch.setattr(bb, "OscillatorySSM", reference)
    torch.manual_seed(0)
    old = SSMBackbone(**kwargs)
    assert sum(p.numel() for p in old.parameters()) == sum(p.numel() for p in new.parameters())
    new.load_state_dict(old.state_dict(), strict=True)
    x = torch.randn(3, 22, 448)
    old.eval(), new.eval()
    assert torch.allclose(old(x), new(x), atol=0, rtol=0)


def test_paper_backbone_unchanged():
    torch.manual_seed(0)
    model = SSMBackbone(n_channels=22, n_classes=4)
    assert not hasattr(model, "osc")
    assert model.interpretation() is None


# ---------------------------------------------------------------- 2. tied frequencies
def test_tie_freq():
    auto = OscillatorySSM(8, 16, FS, n_virtual=6, n_modes=5, sync_mode="concat")
    assert auto.tie_freq and auto.freq_logit.shape == (1, 5)
    assert auto.frequencies().shape == (6, 5) and auto.bandwidths().shape == (6, 5)
    assert torch.allclose(auto.frequencies()[0], auto.frequencies()[-1])
    untied = OscillatorySSM(8, 16, FS, n_virtual=6, n_modes=5, sync_mode="concat", tie_freq=False)
    assert untied.freq_logit.shape == (6, 5)
    assert not OscillatorySSM(8, 16, FS, sync_mode="none").tie_freq


# ---------------------------------------------------------------- shapes, gradients, zero input
@pytest.mark.parametrize("sync", ["none", "concat", "only"])
@pytest.mark.parametrize("branch", ["none", "feed", "replace"])
@pytest.mark.parametrize("zero", [False, True])
def test_forward_backward(branch, sync, zero):
    torch.manual_seed(0)
    model = SSMBackbone(n_channels=4, n_classes=3, osc_branch=branch, osc_sfreq=FS, osc_sync=sync,
                        osc_channel_side=[-1, 0, 1, 1]).to(DEVICE)
    x = torch.zeros(3, 4, 448, device=DEVICE) if zero else torch.randn(3, 4, 448, device=DEVICE)
    out = model(x)
    assert out.shape == (3, 3) and torch.isfinite(out).all()
    out.square().sum().backward()
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), name
    if branch != "none":
        assert model.osc(x).shape == (3, 16, 448 // 56)
        info = model.interpretation()
        assert ("readout_weight" in info) == (sync != "only")
        assert ("sync_readout_weight" in info) == (sync != "none")


def test_feature_counts():
    for sync, n in (("none", 16 * 8), ("concat", 16 * 8 + 3 * 8), ("only", 3 * 8)):
        osc = OscillatorySSM(10, 16, FS, sync_mode=sync)
        assert osc.power_norm.num_features == n and osc.readout[0].in_channels == n
        assert hasattr(osc, "sync") == (sync != "none")


# ---------------------------------------------------------------- 3. range of R
def test_sync_range():
    torch.manual_seed(0)
    probes = SyncProbes(16, 8, 3)
    with torch.no_grad():
        probes.phase.uniform_(-math.pi, math.pi)
    h_re, h_im = torch.randn(4, 16, 8, 300) * 3, torch.randn(4, 16, 8, 300) * 3
    R = probes(h_re, h_im)
    assert R.shape == (4, 24, 300)
    assert R.min() >= 0 and R.max() <= 1 + 1e-5


def _identity_osc(n, sync="only", **kw):
    osc = OscillatorySSM(n, 16, FS, n_virtual=n, n_modes=8, sync_mode=sync, **kw)
    with torch.no_grad():
        osc.spatial.weight.copy_(torch.eye(n)[:, :, None])
    return osc


def _synchrony(osc, x):
    with torch.no_grad():
        h_re, h_im = osc.band_state(x)
        return osc.synchrony(h_re, h_im)                       # [B, J*K, T]


# ---------------------------------------------------------------- 4. perfect synchrony
def test_perfect_synchrony():
    torch.manual_seed(0)
    osc = _identity_osc(16)
    t = torch.arange(int(2 * FS)) / FS
    x = torch.sin(2 * math.pi * 10 * t).expand(1, 16, -1).contiguous()
    R = _synchrony(osc, x).reshape(1, 3, 8, -1)
    k = _nearest_mode(osc, 10.0)
    assert R[:, :, k, int(0.5 * FS):].min() > 0.95


# ---------------------------------------------------------------- 5. fixed phase lag invariance
def test_sign_flip_phase_compensation():
    torch.manual_seed(0)
    osc = OscillatorySSM(12, 16, FS, n_virtual=16, n_modes=8, sync_mode="only")
    x = torch.randn(2, 12, 500)
    before = _synchrony(osc, x)
    flipped = torch.arange(16) < 8
    with torch.no_grad():
        osc.spatial.weight[flipped] *= -1
        osc.sync.phase[:, flipped, :] = math.pi
    after = _synchrony(osc, x)
    assert torch.allclose(before, after, atol=1e-4)


# ---------------------------------------------------------------- 6. desynchronization
def test_desynchrony():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    osc = _identity_osc(16)
    n = int(2 * FS)
    t = np.arange(n) / FS
    phase = rng.uniform(0, 2 * np.pi, (4, 16, 1)) + np.cumsum(rng.normal(0, 0.05, (4, 16, n)), axis=-1)
    x = torch.tensor(np.sin(2 * np.pi * 10 * t + phase), dtype=torch.float32)
    R = _synchrony(osc, x).reshape(4, 3, 8, -1)
    k = _nearest_mode(osc, 10.0)
    assert R[:, :, k, int(0.5 * FS):].mean() < 0.5


# ---------------------------------------------------------------- montage sides
def test_channel_sides():
    assert channel_sides(["C3", "Cz", "C4"]) == [-1, 0, 1]
    assert channel_sides(["FP1", "fcz", "O2"]) == [-1, 0, 1]


# ---------------------------------------------------------------- 7. learning sanity check
def _pink(rng, shape):
    spectrum = np.fft.rfft(rng.normal(size=shape), axis=-1)
    f = np.arange(spectrum.shape[-1])
    spectrum[..., 1:] /= np.sqrt(f[1:])
    spectrum[..., 0] = 0
    noise = np.fft.irfft(spectrum, n=shape[-1], axis=-1)
    return noise / noise.std(axis=-1, keepdims=True)


def synthetic_sync_task(n, seed, seconds=2.0):
    """Two classes, 8 channels (0-3 left, 4-7 right), 10 Hz in phase within each side plus pink
    noise. From 1 s on (0.5-s ramp) every channel takes a phase offset uniform in (-pi, pi]:
    one offset shared by its side, or, on the desynchronized side (left for class 0, right for
    class 1), its own. Each channel alone is distributed identically in both classes; only the
    phase relation between channels differs."""
    rng = np.random.default_rng(seed)
    length = int(seconds * FS)
    t = np.arange(length) / FS
    ramp = np.clip((np.arange(length) - int(1.0 * FS)) / int(0.5 * FS), 0, 1)
    y = rng.integers(0, 2, n)
    X = np.zeros((n, 8, length))
    for i in range(n):
        for side, channels in ((0, range(0, 4)), (1, range(4, 8))):
            base, common = rng.uniform(0, 2 * np.pi), rng.uniform(-np.pi, np.pi)
            for c in channels:
                offset = rng.uniform(-np.pi, np.pi) if y[i] == side else common
                X[i, c] = rng.uniform(0.8, 1.2) * np.sin(2 * np.pi * 10 * t + base + offset * ramp)
        X[i] += 0.5 * _pink(rng, (8, length))
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y)


class _Probe(nn.Module):
    def __init__(self, sync, learn_spatial):
        super().__init__()
        self.osc = OscillatorySSM(8, 16, FS, n_virtual=8, n_modes=8, sync_mode=sync, dropout=0.0,
                                  channel_side=[-1] * 4 + [1] * 4)
        if not learn_spatial:
            with torch.no_grad():
                self.osc.spatial.weight.copy_(torch.eye(8)[:, :, None])
            self.osc.spatial.weight.requires_grad_(False)
        self.head = nn.Linear(16, 2)

    def forward(self, x):
        return self.head(self.osc(x).mean(dim=-1))


def train_probe(sync, learn_spatial, epochs=40, seed=0):
    torch.manual_seed(seed)
    X, y = synthetic_sync_task(400, seed=1)
    X_test, y_test = synthetic_sync_task(200, seed=2)
    model = _Probe(sync, learn_spatial).to(DEVICE)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=3e-3)
    for _ in range(epochs):
        model.train()
        for idx in torch.randperm(len(X)).split(32):
            loss = nn.functional.cross_entropy(model(X[idx].to(DEVICE)), y[idx].to(DEVICE))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        return (model(X_test.to(DEVICE)).argmax(1).cpu() == y_test).float().mean().item()


@pytest.mark.slow
def test_learning_sanity_fixed_electrodes():
    """Virtual channel = electrode (identity spatial filters, frozen): power is blind to the
    phase relation, synchrony is not."""
    acc_only = train_probe("only", learn_spatial=False)
    acc_none = train_probe("none", learn_spatial=False)
    print(f"\nfixed electrodes: sync only {acc_only:.3f} | power only {acc_none:.3f}")
    assert acc_only > 0.95
    assert acc_none < 0.65


@pytest.mark.slow
def test_learning_sanity_learned_spatial_filters():
    """Learned spatial filters (spec setting). Synchrony must solve the task. Power is printed,
    not asserted: a filter that sums in-phase electrodes has more power than one that sums
    desynchronized ones, so learned filters followed by power can detect the change too
    (as CSP does through the covariance)."""
    acc_only = train_probe("only", learn_spatial=True)
    acc_none = train_probe("none", learn_spatial=True)
    print(f"\nlearned spatial filters: sync only {acc_only:.3f} | power only {acc_none:.3f}")
    assert acc_only > 0.95
