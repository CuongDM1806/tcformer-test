"""Chunked closed-form scan of OscillatorySSM (osc_scan="chunked") against the Hillis-Steele scan.

Run from the repository root:  python -m pytest -q tests/test_osc_scan.py
"""
import math
import time

import pytest
import torch

from ssm_dan.models.backbone import OscillatorySSM, SSMBackbone

FS = 250.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _pair(n_channels=22, **kw):
    torch.manual_seed(0)
    parallel = OscillatorySSM(n_channels, 16, FS, scan="parallel", **kw)
    with torch.no_grad():                                       # input-dependent steps, not Delta = 1
        parallel.delta_proj.weight.normal_(0, 0.5)
        parallel.freq_logit.add_(0.3 * torch.randn_like(parallel.freq_logit))
        parallel.bandwidth_logit.add_(0.3 * torch.randn_like(parallel.bandwidth_logit))
    chunked = OscillatorySSM(n_channels, 16, FS, scan="chunked", **kw)
    chunked.load_state_dict(parallel.state_dict(), strict=True)
    return parallel.to(DEVICE), chunked.to(DEVICE)


@pytest.mark.parametrize("length", [7, 16, 448, 1000, 1125])
def test_band_power_matches(length):
    parallel, chunked = _pair()
    x = torch.randn(3, 22, length, device=DEVICE)
    with torch.no_grad():
        a, b = parallel.band_power(x), chunked.band_power(x)
    assert torch.allclose(a, b, rtol=1e-3, atol=1e-5), (a - b).abs().max()


def test_forward_and_gradients_match():
    parallel, chunked = _pair()
    x = torch.randn(4, 22, 1000, device=DEVICE)
    parallel.eval(), chunked.eval()
    out_p, out_c = parallel(x), chunked(x)
    assert torch.allclose(out_p, out_c, rtol=1e-3, atol=1e-4), (out_p - out_c).abs().max()
    out_p.square().sum().backward()
    out_c.square().sum().backward()
    for (name, p), (_, c) in zip(parallel.named_parameters(), chunked.named_parameters()):
        if p.grad is None:
            continue
        scale = p.grad.abs().max().clamp(min=1e-6)
        assert ((p.grad - c.grad).abs().max() / scale) < 1e-2, name


def test_extreme_decay_is_finite():
    _, chunked = _pair()
    with torch.no_grad():                                       # widest bands, largest step (Delta = 4)
        chunked.bandwidth_logit.fill_(20.0)
        chunked.delta_proj.weight.zero_()
        chunked.delta_proj.bias.fill_(50.0)
    x = torch.randn(2, 22, 1000, device=DEVICE, requires_grad=True)
    out = chunked(x)
    out.sum().backward()
    assert torch.isfinite(out).all() and torch.isfinite(x.grad).all()


@pytest.mark.parametrize("branch", ["feed", "replace"])
def test_backbone_chunked(branch):
    torch.manual_seed(0)
    model = SSMBackbone(n_channels=22, n_classes=4, osc_branch=branch, osc_sfreq=FS, osc_scan="chunked").to(DEVICE)
    reference = SSMBackbone(n_channels=22, n_classes=4, osc_branch=branch, osc_sfreq=FS).to(DEVICE)
    reference.load_state_dict(model.state_dict(), strict=True)  # same parameters: checkpoints interchange
    x = torch.randn(3, 22, 1000, device=DEVICE)
    out = model(x)
    out.sum().backward()
    assert out.shape == (3, 4) and torch.isfinite(out).all()


def test_timing_report():
    """Not a pass/fail check: prints forward + backward time of the two scans."""
    parallel, chunked = _pair()
    x = torch.randn(96, 22, 1000, device=DEVICE)
    for name, model in (("parallel", parallel), ("chunked", chunked)):
        model.train()
        for _ in range(2):
            model(x).sum().backward()
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(5):
            model(x).sum().backward()
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        print(f"\n{name}: {(time.perf_counter() - start) / 5 * 1000:.1f} ms per forward+backward (batch 96, T=1000, {DEVICE})")
