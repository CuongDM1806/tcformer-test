import copy
import time

import torch

import models.tcformer as tcformer_module
from models.tcformer import _SelectiveSSMMixer


def _make_pair(d_model=48, d_state=8, d_conv=3):
    torch.manual_seed(7)
    reference = _SelectiveSSMMixer(
        d_model=d_model,
        d_state=d_state,
        d_conv=d_conv,
        scan_mode="sequential",
    )
    scripted = copy.deepcopy(reference)
    scripted.scan_mode = "scripted"
    return reference, scripted


def test_scripted_scan_matches_reference_forward_and_backward():
    reference, scripted = _make_pair()
    x_reference = torch.randn(4, 18, 48, requires_grad=True)
    x_scripted = x_reference.detach().clone().requires_grad_(True)

    y_reference = reference(x_reference)
    y_scripted = scripted(x_scripted)
    torch.testing.assert_close(y_scripted, y_reference, rtol=1e-6, atol=1e-7)

    loss_reference = y_reference.square().mean()
    loss_scripted = y_scripted.square().mean()
    loss_reference.backward()
    loss_scripted.backward()
    torch.testing.assert_close(
        x_scripted.grad, x_reference.grad, rtol=5e-5, atol=5e-7
    )
    for (_, parameter_reference), (_, parameter_scripted) in zip(
        reference.named_parameters(), scripted.named_parameters()
    ):
        torch.testing.assert_close(
            parameter_scripted.grad,
            parameter_reference.grad,
            rtol=5e-5,
            atol=5e-7,
        )


def test_scripted_scan_cpu_benchmark_smoke():
    reference, scripted = _make_pair()
    x = torch.randn(1, 18, 48)
    for model in (reference, scripted):
        model.eval()
        with torch.inference_mode():
            for _ in range(10):
                model(x)

    timings = {}
    with torch.inference_mode():
        for name, model in (("sequential", reference), ("scripted", scripted)):
            start = time.perf_counter()
            for _ in range(100):
                model(x)
            timings[name] = time.perf_counter() - start

    # A generous guard against accidental regressions; exact speed depends on CPU.
    speedup = timings["sequential"] / timings["scripted"]
    print({**timings, "speedup": speedup})
    assert timings["scripted"] < timings["sequential"] * 1.25, timings


def _fake_selective_scan(u, delta, A, B, C, D, z, **_):
    """CPU reference with the public selective_scan_fn tensor contract."""
    state = u.new_zeros(u.size(0), u.size(1), A.size(1))
    outputs = []
    for step in range(u.size(2)):
        dt = delta[:, :, step].unsqueeze(-1)
        state = (
            torch.exp(dt * A.unsqueeze(0)) * state
            + dt * B[:, :, step].unsqueeze(1) * u[:, :, step].unsqueeze(-1)
        )
        readout = (state * C[:, :, step].unsqueeze(1)).sum(dim=-1)
        outputs.append(readout + D * u[:, :, step])
    output = torch.stack(outputs, dim=2)
    return output * torch.nn.functional.silu(z)


def test_fused_layout_matches_scripted_reference(monkeypatch):
    reference, fused = _make_pair(d_model=12, d_state=4)
    monkeypatch.setattr(tcformer_module, "selective_scan_fn", _fake_selective_scan)
    x = torch.randn(2, 9, 12)

    values, gate = fused.in_proj(x).chunk(2, dim=-1)
    values = fused.local_conv(values.transpose(1, 2))[..., : x.size(1)]
    values = torch.nn.functional.silu(values.transpose(1, 2))
    delta_raw, B, C = torch.split(
        fused.x_proj(values), [fused.d_model, fused.d_state, fused.d_state], dim=-1
    )
    delta = torch.nn.functional.softplus(delta_raw).clamp(max=1.0)
    A = -torch.exp(fused.A_log).to(dtype=x.dtype)

    fused_scan = fused._fused_scan(values, delta, A, B, C, gate)
    scripted_scan = tcformer_module._scripted_selective_scan(
        values, delta, A, B, C, fused.D
    ) * torch.nn.functional.silu(gate)
    torch.testing.assert_close(fused_scan, scripted_scan, rtol=1e-6, atol=1e-7)


def test_fused_mode_uses_scripted_fallback_on_cpu():
    reference, fused = _make_pair(d_model=12, d_state=4)
    fused.scan_mode = "fused"
    x = torch.randn(2, 9, 12)
    torch.testing.assert_close(fused(x), reference(x), rtol=1e-6, atol=1e-7)
