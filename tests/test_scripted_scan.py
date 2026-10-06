import copy
import time

import torch

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
