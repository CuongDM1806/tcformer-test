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
    parallel = copy.deepcopy(reference)
    parallel.scan_mode = "parallel"
    return reference, parallel


def test_parallel_scan_matches_reference_forward_and_backward():
    reference, parallel = _make_pair()
    # Make the learned decays large so long-range decay products are exercised.
    with torch.no_grad():
        for model in (reference, parallel):
            model.A_log.add_(1.5)
    x_reference = torch.randn(4, 20, 48, requires_grad=True)
    x_parallel = x_reference.detach().clone().requires_grad_(True)

    y_reference = reference(x_reference)
    y_parallel = parallel(x_parallel)
    assert torch.isfinite(y_parallel).all()
    torch.testing.assert_close(y_parallel, y_reference, rtol=1e-5, atol=1e-6)

    y_reference.square().mean().backward()
    y_parallel.square().mean().backward()
    assert torch.isfinite(x_parallel.grad).all()
    torch.testing.assert_close(
        x_parallel.grad, x_reference.grad, rtol=1e-4, atol=1e-6
    )
    for (name, parameter_reference), (_, parameter_parallel) in zip(
        reference.named_parameters(), parallel.named_parameters()
    ):
        assert torch.isfinite(parameter_parallel.grad).all(), name
        torch.testing.assert_close(
            parameter_parallel.grad,
            parameter_reference.grad,
            rtol=1e-4,
            atol=1e-6,
            msg=name,
        )


def test_parallel_scan_handles_single_step_sequence():
    reference, parallel = _make_pair()
    x = torch.randn(2, 1, 48)
    torch.testing.assert_close(parallel(x), reference(x), rtol=1e-6, atol=1e-7)


def test_parallel_scan_cpu_benchmark_smoke():
    reference, parallel = _make_pair()
    x = torch.randn(1, 20, 48)
    for model in (reference, parallel):
        model.eval()
        with torch.inference_mode():
            for _ in range(10):
                model(x)

    timings = {}
    with torch.inference_mode():
        for name, model in (("sequential", reference), ("parallel", parallel)):
            start = time.perf_counter()
            for _ in range(100):
                model(x)
            timings[name] = time.perf_counter() - start

    print({**timings, "speedup": timings["sequential"] / timings["parallel"]})
    assert timings["parallel"] < timings["sequential"], timings
