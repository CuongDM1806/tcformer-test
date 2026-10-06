"""Benchmark the source-only selective scan implementations on one CUDA GPU."""

import argparse
import copy
import time

import torch

from models.tcformer import _SelectiveSSMMixer, selective_scan_fn


def synchronize():
    torch.cuda.synchronize()


def time_forward_backward(model, sample, warmup, runs):
    model.train()
    for _ in range(warmup):
        model.zero_grad(set_to_none=True)
        sample.grad = None
        output = model(sample)
        output.square().mean().backward()
    synchronize()

    started_at = time.perf_counter()
    for _ in range(runs):
        model.zero_grad(set_to_none=True)
        sample.grad = None
        output = model(sample)
        output.square().mean().backward()
    synchronize()
    return 1000.0 * (time.perf_counter() - started_at) / runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--length", type=int, default=18)
    parser.add_argument("--d-model", type=int, default=48)
    parser.add_argument("--d-state", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=50)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA GPU.")
    if selective_scan_fn is None:
        raise RuntimeError("mamba-ssm selective_scan_fn is unavailable.")

    torch.manual_seed(7)
    device = torch.device("cuda")
    scripted = _SelectiveSSMMixer(
        args.d_model, args.d_state, d_conv=3, scan_mode="scripted"
    ).to(device)
    fused = copy.deepcopy(scripted)
    fused.scan_mode = "fused"
    compiled = copy.deepcopy(fused)
    compiled.compile(mode="reduce-overhead", dynamic=False)

    sample = torch.randn(
        args.batch_size,
        args.length,
        args.d_model,
        device=device,
        requires_grad=True,
    )
    with torch.no_grad():
        expected = scripted(sample)
        actual = fused(sample)
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)

    timings = {
        "scripted_ms": time_forward_backward(
            scripted, sample, args.warmup, args.runs
        ),
        "fused_ms": time_forward_backward(fused, sample, args.warmup, args.runs),
        "fused_compile_ms": time_forward_backward(
            compiled, sample, args.warmup, args.runs
        ),
    }
    timings["fused_speedup"] = timings["scripted_ms"] / timings["fused_ms"]
    timings["compiled_speedup"] = (
        timings["scripted_ms"] / timings["fused_compile_ms"]
    )
    print({key: round(value, 4) for key, value in timings.items()})


if __name__ == "__main__":
    main()
