"""Compare inference latency of Mamba scan modes against the pure TCFormer.

Usage (from the repository root):
    python scripts/benchmark_scan_modes.py --channels 14 --samples 1125
"""
import argparse
import os
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Notebook kernels export an inline backend the venv matplotlib rejects.
os.environ["MPLBACKEND"] = "Agg"

from models.tcformer import TCFormerModule  # noqa: E402
from utils.latency import measure_latency  # noqa: E402

MODULE_KEYS = {
    "F1", "temp_kernel_lengths", "pool_length_1", "pool_length_2", "D",
    "dropout_conv", "d_group", "tcn_depth", "kernel_length_tcn",
    "dropout_tcn", "use_group_attn", "trans_depth", "trans_dropout",
    "sequence_block_types", "mamba_d_state", "mamba_d_conv",
}


def build(model_kwargs, n_channels, n_classes, scan_mode=None):
    kwargs = {k: v for k, v in model_kwargs.items() if k in MODULE_KEYS}
    if scan_mode is None:
        kwargs["sequence_block_types"] = ["transformer"] * kwargs["trans_depth"]
    else:
        kwargs["mamba_scan_mode"] = scan_mode
    torch.manual_seed(0)
    return TCFormerModule(n_channels=n_channels, n_classes=n_classes, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/full_mamba_source_only.yaml")
    parser.add_argument("--channels", type=int, default=14)
    parser.add_argument("--samples", type=int, default=1125)
    parser.add_argument("--classes", type=int, default=3)
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 32])
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--cpu_runs", type=int, default=20)
    args = parser.parse_args()

    model_kwargs = yaml.safe_load(Path(args.config).read_text())["model_kwargs"]
    variants = {
        "transformer (TCFormer)": None,
        "mamba scripted": "scripted",
        "mamba parallel": "parallel",
        "mamba hoisted": "hoisted",
    }
    devices = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])

    # Numerical equivalence check of the two Mamba scans on identical weights.
    x = torch.randn(4, args.channels, args.samples)
    scripted = build(model_kwargs, args.channels, args.classes, "scripted").eval()
    parallel = build(model_kwargs, args.channels, args.classes, "parallel").eval()
    hoisted = build(model_kwargs, args.channels, args.classes, "hoisted").eval()
    parallel.load_state_dict(scripted.state_dict())
    hoisted.load_state_dict(scripted.state_dict())
    with torch.no_grad():
        reference = scripted(x)
        for name, model in (("parallel", parallel), ("hoisted", hoisted)):
            max_diff = (reference - model(x)).abs().max().item()
            print(f"max |scripted - {name}| logits = {max_diff:.2e}", flush=True)

    for device in devices:
        for batch_size in args.batch_sizes:
            for name, scan_mode in variants.items():
                model = build(model_kwargs, args.channels, args.classes, scan_mode)
                latency = measure_latency(
                    model,
                    (batch_size, args.channels, args.samples),
                    device=device,
                    warmup=args.warmup if device != "cpu" else 5,
                    runs=args.runs if device != "cpu" else args.cpu_runs,
                )
                print(
                    f"{device:7s} | batch={batch_size:3d} | {name:24s} | "
                    f"{latency:8.3f} ms/forward",
                    flush=True,
                )


if __name__ == "__main__":
    main()
