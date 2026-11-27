"""Check the parallel (segment-sum) selective scan against the hoisted and step-by-step scans.

    python -m ssm_dan.scan_check [--dataset bcic2a] [--samples 1000]

1. Recurrence alone, on random inputs: outputs and gradients of the three scans
   (float64 and float32, several sequence lengths and chunk sizes).
2. Whole model: the source-only SSM-DAN network of ``--dataset`` built with ``ssm_scan=hoisted``,
   the same weights loaded into ``ssm_scan=segsum``; logits compared, then single-trial CPU latency of
   both, measured as in the training pipeline (ssm_dan.utils.latency.measure_latency).
Exits with status 1 if a difference exceeds its tolerance.
"""
import argparse
import sys

import torch

from . import config as cfg
from .data import get_datamodule_cls
from .models import get_model_cls
from .models.backbone import _hoisted_selective_scan, _segsum_selective_scan
from .utils.latency import measure_latency

TOLERANCE = {torch.float64: 1e-9, torch.float32: 1e-4}


def _sequential(values, delta, A, B, C, D):
    state = values.new_zeros(values.size(0), values.size(2), A.size(1))
    outputs = []
    for step in range(values.size(1)):
        dt = delta[:, step].unsqueeze(-1)
        state = torch.exp(dt * A) * state + dt * B[:, step].unsqueeze(1) * values[:, step].unsqueeze(-1)
        outputs.append((state * C[:, step].unsqueeze(1)).sum(-1) + D * values[:, step])
    return torch.stack(outputs, dim=1)


def _inputs(batch, length, d_model, d_state, dtype, generator):
    def rand(*shape):
        return torch.randn(*shape, dtype=dtype, generator=generator)
    values, B, C = rand(batch, length, d_model), rand(batch, length, d_state), rand(batch, length, d_state)
    delta = torch.nn.functional.softplus(rand(batch, length, d_model)).clamp(max=1.0)  # as in the mixer
    A = -torch.exp(torch.log(torch.arange(1, d_state + 1, dtype=dtype)).unsqueeze(0).repeat(d_model, 1))
    D = rand(d_model)
    return [t.requires_grad_(True) for t in (values, delta, A, B, C, D)]


def _gradients(output, inputs, weight):
    return torch.autograd.grad((output * weight).sum(), inputs)


def _relative(a, b):
    """max |a - b| relative to the size of the reference b."""
    return (a - b).abs().max().item() / max(b.abs().max().item(), 1e-30)


def check_recurrence():
    ok = True
    generator = torch.Generator().manual_seed(0)
    print("1. Recurrence: max relative difference to the step-by-step scan (outputs, then gradients)")
    print(f"   {'dtype':8s} {'L':>4s} {'chunk':>5s} {'hoisted out':>12s} {'segsum out':>12s} "
          f"{'hoisted grad':>12s} {'segsum grad':>12s}")
    for dtype in (torch.float64, torch.float32):
        for length, chunk in ((1, 32), (17, 32), (17, 5), (17, 1), (40, 16), (100, 32)):
            inputs = _inputs(3, length, 48, 8, dtype, generator)
            reference = _sequential(*inputs)
            weight = torch.randn(reference.shape, dtype=dtype, generator=generator)
            ref_grad = _gradients(reference, inputs, weight)
            row = []
            for out in (_hoisted_selective_scan(*inputs), _segsum_selective_scan(*inputs, chunk_size=chunk)):
                row.append((_relative(out, reference),
                            max(_relative(g, r) for g, r in zip(_gradients(out, inputs, weight), ref_grad))))
            errors = [row[0][0], row[1][0], row[0][1], row[1][1]]
            passed = all(e <= TOLERANCE[dtype] for e in errors)
            ok &= passed
            print(f"   {str(dtype).split('.')[-1]:8s} {length:4d} {chunk:5d} " + " ".join(f"{e:12.2e}" for e in errors)
                  + ("" if passed else "   <-- above tolerance"))
    return ok


def check_model(dataset, samples):
    config = cfg.resolve("ssm_dan", dataset, ["source_only"], seed=0)
    datamodule_cls = get_datamodule_cls(dataset)
    kwargs = cfg.model_kwargs(config)
    kwargs.update(n_channels=datamodule_cls.channels, n_classes=datamodule_cls.classes)
    model_cls = get_model_cls(config["model"])
    torch.manual_seed(0)
    hoisted = model_cls(**{**kwargs, "ssm_scan": "hoisted"}).eval()
    segsum = model_cls(**{**kwargs, "ssm_scan": "segsum"}).eval()
    segsum.load_state_dict(hoisted.state_dict())

    print(f"\n2. Whole source-only model, {dataset}: input {datamodule_cls.channels} x {samples}")
    ok = True
    with torch.no_grad():
        for batch in (1, 48):
            x = torch.randn(batch, datamodule_cls.channels, samples)
            a, b = hoisted(x), segsum(x)
            difference = _relative(b, a)
            same_class = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
            passed = difference <= TOLERANCE[torch.float32]
            ok &= passed
            print(f"   batch {batch:2d}: max relative logit difference {difference:.2e}, same predicted class "
                  f"{same_class * 100:.1f}%" + ("" if passed else "   <-- above tolerance"))

    print(f"\n3. Single-trial CPU latency (torch threads: {torch.get_num_threads()})")
    shape = (1, datamodule_cls.channels, samples)
    for name, model in (("hoisted", hoisted), ("segsum", segsum), ("hoisted", hoisted), ("segsum", segsum)):
        print(f"   {name:8s} {measure_latency(model, shape, device='cpu'):7.2f} ms")
    return ok


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="bcic2a")
    parser.add_argument("--samples", type=int, default=1000, help="time samples per trial (BCI IV-2a: 4 s x 250 Hz)")
    args = parser.parse_args(argv)
    ok = check_recurrence()
    ok &= check_model(args.dataset, args.samples)
    print("\nAll differences within tolerance." if ok else "\nSOME DIFFERENCES ABOVE TOLERANCE.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
