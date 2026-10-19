"""Learned AutoDIAL alpha versus source-target feature shift, per target subject.

Reads the best source-validation checkpoints of the LOSO runs (one folder per
dataset, files subject_<id>_best-*.ckpt) and, for every target subject and
AutoDIAL feature group, reports

* alpha: sigmoid(alpha_logit) for the learned weights (unchanged by IM-TTA),
  or the group mean of the running shrinkage weights (feature/hada-autodial-
  shrinkage-alpha, value at the best checkpoint, before target recalibration);
* shift: mean over the group's dimensions of |mu_t - mu_s| / sigma_s, from the
  running source and target statistics of the AutoDIAL layer (target = the
  unlabeled earlier sessions; the evaluation sessions are not involved);
* w2: 2-Wasserstein distance between the diagonal Gaussians of the two
  domains, sqrt(mean((mu_t - mu_s)^2 + (sigma_t - sigma_s)^2)) / mean(sigma_s).

Usage:
    python analyze_autodial_alpha.py --run bcic2a=path/to/2a/checkpoints \
        --run bcic2b=path/to/2b/checkpoints --run zhou2016=path/to/zhou/checkpoints \
        --out autodial_alpha.csv
"""
import argparse
import csv
import re
from pathlib import Path

import torch

GROUP_NAMES = ["CNN k=20", "CNN k=32", "CNN k=64", "SSM"]


def domain_norm_state(state_dict):
    """The AutoDIAL buffers, whatever the module prefix in the checkpoint."""
    wanted = ["source_mean", "source_var", "target_mean", "target_var"]
    found = {}
    for key, value in state_dict.items():
        if "domain_norm." in key:
            name = key.rsplit(".", 1)[-1]
            if name in wanted + ["alpha_logit", "alpha_running"]:
                found[name] = value.float()
    missing = set(wanted) - set(found)
    if "alpha_logit" not in found and "alpha_running" not in found:
        missing.add("alpha_logit or alpha_running")
    if missing:
        raise KeyError(f"AutoDIAL entries {sorted(missing)} not found in checkpoint")
    return found


def analyse_checkpoint(path, eps=1e-5):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    s = domain_norm_state(checkpoint.get("state_dict", checkpoint))
    if "alpha_logit" in s:
        alpha = torch.sigmoid(s["alpha_logit"])
    else:
        alpha = s["alpha_running"].view(4, -1).mean(dim=1)
    n_groups = alpha.numel()
    sd_s = (s["source_var"] + eps).sqrt()
    sd_t = (s["target_var"] + eps).sqrt()
    rows = []
    for g, (mu_s, mu_t, sig_s, sig_t) in enumerate(zip(
            s["source_mean"].chunk(n_groups), s["target_mean"].chunk(n_groups),
            sd_s.chunk(n_groups), sd_t.chunk(n_groups))):
        shift = ((mu_t - mu_s).abs() / sig_s).mean().item()
        w2 = (((mu_t - mu_s) ** 2 + (sig_t - sig_s) ** 2).mean().sqrt() / sig_s.mean()).item()
        rows.append(dict(group=g, group_name=GROUP_NAMES[g] if n_groups == 4 else f"group {g}",
                         alpha=alpha[g].item(), shift=shift, w2=w2))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True,
                        help="dataset=folder with subject_<id>_best-*.ckpt files")
    parser.add_argument("--out", default="autodial_alpha.csv")
    args = parser.parse_args()

    records = []
    for spec in args.run:
        dataset, folder = spec.split("=", 1)
        checkpoints = sorted(Path(folder).glob("subject_*_best*.ckpt"),
                             key=lambda p: int(re.search(r"subject_(\d+)_", p.name).group(1)))
        if not checkpoints:
            raise FileNotFoundError(f"No subject_*_best*.ckpt in {folder}")
        for path in checkpoints:
            subject = int(re.search(r"subject_(\d+)_", path.name).group(1))
            for row in analyse_checkpoint(path):
                records.append(dict(dataset=dataset, subject=subject, **row))

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    print(f"{'dataset':<9} {'subj':>4}  " + "  ".join(f"{n:>17}" for n in GROUP_NAMES))
    print(f"{'':<9} {'':>4}  " + "  ".join(f"{'alpha / shift':>17}" for _ in GROUP_NAMES))
    keys = sorted({(r["dataset"], r["subject"]) for r in records},
                  key=lambda k: ([s.split('=')[0] for s in args.run].index(k[0]), k[1]))
    for dataset, subject in keys:
        rows = [r for r in records if r["dataset"] == dataset and r["subject"] == subject]
        cells = "  ".join(f"{r['alpha']:>8.3f} / {r['shift']:<6.3f}" for r in rows)
        print(f"{dataset:<9} {subject:>4}  {cells}")
    print(f"\nSaved {len(records)} rows to {args.out}")


if __name__ == "__main__":
    main()
