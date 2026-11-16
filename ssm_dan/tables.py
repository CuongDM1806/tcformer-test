"""Paper-style tables from results/: per-subject accuracy, mean +- std, kappa, ITR and paired tests.

    python -m ssm_dan.tables                         # every dataset under results/
    python -m ssm_dan.tables --dataset bcic2a --reference ssm_dan
    python -m ssm_dan.tables --metric no_tta_acc     # e.g. the same weights without IM-TTA

For every dataset, each run (method + variants, seed 0 by default) is one row.
``p`` is the two-sided Wilcoxon signed-rank test of the per-subject accuracy
against the reference run (exact up to 25 subjects, normal approximation
otherwise); ``p_holm`` applies Holm's correction over the compared runs. The
paper forms one family per target-data access, e.g.

    python -m ssm_dan.tables --dataset bcic2a --reference ssm_dan-source_only \
        --runs eegconformer,atcnet,tcformer,eegencoder,satransnet,ssm_dan-source_only Standard deviations are population SDs across subjects, as in the
paper. Tables are printed and written to <results>/tables/<dataset>.csv.
"""
import argparse
import csv
import math
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from . import results

N_CLASSES = {"bcic2a": 4, "bcic2b": 2, "zhou2016": 3, "openbmi": 2}
WINDOW_SECONDS = {"bcic2a": 4.0, "bcic2b": 4.0, "zhou2016": 4.5, "openbmi": 4.0}


def itr(accuracy, n_classes, seconds):
    """Wolpaw information transfer rate (bits/min); zero at or below chance."""
    p, n = accuracy, n_classes
    if p <= 1.0 / n:
        return 0.0
    bits = math.log2(n) + p * math.log2(p)
    if p < 1.0:
        bits += (1 - p) * math.log2((1 - p) / (n - 1))
    return bits * 60.0 / seconds


def holm(p_values):
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def table(root, dataset, reference, metric, seed, only=None):
    runs = {}
    for directory in sorted((root / dataset).glob(f"*/seed{seed}")):
        if only and directory.parent.name not in only:
            continue
        records = results.load_subjects(directory)
        if records and all(metric in r for r in records):
            runs[directory.parent.name] = {r["subject_id"]: r for r in records}
    if not runs:
        return None
    subjects = sorted(set.intersection(*(set(r) for r in runs.values())))
    rows = []
    for name, by_subject in runs.items():
        acc = np.array([by_subject[s][metric] for s in subjects])
        kappa_key = metric.replace("acc", "kappa")
        kappa = np.array([by_subject[s].get(kappa_key, np.nan) for s in subjects])
        rates = np.array([itr(a, N_CLASSES[dataset], WINDOW_SECONDS[dataset]) for a in acc])
        rows.append({"run": name, "acc": acc, "kappa": kappa, "itr": rates})

    ref = next((row for row in rows if row["run"] == reference), None)
    tested = [row for row in rows if ref is not None and row is not ref]
    for row in tested:
        diff = row["acc"] - ref["acc"]
        if np.allclose(diff, 0):
            row["p"] = 1.0
        else:
            method = "exact" if len(subjects) <= 25 else "approx"
            row["p"] = float(wilcoxon(row["acc"], ref["acc"], method=method).pvalue)
    if tested:
        for row, adjusted in zip(tested, holm(np.array([row["p"] for row in tested]))):
            row["p_holm"] = float(adjusted)
    return subjects, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=None, help="results root (default: ./results)")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--reference", default="ssm_dan", help="run compared against (default: ssm_dan)")
    parser.add_argument("--metric", default="test_acc", help="test_acc | no_tta_acc | all_sessions_acc")
    parser.add_argument("--runs", default=None, help="comma-separated runs to include (default: all)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    only = set(args.runs.split(",")) | {args.reference} if args.runs else None

    root = Path(args.results) if args.results else results.DEFAULT_RESULTS
    datasets = [args.dataset] if args.dataset else sorted(p.name for p in root.iterdir()
                                                          if p.is_dir() and p.name in N_CLASSES)
    (root / "tables").mkdir(parents=True, exist_ok=True)
    for dataset in datasets:
        built = table(root, dataset, args.reference, args.metric, args.seed, only)
        if built is None:
            continue
        subjects, rows = built
        header = (["run"] + [f"S{s}" for s in subjects]
                  + ["acc_mean", "acc_std", "kappa_mean", "kappa_std", "itr_mean", "itr_std", "p", "p_holm"])
        lines = []
        for row in rows:
            lines.append([row["run"]] + [f"{a * 100:.2f}" for a in row["acc"]] + [
                f"{row['acc'].mean() * 100:.2f}", f"{row['acc'].std() * 100:.2f}",
                f"{np.nanmean(row['kappa']):.3f}", f"{np.nanstd(row['kappa']):.3f}",
                f"{row['itr'].mean():.2f}", f"{row['itr'].std():.2f}",
                f"{row['p']:.4f}" if "p" in row else "--", f"{row['p_holm']:.4f}" if "p_holm" in row else "--"])
        with open(root / "tables" / f"{dataset}.csv", "w", newline="") as f:
            csv.writer(f).writerows([header] + lines)
        print(f"\n== {dataset} | {args.metric} | {len(subjects)} subjects | reference: {args.reference}")
        print(f"{'run':32s} {'acc (%)':>15s} {'kappa':>14s} {'ITR':>12s} {'p':>8s} {'p_holm':>8s}")
        for row, line in zip(rows, lines):
            print(f"{row['run']:32s} {line[-8]:>7s} +- {line[-7]:>5s} {line[-6]:>6s} +- {line[-5]:>5s} "
                  f"{line[-4]:>5s} +- {line[-3]:>4s} {line[-2]:>8s} {line[-1]:>8s}")


if __name__ == "__main__":
    main()
