"""One output layout for every method, dataset and variant.

    results/<dataset>/<run name>/seed<seed>/
        config.yaml                  resolved configuration
        subjects/S01.json            one record per target subject (fold)
        checkpoints/S01.ckpt         best source-validation checkpoint (Lightning methods)
        figures/S01_confusion.png    per-fold confusion matrix and learning curves
        summary.csv                  one row per subject + mean and std rows
        summary.txt                  readable summary

``<run name>`` is the method, followed by its variants (e.g. ``ssm_dan-no_ra``).
Folds of one run may be trained by several processes (e.g. one per GPU); each
writes its own subject file and the summary is rebuilt from all of them.
"""
import csv
import json
import os
from pathlib import Path

import numpy as np
import yaml

from .config import ROOT

DEFAULT_RESULTS = ROOT / "results"
METRICS = ("test_acc", "test_kappa", "no_tta_acc", "no_tta_kappa", "all_sessions_acc",
           "all_sessions_kappa", "best_val_acc", "train_time_min", "cpu_latency_ms")


def run_dir(config, results_root=None) -> Path:
    root = Path(results_root or os.environ.get("SSM_DAN_RESULTS", DEFAULT_RESULTS))
    return root / config["dataset"] / config["name"] / f"seed{config['seed']}"


def prepare(directory: Path, config) -> Path:
    for sub in ("subjects", "checkpoints", "figures"):
        (directory / sub).mkdir(parents=True, exist_ok=True)
    _atomic_write(directory / "config.yaml",
                  yaml.safe_dump({k: v for k, v in config.items() if k != "subjects"}, sort_keys=False))
    return directory


def subject_file(directory: Path, subject_id: int) -> Path:
    return directory / "subjects" / f"S{subject_id:02d}.json"


def is_done(directory: Path, subject_id: int) -> bool:
    return subject_file(directory, subject_id).is_file()


def _atomic_write(path: Path, text: str):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_subject(directory: Path, record: dict):
    _atomic_write(subject_file(directory, record["subject_id"]), json.dumps(record, indent=2))


def load_subjects(directory: Path):
    records = []
    for path in sorted((Path(directory) / "subjects").glob("S*.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    return sorted(records, key=lambda r: r["subject_id"])


def summarize(directory: Path):
    """Rebuild summary.csv / summary.txt from every subject record of the run."""
    directory = Path(directory)
    records = load_subjects(directory)
    if not records:
        return None
    columns = ["subject_id"] + [m for m in METRICS if any(m in r for r in records)]
    rows = [{c: r.get(c, "") for c in columns} for r in records]
    stats = {}
    for column in columns[1:]:
        values = np.array([r[column] for r in records if r.get(column) not in (None, "")], dtype=float)
        # Population standard deviation across target subjects, as in the paper.
        stats[column] = (values.mean(), values.std(), len(values))

    with open(directory / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow({"subject_id": "mean", **{c: f"{stats[c][0]:.6f}" for c in columns[1:]}})
        writer.writerow({"subject_id": "std", **{c: f"{stats[c][1]:.6f}" for c in columns[1:]}})

    config = yaml.safe_load((directory / "config.yaml").read_text()) if (directory / "config.yaml").exists() else {}
    lines = [f"Run: {config.get('name', directory.parent.name)} | dataset: {config.get('dataset', '')} | "
             f"seed: {config.get('seed', '')} | subjects: {[r['subject_id'] for r in records]}"]
    if "param_count" in records[0]:
        lines.append(f"Trainable parameters: {records[0]['param_count']}")
    for column in columns[1:]:
        mean, std, count = stats[column]
        scale = 100.0 if column.endswith("acc") else 1.0
        lines.append(f"{column:>20}: {mean * scale:.2f} +- {std * scale:.2f}  (n={count})")
    text = "\n".join(lines) + "\n"
    _atomic_write(directory / "summary.txt", text)
    return text


def collect(results_root=None, dataset=None):
    """Mean accuracy/kappa of every run under ``results/`` as a list of dicts."""
    root = Path(results_root or os.environ.get("SSM_DAN_RESULTS", DEFAULT_RESULTS))
    table = []
    for directory in sorted(root.glob(f"{dataset or '*'}/*/seed*")):
        records = load_subjects(directory)
        if not records:
            continue
        row = {"dataset": directory.parent.parent.name, "run": directory.parent.name,
               "seed": directory.name[4:], "n_subjects": len(records)}
        for metric in ("test_acc", "test_kappa", "no_tta_acc"):
            values = [r[metric] for r in records if metric in r]
            if values:
                row[f"{metric}_mean"] = float(np.mean(values))
                row[f"{metric}_std"] = float(np.std(values))
        table.append(row)
    return table
