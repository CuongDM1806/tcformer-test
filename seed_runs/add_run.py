"""Append one run to per_subject.csv from the notebook's pasted per-subject table.

Usage: python3 add_run.py MODEL DATASET SEED < paste.txt
The paste is one value per line: the header names (any number), then one value
per header for each subject, until the 'mean' row (mean/std rows are ignored and recomputed). Replaces any
existing rows for the same (model, dataset, seed).
"""
import csv, sys
from pathlib import Path

model, ds, seed = sys.argv[1:4]
toks = [t.strip() for t in sys.stdin.read().split() if t.strip()]
k = toks.index("1")  # first subject id ends the header
hdr = toks[:k]
assert hdr[0] == "subject_id", hdr
vals = toks[k:toks.index("mean")]
assert len(vals) % k == 0, len(vals)

path = Path(__file__).parent / "per_subject.csv"
rows = list(csv.DictReader(open(path)))
cols = list(rows[0].keys())
for i, c in enumerate(hdr):  # new metric columns go right after their predecessor in the paste
    if c not in cols:
        cols.insert(cols.index(hdr[i - 1]) + 1, c)
rows = [r for r in rows if (r["model"], r["dataset"], r["seed"]) != (model, ds, seed)]
for i in range(0, len(vals), k):
    d = dict(zip(hdr, vals[i:i + k]))
    rows.append({"model": model, "dataset": ds, "seed": seed, "source": "run", **d})
for r in rows:
    for c in cols:
        r.setdefault(c, "")
rows.sort(key=lambda r: (r["dataset"], r["model"], int(r["seed"]), int(r["subject_id"])))
w = csv.DictWriter(open(path, "w", newline=""), fieldnames=cols, lineterminator="\n")
w.writeheader(); w.writerows(rows)
print(f"{model} {ds} seed {seed}: {len(vals)//k} subjects, columns {hdr[1:]}")
