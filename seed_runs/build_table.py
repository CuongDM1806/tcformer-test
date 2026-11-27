"""Build SEED_RESULTS.md from per_subject.csv (append new runs to the CSV, then rerun)."""
import csv, statistics as st
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
rows = list(csv.DictReader(open(HERE / "per_subject.csv")))
META = {"model", "dataset", "seed", "subject_id", "source"}
ALL_M = [c for c in rows[0].keys() if c not in META]
PCT = {c for c in ALL_M if c.endswith("_acc")}

def fmt(v, m):
    return f"{100*v:.2f}" if m in PCT else (f"{v:.3f}" if "kappa" in m else f"{v:.2f}")

def sd(xs):  # across subjects: population std (ddof=0), same as the run's summary log
    return st.pstdev(xs)

def sd_seed(xs):  # across seeds: sample std (ddof=1)
    return st.stdev(xs) if len(xs) > 1 else 0.0

# Trainable parameters, from the "Trainable parameters:" line of each run log.
PARAMS = {"atcnet": 113732, "eegconformer": 789572, "ttime": 9520, "rmetnet": 193783,
          "satransnet": 155636, "tcformer": 127212, "ssm_dan-source_only": 98416,
          "ssm_dan-no_da": 131890, "ssm_dan-no_ra-no_da": 131890, "ssm_dan-per_session_ra": 132022, "eegencoder": 197172}

runs = defaultdict(list)
for r in rows:
    runs[(r["dataset"], r["model"])].append(r)

out = ["# Seed results (chưa đưa vào bài báo)", "",
       "Acc theo %, kappa thô. Mean ± std qua subject (ddof=0, khớp log run); dòng across seeds = mean ± std (ddof=1) của các seed mean. Nguồn: `per_subject.csv`.",
       "Seed đánh dấu (paper): khôi phục từ bảng bài báo (chỉ có test acc/kappa theo subject; acc ×288 trial, kappa = (acc−0.25)/0.75); các cột khác để trống, dòng across seeds ghi n khi thiếu seed.", ""]
# ---- summary: one row per model, mean ± std over seed means ----
SUM = [("test_acc", "test_acc (%)"), ("test_kappa", "κ"), ("no_tta_acc", "no_tta_acc (%)"),
       ("train_time_min", "train (phút)"), ("cpu_latency_ms", "CPU (ms)")]
for ds in sorted({d for d, _ in runs}):
    items = []
    for (d, model), rs in runs.items():
        if d != ds:
            continue
        seeds = sorted({int(r["seed"]) for r in rs})
        cells = []
        for m, _ in SUM:
            xs = [st.mean(v) for sd_ in seeds
                  if (v := [float(r[m]) for r in rs if int(r["seed"]) == sd_ and r.get(m, "") != ""])]
            if not xs:
                cells.append("—")
            elif len(xs) == 1:
                cells.append(fmt(xs[0], m) + ("" if len(seeds) == 1 else " (n=1)"))
            else:
                cells.append(f"{fmt(st.mean(xs), m)} ± {fmt(sd_seed(xs), m)}"
                             + ("" if len(xs) == len(seeds) else f" (n={len(xs)})"))
        acc = st.mean(st.mean(float(r["test_acc"]) for r in rs if int(r["seed"]) == sd_) for sd_ in seeds)
        par = f"{PARAMS[model]/1000:.1f}" if model in PARAMS else "—"
        items.append((acc, f"| {model} | {','.join(map(str, seeds))} | " + " | ".join(cells) + f" | {par} |"))
    out += [f"## Tổng hợp — {ds} (mean ± std qua các seed, ddof=1)", "",
            "| model | seeds | " + " | ".join(h for _, h in SUM) + " | params (K) |",
            "|" + "---|" * (len(SUM) + 3)]
    out += [row for _, row in sorted(items)]
    out += ["", "Thời gian/latency phụ thuộc máy chạy từng seed (Kaggle), chỉ nên so sánh thô. (n=k): chỉ k seed có cột đó.", ""]

for (ds, model), rs in sorted(runs.items()):
    seeds = sorted({int(r["seed"]) for r in rs})
    subs = sorted({int(r["subject_id"]) for r in rs})
    M = [m for m in ALL_M if any(r.get(m, "") != "" for r in rs)]  # only metrics this model reports
    out += [f"## {model} — {ds}", ""]
    # per-seed summary
    out += ["| seed | n | " + " | ".join(M) + " |", "|" + "---|" * (len(M) + 2)]
    seed_means = defaultdict(list)
    for s in seeds:
        sr = [r for r in rs if int(r["seed"]) == s]
        cells = []
        for m in M:
            xs = [float(r[m]) for r in sr if r[m] != ""]
            if not xs:
                cells.append("—")
                continue
            seed_means[m].append(st.mean(xs))
            cells.append(f"{fmt(st.mean(xs), m)} ± {fmt(sd(xs), m)}")
        src = "/".join(sorted({r.get("source", "run") for r in sr}))
        tag = "" if src == "run" else f" ({src})"
        out.append(f"| {s}{tag} | {len(sr)} | " + " | ".join(cells) + " |")
    if len(seeds) > 1:
        def across(xs, m):
            if not xs:
                return "—"
            if len(xs) == 1:
                return f"{fmt(xs[0], m)} (n=1)"
            tail = "" if len(xs) == len(seeds) else f" (n={len(xs)})"
            return f"{fmt(st.mean(xs), m)} ± {fmt(sd_seed(xs), m)}{tail}"
        cells = [across(seed_means[m], m) for m in M]
        out.append(f"| **across seeds** | {len(seeds)} | " + " | ".join(cells) + " |")
    out.append("")
    # per-subject test_acc by seed
    out += ["test_acc (%) theo subject:", "",
            "| subject | " + " | ".join(f"s{s}" for s in seeds) + (" | mean | std |" if len(seeds) > 1 else " |"),
            "|" + "---|" * (len(seeds) + 1 + (2 if len(seeds) > 1 else 0))]
    for sub in subs:
        xs = []
        cells = []
        for s in seeds:
            v = [float(r["test_acc"]) for r in rs if int(r["seed"]) == s and int(r["subject_id"]) == sub]
            cells.append(fmt(v[0], "test_acc") if v else "—")
            xs += v
        tail = f" | {100*st.mean(xs):.2f} | {100*sd_seed(xs):.2f} |" if len(seeds) > 1 else " |"
        out.append(f"| {sub} | " + " | ".join(cells) + tail)
    out.append("")

(HERE / "SEED_RESULTS.md").write_text("\n".join(out))
print("\n".join(out))
