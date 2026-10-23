"""ITR, per-class recall and macro-F1 for the manuscript (Tables 3-5 and 9).

Pure Python. ITR (Wolpaw) is computed per subject from the accuracies in
Tables 3-5 (holm_tables.ACC) and then averaged. Recall and macro-F1 are
computed for SSM-DAN from its saved confusion-matrix figures
(results/*/confmats/confmat_subject_*.png in the autodial_*_results.zip
archives), which show row-normalised percentages rounded to 0.1. Trial
counts per class are recovered by finding the class sizes that reproduce
every printed percentage and the accuracy reported in results.txt.
"""
import math
from statistics import mean, stdev

from holm_tables import ACC

# Window length T (s) fed to the decoder: BCI IV-2a full MOABB window [2, 6] s,
# BCI IV-2b [3, 7.5] s minus the last 0.5 s, Zhou2016 0.5-5.0 s after cue.
TRIAL_SECONDS = {"BCI IV-2a": 4.0, "BCI IV-2b": 4.0, "Zhou2016": 4.5}
CLASSES = {
    "BCI IV-2a": ["feet", "hand(L)", "hand(R)", "tongue"],
    "BCI IV-2b": ["hand(L)", "hand(R)"],
    "Zhou2016": ["hand(L)", "hand(R)", "feet"],
}

# Row-normalised confusion matrices (%) of SSM-DAN, read from the figures.
CONFMAT_PCT = {
    "BCI IV-2a": [
        [[73.6, 5.6, 1.4, 19.4], [2.8, 77.8, 13.9, 5.6], [0.0, 12.5, 87.5, 0.0], [19.4, 1.4, 0.0, 79.2]],
        [[66.7, 15.3, 1.4, 16.7], [19.4, 40.3, 22.2, 18.1], [12.5, 6.9, 59.7, 20.8], [8.3, 20.8, 27.8, 43.1]],
        [[97.2, 1.4, 1.4, 0.0], [4.2, 88.9, 2.8, 4.2], [1.4, 4.2, 91.7, 2.8], [1.4, 2.8, 0.0, 95.8]],
        [[56.9, 6.9, 11.1, 25.0], [5.6, 84.7, 9.7, 0.0], [9.7, 8.3, 77.8, 4.2], [4.2, 5.6, 13.9, 76.4]],
        [[68.1, 9.7, 12.5, 9.7], [4.2, 72.2, 6.9, 16.7], [5.6, 2.8, 75.0, 16.7], [13.9, 9.7, 13.9, 62.5]],
        [[61.1, 12.5, 11.1, 15.3], [4.2, 73.6, 15.3, 6.9], [8.3, 13.9, 61.1, 16.7], [4.2, 8.3, 19.4, 68.1]],
        [[81.9, 8.3, 5.6, 4.2], [2.8, 95.8, 1.4, 0.0], [6.9, 4.2, 87.5, 1.4], [5.6, 0.0, 4.2, 90.3]],
        [[76.4, 13.9, 6.9, 2.8], [5.6, 90.3, 0.0, 4.2], [0.0, 2.8, 91.7, 5.6], [5.6, 1.4, 6.9, 86.1]],
        [[62.5, 9.7, 18.1, 9.7], [0.0, 87.5, 11.1, 1.4], [19.4, 0.0, 70.8, 9.7], [5.6, 5.6, 8.3, 80.6]],
    ],
    "BCI IV-2b": [
        [[75.0, 25.0], [19.4, 80.6]],
        [[70.7, 29.3], [32.1, 67.9]],
        [[66.2, 33.8], [34.4, 65.6]],
        [[91.9, 8.1], [11.9, 88.1]],
        [[98.8, 1.2], [4.4, 95.6]],
        [[86.2, 13.8], [16.9, 83.1]],
        [[90.6, 9.4], [7.5, 92.5]],
        [[93.1, 6.9], [2.5, 97.5]],
        [[75.6, 24.4], [5.6, 94.4]],
    ],
    "Zhou2016": [
        [[96.0, 4.0, 0.0], [4.0, 82.0, 14.0], [16.0, 10.0, 74.0]],
        [[78.0, 20.0, 2.0], [24.0, 76.0, 0.0], [0.0, 4.0, 96.0]],
        [[86.0, 2.0, 12.0], [10.0, 78.0, 12.0], [4.0, 6.0, 90.0]],
        [[80.0, 14.0, 6.0], [8.0, 86.0, 6.0], [6.0, 4.0, 90.0]],
    ],
}


def wolpaw_bits(p, n_classes):
    """Bits per trial; zero at or below chance."""
    if p <= 1.0 / n_classes:
        return 0.0
    bits = math.log2(n_classes) + p * math.log2(p)
    if p < 1.0:
        bits += (1 - p) * math.log2((1 - p) / (n_classes - 1))
    return bits


def itr(acc_percent, n_classes, seconds):
    return wolpaw_bits(acc_percent / 100, n_classes) * 60 / seconds


def row_size(row):
    """Smallest class size that reproduces every printed percentage."""
    for n in range(20, 400):
        if all(abs(round(round(v * n / 100) / n * 100, 1) - v) < 1e-6 for v in row):
            yield n


def counts_from_percent(pct, acc_percent):
    """Recover integer counts whose accuracy matches the reported one."""
    candidates = [list(row_size(row))[:40] for row in pct]

    def search(i, sizes):
        if i == len(pct):
            total = sum(sizes)
            correct = sum(round(pct[k][k] * sizes[k] / 100) for k in range(len(pct)))
            return sizes if abs(100 * correct / total - acc_percent) < 0.005 else None
        for n in candidates[i]:
            found = search(i + 1, sizes + [n])
            if found:
                return found
        return None

    sizes = search(0, [])
    if sizes is None:
        raise ValueError("no class sizes reproduce the figure and accuracy")
    return [[round(v * n / 100) for v in row] for row, n in zip(pct, sizes)]


def recall_f1(cm):
    k = len(cm)
    recall, f1 = [], []
    for c in range(k):
        tp = cm[c][c]
        actual = sum(cm[c])
        predicted = sum(cm[r][c] for r in range(k))
        rec = tp / actual
        prec = tp / predicted if predicted else 0.0
        recall.append(rec)
        f1.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return recall, f1


def fmt(values, scale=1.0, digits=2):
    values = [v * scale for v in values]
    return f"{mean(values):.{digits}f} ± {stdev(values):.{digits}f}"


if __name__ == "__main__":
    for dataset, acc in ACC.items():
        n_classes = len(CLASSES[dataset])
        seconds = TRIAL_SECONDS[dataset]
        print(f"\n== {dataset}  (N = {n_classes}, T = {seconds} s)")
        print("ITR (bits/min), mean ± SD over subjects:")
        for method, subject_acc in acc.items():
            print(f"  {method:24s} {fmt([itr(a, n_classes, seconds) for a in subject_acc])}")

        recalls, macro_f1 = [], []
        for pct, a in zip(CONFMAT_PCT[dataset], acc["SSM-DAN"]):
            cm = counts_from_percent(pct, a)
            rec, f1 = recall_f1(cm)
            recalls.append(rec)
            macro_f1.append(mean(f1))
        print("SSM-DAN per-class recall (%), mean ± SD over subjects:")
        for c, name in enumerate(CLASSES[dataset]):
            print(f"  {name:8s} {fmt([r[c] for r in recalls], 100)}")
        print(f"SSM-DAN macro-F1: {fmt(macro_f1, 1, 3)}")
        print("  per subject:", " ".join(f"{v:.3f}" for v in macro_f1))
