"""Exact Wilcoxon signed-rank tests with Holm correction for Tables 3-5.

Pure Python (no scipy/statsmodels needed). Comparison families follow
Section 3.5 of the manuscript: each method is compared with the SSM-DAN
variant that has the same target-data access, and Holm is applied within
each family (F1: vs SSM-DAN, F2: vs SSM-DAN (Stage 1 UDA), F3: vs SSM-DAN (IM-TTA)).
"""
from itertools import product

# Per-subject accuracy (%) copied from Tables 3-5.
ACC = {}
ACC["BCI IV-2a"] = {  # per-subject accuracy (%), S1..S9, from Table 3
    "Conformer":             [58.68, 43.06, 78.82, 43.06, 44.44, 41.67, 72.92, 65.28, 49.31],
    "ATCNet":                [69.44, 40.97, 76.74, 50.35, 43.40, 42.01, 61.81, 70.49, 57.99],
    "TCFormer":              [65.28, 49.65, 79.86, 54.86, 61.81, 38.89, 71.88, 66.67, 58.33],
    "EEGEncoder":            [64.24, 49.31, 78.82, 46.18, 54.86, 40.97, 74.31, 65.97, 47.92],
    "SATrans-Net":           [68.06, 47.92, 73.96, 53.47, 61.11, 45.14, 73.61, 71.53, 63.19],
    "SSM-DAN (source-only)": [68.06, 51.04, 86.46, 61.11, 64.24, 39.58, 72.92, 73.61, 62.15],
    "RMETNet":               [67.01, 39.24, 77.78, 55.90, 49.65, 52.78, 63.54, 66.67, 68.40],
    "SSM-DAN (Stage 1 UDA)": [79.17, 47.57, 92.71, 73.61, 57.64, 59.72, 82.29, 82.64, 73.96],
    "T-TIME":                [67.01, 53.47, 78.47, 64.58, 68.75, 49.65, 79.51, 69.44, 65.28],
    "BFT":                   [64.24, 44.44, 76.74, 65.62, 61.81, 51.39, 73.61, 70.49, 59.38],
    "SSM-DAN (IM-TTA)":      [77.43, 47.92, 88.19, 66.32, 64.24, 61.81, 79.86, 82.99, 73.61],
    "SSM-DAN":               [79.51, 52.43, 93.40, 73.96, 69.44, 65.97, 88.89, 86.11, 75.35],
}
ACC["BCI IV-2b"] = {
    "Conformer":             [75.31, 62.86, 55.62, 92.50, 94.69, 74.06, 82.81, 82.50, 81.88],
    "ATCNet":                [76.25, 65.71, 57.19, 90.00, 96.88, 77.50, 84.06, 84.69, 82.19],
    "TCFormer":              [78.44, 69.64, 64.38, 90.62, 95.00, 75.94, 86.87, 85.31, 84.38],
    "EEGEncoder":            [73.44, 65.71, 59.06, 92.50, 97.19, 85.00, 85.94, 86.56, 83.75],
    "SATrans-Net":           [77.50, 69.29, 69.69, 90.62, 97.50, 79.69, 87.50, 84.38, 79.69],
    "SSM-DAN (source-only)": [77.81, 68.57, 63.13, 89.38, 96.56, 73.44, 84.69, 86.25, 85.00],
    "RMETNet":               [73.12, 57.86, 55.00, 91.56, 94.06, 82.81, 85.62, 88.75, 83.13],
    "SSM-DAN (Stage 1 UDA)": [78.44, 70.71, 63.44, 90.31, 97.81, 83.75, 89.69, 95.63, 85.94],
    "T-TIME":                [59.69, 65.36, 85.31, 93.75, 97.19, 85.00, 74.69, 91.25, 59.06],
    "BFT":                   [75.00, 65.00, 59.69, 93.44, 94.69, 79.06, 88.12, 94.37, 81.25],
    "SSM-DAN (IM-TTA)":      [77.81, 68.57, 66.25, 93.75, 96.25, 81.88, 89.69, 94.69, 81.25],
    "SSM-DAN":               [77.81, 69.29, 65.94, 90.00, 97.19, 84.69, 91.56, 95.31, 85.00],
}
ACC["Zhou2016"] = {
    "Conformer":             [76.00, 79.33, 70.67, 82.67],
    "ATCNet":                [72.00, 82.67, 76.67, 74.67],
    "TCFormer":              [68.00, 78.67, 66.67, 80.67],
    "EEGEncoder":            [62.00, 84.00, 74.67, 76.67],
    "SATrans-Net":           [81.33, 80.00, 67.33, 70.00],
    "SSM-DAN (source-only)": [72.00, 78.67, 71.33, 78.67],
    "RMETNet":               [70.00, 89.33, 80.67, 84.67],
    "SSM-DAN (Stage 1 UDA)": [79.33, 84.67, 87.33, 78.00],
    "T-TIME":                [88.67, 84.00, 81.33, 86.00],
    "BFT":                   [84.00, 86.67, 76.00, 84.67],
    "SSM-DAN (IM-TTA)":      [79.33, 84.00, 83.33, 83.33],
    "SSM-DAN":               [84.00, 83.33, 84.67, 85.33],
}

# (method, reference, family)
COMPARISONS = [
    ("Conformer", "SSM-DAN", "F1"),
    ("ATCNet", "SSM-DAN", "F1"),
    ("TCFormer", "SSM-DAN", "F1"),
    ("EEGEncoder", "SSM-DAN", "F1"),
    ("SATrans-Net", "SSM-DAN", "F1"),
    ("SSM-DAN (source-only)", "SSM-DAN", "F1"),
    ("SSM-DAN (Stage 1 UDA)", "SSM-DAN", "F1"),
    ("SSM-DAN (IM-TTA)", "SSM-DAN", "F1"),
    ("RMETNet", "SSM-DAN (Stage 1 UDA)", "F2"),
    ("T-TIME", "SSM-DAN (IM-TTA)", "F3"),
    ("BFT", "SSM-DAN (IM-TTA)", "F3"),
]


def ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    r = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def wilcoxon_exact(a, b):
    """Two-sided exact signed-rank test; zero differences are dropped."""
    d = [round(x - y, 6) for x, y in zip(a, b) if round(x - y, 6) != 0]
    r = ranks([abs(x) for x in d])
    w_obs = sum(rk for rk, x in zip(r, d) if x > 0)
    total = sum(r)
    dev_obs = abs(w_obs - total / 2)
    hits = sum(
        abs(sum(rk for rk, s in zip(r, signs) if s) - total / 2) >= dev_obs - 1e-9
        for signs in product((0, 1), repeat=len(d))
    )
    return hits / 2 ** len(d), sum(x > 0 for x in d), len(d)


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj, running = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


if __name__ == "__main__":
    for dataset, acc in ACC.items():
        rows = []
        for method, ref, fam in COMPARISONS:
            p, wins, n = wilcoxon_exact(acc[ref], acc[method])
            rows.append([method, ref, fam, wins, n, p])

        for fam in sorted({r[2] for r in rows}):
            idx = [i for i, r in enumerate(rows) if r[2] == fam]
            for i, ph in zip(idx, holm([rows[i][5] for i in idx])):
                rows[i].append(ph)

        print(f"\n== {dataset}")
        print(f"{'Method':24s} {'Reference':24s} Fam  Wins   p_raw   p_Holm  Sig")
        for m, ref, fam, wins, n, p, ph in rows:
            print(f"{m:24s} {ref:24s} {fam:3s}  {wins}/{n}  {p:.4f}  {ph:.4f}  {'yes' if ph < 0.05 else 'no'}")
