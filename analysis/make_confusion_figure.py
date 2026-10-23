"""Draw the mean confusion matrices of SSM-DAN on the three datasets (one figure).

Writes a standalone TikZ file next to the paper's other figures and compiles it
to PDF. Each matrix is row-normalised per target subject and then averaged over
subjects, so its diagonal equals the mean class recall reported in the text.

    python3 analysis/make_confusion_figure.py
"""
from pathlib import Path
from statistics import mean
import subprocess

from holm_tables import ACC
from extra_metrics import CLASSES, CONFMAT_PCT, counts_from_percent, recall_f1

FIGURE_DIR = Path(__file__).resolve().parents[1] / "docs/elsevier_bspc_template/original/elsarticle/figures"
NAME = "classwise_confusion"

# Common class order and display names across the three panels.
ORDER = ["hand(L)", "hand(R)", "feet", "tongue"]
LABEL = {"hand(L)": "Left hand", "hand(R)": "Right hand", "feet": "Feet", "tongue": "Tongue"}
TITLE = {"BCI IV-2a": "BCI IV-2a", "BCI IV-2b": "BCI IV-2b", "Zhou2016": "Zhou2016"}

# Sequential single-hue ramp (near zero recedes to the panel colour).
RAMP = [(0.0, (0xF6, 0xF8, 0xFA)), (0.25, (0xCD, 0xE2, 0xFB)), (0.5, (0x6D, 0xA7, 0xEC)),
        (0.75, (0x25, 0x6A, 0xBF)), (1.0, (0x0D, 0x36, 0x6B))]
CELL = 1.0  # cm
GAP = 1.9   # cm between panels (room for the next panel's row labels)


def colour(value):
    """Interpolate the ramp at value in [0, 100]."""
    x = max(0.0, min(1.0, value / 100))
    for (x0, c0), (x1, c1) in zip(RAMP, RAMP[1:]):
        if x <= x1:
            t = (x - x0) / (x1 - x0)
            return "".join(f"{round(a + t * (b - a)):02X}" for a, b in zip(c0, c1))
    return "".join(f"{v:02X}" for v in RAMP[-1][1])


def mean_matrix(dataset):
    classes = CLASSES[dataset]
    perm = [classes.index(c) for c in ORDER if c in classes]
    rows, f1s = [], []
    for pct, acc in zip(CONFMAT_PCT[dataset], ACC[dataset]["SSM-DAN"]):
        cm = counts_from_percent(pct, acc)
        f1s.append(mean(recall_f1(cm)[1]))
        rows.append([[100 * cm[i][j] / sum(cm[i]) for j in perm] for i in perm])
    k = len(perm)
    avg = [[mean(r[i][j] for r in rows) for j in range(k)] for i in range(k)]
    return [classes[i] for i in perm], avg, mean(f1s)


def panel(x0, dataset):
    names, cm, f1 = mean_matrix(dataset)
    k = len(names)
    top = 4 * CELL  # align all panels to the top edge of the 4x4 panel
    out = [f"% {dataset}"]
    for i in range(k):
        for j in range(k):
            v = cm[i][j]
            x, y = x0 + j * CELL, top - (i + 1) * CELL
            ink = "white" if v >= 55 else "ink"
            weight = r"\bfseries" if i == j else ""
            out.append(
                f"\\fill[fill={{rgb,255:red,{int(colour(v)[0:2],16)};green,{int(colour(v)[2:4],16)};blue,{int(colour(v)[4:6],16)}}}] "
                f"({x + 0.02:.3f},{y + 0.02:.3f}) rectangle ({x + CELL - 0.02:.3f},{y + CELL - 0.02:.3f});"
            )
            out.append(f"\\node[text={ink}, font=\\normalsize{weight}] at ({x + CELL / 2:.3f},{y + CELL / 2:.3f}) {{{v:.1f}}};")
        out.append(
            f"\\node[anchor=east, font=\\small, text=ink] at ({x0 - 0.06:.3f},{top - (i + 0.5) * CELL:.3f}) {{{LABEL[names[i]]}}};"
        )
        out.append(
            f"\\node[anchor=east, rotate=40, font=\\small, text=ink] at ({x0 + (i + 0.5) * CELL + 0.12:.3f},{top - k * CELL - 0.08:.3f}) {{{LABEL[names[i]]}}};"
        )
    out.append(f"\\node[anchor=south west, font=\\large\\bfseries, text=navy] at ({x0 - 0.04:.3f},{top + 0.45:.3f}) {{{TITLE[dataset]}}};")
    out.append(f"\\node[anchor=south west, text=muted, font=\\small] at ({x0 - 0.04:.3f},{top + 0.02:.3f}) {{macro-F1 = {f1:.3f}}};")
    return out, k


def main():
    body, x = [], 0.0
    for dataset in ACC:
        lines, k = panel(x, dataset)
        body += lines
        x += k * CELL + GAP

    # Colour bar, shared by the three panels.
    bx, h = x - GAP + 0.45, 4 * CELL
    steps = 50
    for s in range(steps):
        c = colour(100 * (s + 0.5) / steps)
        body.append(
            f"\\fill[fill={{rgb,255:red,{int(c[0:2],16)};green,{int(c[2:4],16)};blue,{int(c[4:6],16)}}}] "
            f"({bx:.3f},{h * s / steps:.3f}) rectangle ({bx + 0.25:.3f},{h * (s + 1) / steps + 0.002:.3f});"
        )
    for v in (0, 25, 50, 75, 100):
        y = h * v / 100
        body.append(f"\\draw[muted, line width=0.4pt] ({bx + 0.25:.3f},{y:.3f}) -- ({bx + 0.33:.3f},{y:.3f});")
        body.append(f"\\node[anchor=west, font=\\small, text=muted] at ({bx + 0.33:.3f},{y:.3f}) {{{v}}};")
    body.append(f"\\node[rotate=90, font=\\small, text=ink] at ({bx + 1.05:.3f},{h / 2:.3f}) {{Row-normalised rate (\\%)}};")
    body.append(f"\\node[rotate=90, font=\\small, text=ink] at ({-2.2:.3f},{h / 2:.3f}) {{True class}};")
    body.append(f"\\node[font=\\small, text=ink] at ({(x - GAP) / 2:.3f},{-1.3:.3f}) {{Predicted class}};")

    tex = r"""\documentclass[tikz,border=4pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{helvet}
\renewcommand{\familydefault}{\sfdefault}
\definecolor{navy}{HTML}{17324D}
\definecolor{ink}{HTML}{263442}
\definecolor{muted}{HTML}{607385}
\begin{document}
\begin{tikzpicture}[font=\sffamily]
""" + "\n".join(body) + "\n\\end{tikzpicture}\n\\end{document}\n"
    (FIGURE_DIR / f"{NAME}.tex").write_text(tex)
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"{NAME}.tex"],
                   cwd=FIGURE_DIR, check=True, stdout=subprocess.DEVNULL)
    for ext in ("aux", "log"):
        (FIGURE_DIR / f"{NAME}.{ext}").unlink(missing_ok=True)
    print(f"wrote {FIGURE_DIR / (NAME + '.pdf')}")


if __name__ == "__main__":
    main()
