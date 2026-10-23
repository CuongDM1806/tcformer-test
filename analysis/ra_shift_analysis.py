"""Input-level subject shift before and after Riemannian alignment (RA).

Run from the root of the code branch (feature/hada-full-mamba-autodial), so
that the paper's own loaders, session splits, and RA functions are used:

    python ra_shift_analysis.py <output_dir>

For every subject, RA is fitted exactly as in training: the reference R is the
Riemannian mean of the OAS trial covariances of the subject's *earlier*
sessions (BCI IV-2a: session 1; BCI IV-2b: sessions 1-3; Zhou2016: sessions
1-2) and is then applied unchanged to the *evaluation* sessions.

Part 1 (evaluation sessions, which RA never saw): how far each target lies from
the source domain before and after RA. After RA the source centre is the
identity and the target is whitened by its own earlier-session mean, so the
remaining distance equals the target's own session shift exactly (RA is a
per-subject congruence): RA removes the inter-subject part of the shift and
cannot touch the cross-session part.

Part 2 (why RA helps BCI IV-2a most), two hypotheses:
  H1  RA mainly helps by homogenising the *training* sessions of the source
      subjects. Measured by the inter-subject shift of the fitting sessions
      before RA (after RA it is zero by construction).
  H2  The shift RA removes on BCI IV-2a is mostly *spatial shape* rather than
      overall *scale*. Every distance is split exactly into
          d^2 / C = scale^2 + shape^2,
      with l_i the log-eigenvalues of A^-1 B, scale = |mean(l)| (common gain on
      all spatial directions) and shape = std(l) (direction-dependent change of
      the spatial covariance). A pure gain leaves spatial filters valid; a shape
      change makes them point at the wrong mixture of channels.
All distances are divided by sqrt(C) (C = number of channels) so that the
datasets can be compared. Differences between datasets are tested with
Kruskal-Wallis and pairwise Mann-Whitney tests (Holm-adjusted) over subjects.

Outputs (in <output_dir>): ra_shift_summary.json, ra_shift_results.npz and the
figures ra_tsne_bcic2a, ra_pairwise_bcic2a, ra_distance_bars, ra_hypotheses
(.png and .pdf). No model is trained.
"""
import json
import os
import sys
from itertools import combinations
from pathlib import Path

# Notebook kernels export their inline backend; the project's imports
# (pytorch_lightning -> torchmetrics -> matplotlib) would fail on it.
os.environ["MPLBACKEND"] = "Agg"

import numpy as np
from scipy.linalg import eigh
from scipy.stats import kruskal, mannwhitneyu
from sklearn.covariance import OAS
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

sys.path.insert(0, os.getcwd())
from datamodules.base import BaseDataModule  # noqa: E402
from datamodules.bcic4_2a import _get_2a_train_test_sessions  # noqa: E402
from datamodules.bcic4_2b import _get_ordered_sessions  # noqa: E402
from datamodules.zhou2016 import _get_three_sessions  # noqa: E402
from utils.load_bcic4 import load_bcic4  # noqa: E402
from utils.load_zhou2016 import load_zhou2016  # noqa: E402

# Preprocessing used for the paper's runs (config.yaml of each result archive).
PREPROCESSING = {
    "BCI IV-2a": dict(sfreq=250, low_cut=None, high_cut=None, start=0.0, stop=0.0),
    "BCI IV-2b": dict(sfreq=250, low_cut=None, high_cut=None, start=0.0, stop=-0.5),
    "Zhou2016": dict(sfreq=250, low_cut=8.0, high_cut=30.0, start=0.5, trial_duration=4.5),
}
SUBJECTS = {"BCI IV-2a": range(1, 10), "BCI IV-2b": range(1, 10), "Zhou2016": range(1, 5)}
SLUG = {"BCI IV-2a": "bcic2a", "BCI IV-2b": "bcic2b", "Zhou2016": "zhou2016"}

# Colours: categorical palette (fixed order) and before/after/session roles.
SUBJECT_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                   "#008300", "#6250d6", "#e34948", "#52514e"]
BEFORE, AFTER, SESSION = "#a3a29c", "#2a78d6", "#eb6834"
SHAPE, SCALE = "#2a78d6", "#eb6834"
INK, MUTED = "#0b0b0b", "#52514e"


def load_dataset(name):
    if name == "Zhou2016":
        return load_zhou2016(list(SUBJECTS[name]), PREPROCESSING[name])
    return load_bcic4(list(SUBJECTS[name]), "2a" if name == "BCI IV-2a" else "2b",
                      PREPROCESSING[name])


def split_sessions(name, subject_dataset):
    """Return (fitting sessions, evaluation sessions) as in Table 2."""
    if name == "BCI IV-2a":
        first, second = _get_2a_train_test_sessions(subject_dataset)
        return [first], [second]
    if name == "BCI IV-2b":
        sessions = _get_ordered_sessions(subject_dataset)
        return sessions[:3], sessions[3:5]
    sessions = _get_three_sessions(subject_dataset)
    return sessions[:2], sessions[2:]


def to_array(parts):
    return np.concatenate([BaseDataModule._dataset_to_arrays(p)[0] for p in parts], axis=0)


def covariances(X):
    return np.stack([OAS().fit(trial.T.astype(np.float64)).covariance_ for trial in X])


def spd_power(C, exponent):
    w, V = eigh(C)
    return (V * np.maximum(w, 1e-12) ** exponent) @ V.T


def log_eigenvalues(A, B):
    """Log-eigenvalues of A^-1 B."""
    return np.log(np.maximum(eigh(B, A, eigvals_only=True), 1e-12))


def riemann_distance(A, B):
    """Affine-invariant distance: sqrt(sum log^2 of the eigenvalues of A^-1 B)."""
    return float(np.sqrt(np.sum(log_eigenvalues(A, B) ** 2)))


def decompose(A, B):
    """Per-channel distance and its exact split into scale and shape parts.

    Returns (d / sqrt(C), scale, shape) with (d / sqrt(C))^2 = scale^2 + shape^2.
    """
    l = log_eigenvalues(A, B)
    return float(np.sqrt(np.mean(l ** 2))), float(abs(l.mean())), float(l.std())


def tangent_vectors(covs, reference):
    """Log-map at the reference, upper triangle with sqrt(2) off-diagonal weights."""
    R = spd_power(reference, -0.5)
    n = reference.shape[0]
    iu = np.triu_indices(n)
    weight = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    out = []
    for C in covs:
        w, V = eigh(R @ C @ R)
        L = (V * np.log(np.maximum(w, 1e-12))) @ V.T
        out.append(L[iu] * weight)
    return np.stack(out)


def analyse(name):
    print(f"\n=== {name}: loading", flush=True)
    by_subject = load_dataset(name).split("subject")
    subjects = list(SUBJECTS[name])
    mean = BaseDataModule._riemannian_mean
    M = {key: [] for key in ("fit_raw", "eval_raw", "fit_ra", "eval_ra")}
    eval_covs = {"raw": [], "ra": []}
    labels = []
    for s in subjects:
        fit, evaluation = split_sessions(name, by_subject[str(s)])
        X_fit, X_eval = to_array(fit), to_array(evaluation)
        # The whitener used in training (reference = fitting sessions). RA maps
        # every covariance by the congruence W C W; re-estimating OAS on the
        # aligned signals would add shrinkage bias, so the congruence is applied
        # to the covariances directly.
        W = BaseDataModule._riemannian_whitener(X_fit)
        C = {"fit_raw": covariances(X_fit), "eval_raw": covariances(X_eval)}
        C["fit_ra"] = np.einsum("ij,njk,kl->nil", W, C["fit_raw"], W)
        C["eval_ra"] = np.einsum("ij,njk,kl->nil", W, C["eval_raw"], W)
        for key in M:
            M[key].append(mean(C[key]))
        eval_covs["raw"].append(C["eval_raw"])
        eval_covs["ra"].append(C["eval_ra"])
        labels += [s] * len(X_eval)
        print(f"  subject {s}: {len(X_fit)} fitting / {len(X_eval)} evaluation trials", flush=True)

    k = len(subjects)
    n_channels = M["fit_raw"][0].shape[0]
    norm = np.sqrt(n_channels)
    others = lambda key, i: mean(np.stack([M[key][j] for j in range(k) if j != i]))  # noqa: E731
    centre = {v: [others(f"fit_{v}", i) for i in range(k)] for v in ("raw", "ra")}

    # ---- Part 1: evaluation sessions --------------------------------------
    pairwise = {v: np.array([[riemann_distance(M[f"eval_{v}"][i], M[f"eval_{v}"][j])
                              for j in range(k)] for i in range(k)]) for v in ("raw", "ra")}
    # Distance from the target's evaluation session to the centre of the
    # source domain (Riemannian mean of the other subjects' fitting sessions).
    to_source = {v: np.array([riemann_distance(centre[v][i], M[f"eval_{v}"][i]) for i in range(k)])
                 for v in ("raw", "ra")}
    # Within-subject session shift (fitting vs evaluation sessions). RA applies
    # one congruence per subject, which leaves this distance unchanged.
    session_shift = np.array([riemann_distance(M["fit_raw"][i], M["eval_raw"][i]) for i in range(k)])

    # ---- Part 2: hypotheses -----------------------------------------------
    # H1: inter-subject shift of the training (fitting) sessions, before RA;
    # after RA it vanishes by construction (kept as a sanity check).
    train_split = {v: np.array([decompose(centre[v][i], M[f"fit_{v}"][i]) for i in range(k)])
                   for v in ("raw", "ra")}
    off = ~np.eye(k, dtype=bool)
    train_pairwise = np.array([[riemann_distance(M["fit_raw"][i], M["fit_raw"][j])
                                for j in range(k)] for i in range(k)])
    # H2: scale/shape split of the target-to-source distance on the evaluation
    # sessions, before and after RA; the difference is what RA removed.
    eval_split = {v: np.array([decompose(centre[v][i], M[f"eval_{v}"][i]) for i in range(k)])
                  for v in ("raw", "ra")}
    session_split = np.array([decompose(M["fit_raw"][i], M["eval_raw"][i]) for i in range(k)])

    labels = np.array(labels)
    raw_all = np.concatenate(eval_covs["raw"])
    ra_all = np.concatenate(eval_covs["ra"])
    tangent = {
        "raw": tangent_vectors(raw_all, mean(np.stack(M["eval_raw"]))),
        "ra": tangent_vectors(ra_all, np.eye(n_channels)),
    }
    silhouette = {v: float(silhouette_score(tangent[v], labels)) for v in tangent}

    summary = {
        "subjects": subjects,
        "n_channels": int(n_channels),
        "pairwise_mean_before": float(pairwise["raw"][off].mean()),
        "pairwise_mean_after": float(pairwise["ra"][off].mean()),
        "to_source_before": to_source["raw"].tolist(),
        "to_source_after": to_source["ra"].tolist(),
        "session_shift": session_shift.tolist(),
        "silhouette_by_subject_before": silhouette["raw"],
        "silhouette_by_subject_after": silhouette["ra"],
        # Part 2, all values per channel (divided by sqrt(C)).
        "h1_train_shift": train_split["raw"][:, 0].tolist(),
        "h1_train_shift_after_ra": train_split["ra"][:, 0].tolist(),
        "h1_train_pairwise_mean": float(train_pairwise[off].mean() / norm),
        "h2_train_scale": train_split["raw"][:, 1].tolist(),
        "h2_train_shape": train_split["raw"][:, 2].tolist(),
        "h2_eval_scale_before": eval_split["raw"][:, 1].tolist(),
        "h2_eval_shape_before": eval_split["raw"][:, 2].tolist(),
        "h2_eval_scale_after": eval_split["ra"][:, 1].tolist(),
        "h2_eval_shape_after": eval_split["ra"][:, 2].tolist(),
        "session_scale": session_split[:, 1].tolist(),
        "session_shape": session_split[:, 2].tolist(),
    }
    summary["pairwise_reduction_percent"] = 100 * (
        1 - summary["pairwise_mean_after"] / summary["pairwise_mean_before"])
    summary["to_source_reduction_percent"] = 100 * (
        1 - to_source["ra"].mean() / to_source["raw"].mean())
    summary["h2_train_shape_fraction"] = (
        train_split["raw"][:, 2] ** 2 / np.maximum(train_split["raw"][:, 0] ** 2, 1e-12)).tolist()
    arrays = {"pairwise_raw": pairwise["raw"], "pairwise_ra": pairwise["ra"],
              "to_source_raw": to_source["raw"], "to_source_ra": to_source["ra"],
              "session_shift": session_shift, "train_pairwise_raw": train_pairwise,
              "train_split_raw": train_split["raw"], "train_split_ra": train_split["ra"],
              "eval_split_raw": eval_split["raw"], "eval_split_ra": eval_split["ra"],
              "session_split": session_split}
    return summary, arrays, tangent, labels


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj, running = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


def compare_datasets(summary, key):
    """Kruskal-Wallis over the three datasets and Holm-adjusted pairwise MWU."""
    groups = {name: np.asarray(s[key]) for name, s in summary.items()}
    h, p = kruskal(*groups.values())
    pairs = list(combinations(groups, 2))
    raw = [float(mannwhitneyu(groups[a], groups[b], alternative="two-sided", method="exact").pvalue)
           for a, b in pairs]
    return {
        "median": {name: float(np.median(v)) for name, v in groups.items()},
        "kruskal_H": float(h), "kruskal_p": float(p),
        "pairwise": {f"{a} vs {b}": {"p": pr, "p_holm": ph}
                     for (a, b), pr, ph in zip(pairs, raw, holm(raw))},
    }


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c2")
    ax.tick_params(colors=MUTED, labelsize=8)


def save(fig, out, stem):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{stem}.{ext}", dpi=300, bbox_inches="tight")


def plot_tsne(tangent, labels, summary, out):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    for ax, key, title in zip(axes, ("raw", "ra"), ("Before RA", "After RA")):
        emb = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(tangent[key])
        for idx, s in enumerate(summary["subjects"]):
            sel = labels == s
            ax.scatter(emb[sel, 0], emb[sel, 1], s=5, color=SUBJECT_COLOURS[idx],
                       alpha=0.75, linewidths=0, label=f"S{s}")
        sil = summary[f"silhouette_by_subject_{'before' if key == 'raw' else 'after'}"]
        ax.set_title(f"{title}\nsilhouette by subject = {sil:.2f}", fontsize=9, color=INK)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#c9c8c2")
    handles, names = axes[1].get_legend_handles_labels()
    fig.legend(handles, names, loc="center right", frameon=False, fontsize=8,
               markerscale=2.5, title="Subject", title_fontsize=8)
    fig.tight_layout(rect=(0, 0, 0.9, 1))
    save(fig, out, "ra_tsne_bcic2a")
    plt.close(fig)


def plot_pairwise(arrays, summary, out):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
    vmax = arrays["pairwise_raw"].max()
    ticks = [f"S{s}" for s in summary["subjects"]]
    for ax, key, title in zip(axes, ("pairwise_raw", "pairwise_ra"), ("Before RA", "After RA")):
        im = ax.imshow(arrays[key], cmap="Blues", vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_xticks(range(len(ticks)), ticks, fontsize=7)
        ax.set_yticks(range(len(ticks)), ticks, fontsize=7)
        for spine in ax.spines.values():
            spine.set_visible(False)
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.03)
    cbar.set_label("Riemannian distance", fontsize=8, color=MUTED)
    cbar.ax.tick_params(labelsize=7, colors=MUTED)
    save(fig, out, "ra_pairwise_bcic2a")
    plt.close(fig)


def plot_bars(results, out):
    """Per-subject distances to the source domain, divided by sqrt(C)."""
    import matplotlib.pyplot as plt
    widths = [len(r[0]["subjects"]) for r in results.values()]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.8), sharey=True,
                             gridspec_kw={"width_ratios": widths})
    for ax, (name, (summary, arrays, _, _)) in zip(axes, results.items()):
        norm = np.sqrt(summary["n_channels"])
        x = np.arange(len(summary["subjects"]))
        w = 0.38
        ax.bar(x - w / 2, arrays["to_source_raw"] / norm, w * 0.92, color=BEFORE, label="Before RA")
        # After RA the source centre is the identity and the target is whitened
        # by its own earlier-session mean, so this equals its session shift.
        ax.bar(x + w / 2, arrays["to_source_ra"] / norm, w * 0.92, color=AFTER,
               label="After RA (= own session shift)")
        ax.set_xticks(x, [f"S{s}" for s in summary["subjects"]])
        ax.set_title(name, fontsize=9, color=INK)
        style(ax)
    axes[0].set_ylabel(r"Distance to source domain ($d/\sqrt{C}$)", fontsize=8, color=MUTED)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper center", ncol=2, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    save(fig, out, "ra_distance_bars")
    plt.close(fig)


def p_label(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def plot_hypotheses(summary, tests, out):
    """(a) H1 training inter-subject shift; (b) its scale/shape split;
    (c) scale and shape removed by RA on the evaluation sessions."""
    import matplotlib.pyplot as plt
    names = list(summary)
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0))

    ax = axes[0]
    for i, name in enumerate(names):
        v = np.asarray(summary[name]["h1_train_shift"])
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=16, color=BEFORE,
                   edgecolors="white", linewidths=0.5, zorder=3)
        ax.plot([i - 0.25, i + 0.25], [np.median(v)] * 2, color=INK, lw=2, zorder=4)
    ax.set_xticks(range(len(names)), names)
    ax.set_ylabel(r"Inter-subject shift ($d/\sqrt{C}$)", fontsize=8, color=MUTED)
    ax.set_title(f"(a) H1: training sessions, before RA\nKruskal-Wallis {p_label(tests['h1']['kruskal_p'])}",
                 fontsize=9, color=INK)
    style(ax)

    ax = axes[1]
    for i, name in enumerate(names):
        for offset, key, colour, label in ((-0.17, "h2_train_scale", SCALE, "Scale"),
                                           (0.17, "h2_train_shape", SHAPE, "Shape")):
            v = np.asarray(summary[name][key])
            ax.scatter(i + offset + rng.uniform(-0.06, 0.06, len(v)), v, s=14, color=colour,
                       edgecolors="white", linewidths=0.5, zorder=3, label=label if i == 0 else None)
            ax.plot([i + offset - 0.12, i + offset + 0.12], [np.median(v)] * 2, color=INK, lw=2, zorder=4)
    ax.set_xticks(range(len(names)), names)
    ax.set_ylabel("Component (per channel)", fontsize=8, color=MUTED)
    ax.set_title(f"(b) H2: split of the training shift\nshape fraction, Kruskal-Wallis "
                 f"{p_label(tests['h2_shape_fraction']['kruskal_p'])}", fontsize=9, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    style(ax)

    ax = axes[2]
    for i, name in enumerate(names):
        s = summary[name]
        for offset, part, colour in ((-0.17, "scale", SCALE), (0.17, "shape", SHAPE)):
            removed = (np.asarray(s[f"h2_eval_{part}_before"]) - np.asarray(s[f"h2_eval_{part}_after"]))
            ax.scatter(i + offset + rng.uniform(-0.06, 0.06, len(removed)), removed, s=14, color=colour,
                       edgecolors="white", linewidths=0.5, zorder=3)
            ax.plot([i + offset - 0.12, i + offset + 0.12], [np.median(removed)] * 2, color=INK, lw=2, zorder=4)
    ax.axhline(0, color="#c9c8c2", lw=0.8, zorder=1)
    ax.set_xticks(range(len(names)), names)
    ax.set_ylabel("Removed by RA (per channel)", fontsize=8, color=MUTED)
    ax.set_title("(c) Evaluation sessions: scale and\nshape removed by RA", fontsize=9, color=INK)
    style(ax)

    fig.tight_layout()
    save(fig, out, "ra_hypotheses")
    plt.close(fig)


def main():
    import matplotlib
    matplotlib.use("Agg")
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "ra_shift_outputs")
    out.mkdir(parents=True, exist_ok=True)

    results = {name: analyse(name) for name in PREPROCESSING}
    summary = {name: r[0] for name, r in results.items()}
    for s in summary.values():
        s["shape_removed_on_eval"] = (np.asarray(s["h2_eval_shape_before"])
                                      - np.asarray(s["h2_eval_shape_after"])).tolist()
        s["scale_removed_on_eval"] = (np.asarray(s["h2_eval_scale_before"])
                                      - np.asarray(s["h2_eval_scale_after"])).tolist()
    tests = {
        "h1": compare_datasets(summary, "h1_train_shift"),
        "h2_shape_fraction": compare_datasets(summary, "h2_train_shape_fraction"),
        "h2_train_shape": compare_datasets(summary, "h2_train_shape"),
        "h2_shape_removed_on_eval": compare_datasets(summary, "shape_removed_on_eval"),
    }
    (out / "ra_shift_summary.json").write_text(json.dumps(summary, indent=2))
    (out / "ra_hypothesis_tests.json").write_text(json.dumps(tests, indent=2))
    np.savez(out / "ra_shift_results.npz",
             **{f"{SLUG[n]}_{k}": v for n, r in results.items() for k, v in r[1].items()})

    s2a, a2a, t2a, l2a = results["BCI IV-2a"]
    plot_tsne(t2a, l2a, s2a, out)
    plot_pairwise(a2a, s2a, out)
    plot_bars(results, out)
    plot_hypotheses(summary, tests, out)

    print("\n=== Part 1: evaluation sessions (distances divided by sqrt(C)) ===")
    print(f"{'Dataset':10s} {'C':>3s} {'to-source before':>17s} {'after':>7s} {'reduction':>10s} "
          f"{'session shift':>14s} {'silhouette before':>18s} {'after':>7s}")
    for name, s in summary.items():
        n = np.sqrt(s["n_channels"])
        print(f"{name:10s} {s['n_channels']:3d} {np.mean(s['to_source_before']) / n:17.3f} "
              f"{np.mean(s['to_source_after']) / n:7.3f} {s['to_source_reduction_percent']:9.1f}% "
              f"{np.mean(s['session_shift']) / n:14.3f} "
              f"{s['silhouette_by_subject_before']:18.3f} {s['silhouette_by_subject_after']:7.3f}")

    print("\n=== Part 2: hypotheses (per-channel values, mean over subjects) ===")
    print(f"{'Dataset':10s} {'H1 train shift':>15s} {'(after RA)':>11s} {'scale':>7s} {'shape':>7s} "
          f"{'shape frac':>11s} {'eval scale removed':>19s} {'eval shape removed':>19s}")
    for name, s in summary.items():
        print(f"{name:10s} {np.mean(s['h1_train_shift']):15.3f} {np.mean(s['h1_train_shift_after_ra']):11.3f} "
              f"{np.mean(s['h2_train_scale']):7.3f} {np.mean(s['h2_train_shape']):7.3f} "
              f"{np.mean(s['h2_train_shape_fraction']):11.2f} "
              f"{np.mean(s['scale_removed_on_eval']):19.3f} {np.mean(s['shape_removed_on_eval']):19.3f}")
    for key, t in tests.items():
        pairs = ", ".join(f"{k}: p_Holm={v['p_holm']:.3f}" for k, v in t["pairwise"].items())
        print(f"{key}: Kruskal-Wallis p={t['kruskal_p']:.4f} | {pairs}")
    print(f"\nSaved to {out.resolve()}")


if __name__ == "__main__":
    main()
