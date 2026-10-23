"""Input-level subject shift before and after Riemannian alignment (RA).

Run from the root of the code branch (feature/hada-full-mamba-autodial), so
that the paper's own loaders, session splits, and RA functions are used:

    python ra_shift_analysis.py <output_dir>

For every subject, RA is fitted exactly as in training: the reference R is the
Riemannian mean of the OAS trial covariances of the subject's *earlier*
sessions (BCI IV-2a: session 1; BCI IV-2b: sessions 1-3; Zhou2016: sessions
1-2) and is then applied unchanged to the *evaluation* sessions. On the fitting
sessions every subject mean becomes the identity by construction, so all shift
measures are computed on the evaluation sessions, which RA never saw.

Outputs (in <output_dir>): ra_shift_summary.json, ra_shift_results.npz and the
figures ra_tsne_bcic2a, ra_pairwise_bcic2a, ra_distance_bars (.png and .pdf).
No model is trained.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
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


def riemann_distance(A, B):
    """Affine-invariant distance: sqrt(sum log^2 of the eigenvalues of A^-1 B)."""
    w = eigh(B, A, eigvals_only=True)
    return float(np.sqrt(np.sum(np.log(np.maximum(w, 1e-12)) ** 2)))


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
        # Exactly the transform used in training (reference = fitting sessions).
        A_fit, A_eval = BaseDataModule._riemannian_align_many(X_fit, X_eval)
        C = {"fit_raw": covariances(X_fit), "eval_raw": covariances(X_eval),
             "fit_ra": covariances(A_fit), "eval_ra": covariances(A_eval)}
        for key in M:
            M[key].append(mean(C[key]))
        eval_covs["raw"].append(C["eval_raw"])
        eval_covs["ra"].append(C["eval_ra"])
        labels += [s] * len(X_eval)
        print(f"  subject {s}: {len(X_fit)} fitting / {len(X_eval)} evaluation trials", flush=True)

    k = len(subjects)
    pairwise = {v: np.array([[riemann_distance(M[f"eval_{v}"][i], M[f"eval_{v}"][j])
                              for j in range(k)] for i in range(k)]) for v in ("raw", "ra")}
    # Distance from the target's evaluation session to the centre of the
    # source domain (Riemannian mean of the other subjects' fitting sessions).
    to_source = {
        v: np.array([riemann_distance(
            M[f"eval_{v}"][i], mean(np.stack([M[f"fit_{v}"][j] for j in range(k) if j != i])))
            for i in range(k)])
        for v in ("raw", "ra")
    }
    # Within-subject session shift (fitting vs evaluation sessions). RA applies
    # one congruence per subject, which leaves this distance unchanged.
    session_shift = np.array([riemann_distance(M["fit_raw"][i], M["eval_raw"][i]) for i in range(k)])

    labels = np.array(labels)
    raw_all = np.concatenate(eval_covs["raw"])
    ra_all = np.concatenate(eval_covs["ra"])
    tangent = {
        "raw": tangent_vectors(raw_all, mean(np.stack(M["eval_raw"]))),
        "ra": tangent_vectors(ra_all, np.eye(raw_all.shape[1])),
    }
    silhouette = {v: float(silhouette_score(tangent[v], labels)) for v in tangent}

    off = ~np.eye(k, dtype=bool)
    summary = {
        "subjects": subjects,
        "pairwise_mean_before": float(pairwise["raw"][off].mean()),
        "pairwise_mean_after": float(pairwise["ra"][off].mean()),
        "to_source_before": to_source["raw"].tolist(),
        "to_source_after": to_source["ra"].tolist(),
        "session_shift": session_shift.tolist(),
        "silhouette_by_subject_before": silhouette["raw"],
        "silhouette_by_subject_after": silhouette["ra"],
    }
    summary["pairwise_reduction_percent"] = 100 * (
        1 - summary["pairwise_mean_after"] / summary["pairwise_mean_before"])
    arrays = {"pairwise_raw": pairwise["raw"], "pairwise_ra": pairwise["ra"],
              "to_source_raw": to_source["raw"], "to_source_ra": to_source["ra"],
              "session_shift": session_shift}
    return summary, arrays, tangent, labels


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
    import matplotlib.pyplot as plt
    widths = [len(r[0]["subjects"]) for r in results.values()]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.8), sharey=True,
                             gridspec_kw={"width_ratios": widths})
    for ax, (name, (summary, arrays, _, _)) in zip(axes, results.items()):
        x = np.arange(len(summary["subjects"]))
        w = 0.27
        ax.bar(x - w, arrays["to_source_raw"], w * 0.92, color=BEFORE, label="Before RA")
        ax.bar(x, arrays["to_source_ra"], w * 0.92, color=AFTER, label="After RA")
        ax.bar(x + w, arrays["session_shift"], w * 0.92, color=SESSION,
               label="Own session shift")
        ax.set_xticks(x, [f"S{s}" for s in summary["subjects"]])
        ax.set_title(name, fontsize=9, color=INK)
        style(ax)
    axes[0].set_ylabel("Distance to source domain", fontsize=8, color=MUTED)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper center", ncol=3, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    save(fig, out, "ra_distance_bars")
    plt.close(fig)


def main():
    import matplotlib
    matplotlib.use("Agg")
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "ra_shift_outputs")
    out.mkdir(parents=True, exist_ok=True)

    results = {name: analyse(name) for name in PREPROCESSING}
    summary = {name: r[0] for name, r in results.items()}
    (out / "ra_shift_summary.json").write_text(json.dumps(summary, indent=2))
    np.savez(out / "ra_shift_results.npz",
             **{f"{SLUG[n]}_{k}": v for n, r in results.items() for k, v in r[1].items()})

    s2a, a2a, t2a, l2a = results["BCI IV-2a"]
    plot_tsne(t2a, l2a, s2a, out)
    plot_pairwise(a2a, s2a, out)
    plot_bars(results, out)

    print("\n=== Summary (evaluation sessions only) ===")
    print(f"{'Dataset':10s} {'pairwise before':>16s} {'after':>7s} {'reduction':>10s} "
          f"{'to-source before':>17s} {'after':>7s} {'session shift':>14s} {'silhouette before':>18s} {'after':>7s}")
    for name, s in summary.items():
        print(f"{name:10s} {s['pairwise_mean_before']:16.3f} {s['pairwise_mean_after']:7.3f} "
              f"{s['pairwise_reduction_percent']:9.1f}% "
              f"{np.mean(s['to_source_before']):17.3f} {np.mean(s['to_source_after']):7.3f} "
              f"{np.mean(s['session_shift']):14.3f} "
              f"{s['silhouette_by_subject_before']:18.3f} {s['silhouette_by_subject_after']:7.3f}")
    print(f"\nSaved to {out.resolve()}")


if __name__ == "__main__":
    main()
