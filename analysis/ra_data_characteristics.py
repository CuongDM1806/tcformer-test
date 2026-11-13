"""Claim 1, item 4: measure the data characteristics that could make RA help BCI IV-2a most.

Everything is measured on the signals RA receives (pipeline preprocessing, before RA),
per subject, for BCI IV-2a, BCI IV-2b, Zhou2016 and OpenBMI. No model is trained.

H1  inter-channel correlation / common reference component
    mean |r| between channels; share of signal energy in the common (average) mode,
    i.e. the part that CAR would remove.
H2  out-of-band power (no band-pass on 2a/2b)
    share of 0.5-45 Hz power below 4 Hz, 4-8 Hz, 8-30 Hz and 30-45 Hz; between-subject
    spread of these shares. Zhou2016 is measured twice: pipeline (8-30 Hz) and unfiltered.
H3  quality of the RA reference
    split-half distance of the reference (fit sessions), its condition number, trials per
    channel, and the distance between the fit-session and evaluation-session references.
H4  kind of inter-subject variability
    tangent-space (log-Euclidean) sums of squares split into subject, class,
    subject x class and residual parts, before and after RA. RA removes the subject part;
    it cannot remove the subject x class part.

Used by ra_data_characteristics.ipynb (Kaggle, CPU only), which embeds this file.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.integrate import trapezoid
from scipy.signal import welch
from sklearn.covariance import OAS

BANDS = {"low_0.5-4": (0.5, 4.0), "theta_4-8": (4.0, 8.0),
         "mubeta_8-30": (8.0, 30.0), "high_30-45": (30.0, 45.0)}
TOTAL_BAND = (0.5, 45.0)          # common to every dataset (OpenBMI is sampled at 100 Hz)
SPLIT_HALF_REPEATS = 10


# ---------------------------------------------------------------- SPD helpers
def spd_power(matrix: np.ndarray, exponent: float) -> np.ndarray:
    values, vectors = eigh(matrix)
    return (vectors * np.maximum(values, 1e-12) ** exponent) @ vectors.T


def spd_log(matrix: np.ndarray) -> np.ndarray:
    values, vectors = eigh(matrix)
    return (vectors * np.log(np.maximum(values, 1e-12))) @ vectors.T


def riemann_mean(covs: np.ndarray, tol: float = 1e-7, max_iter: int = 50) -> np.ndarray:
    """Affine-invariant Riemannian mean (same iteration as the RA used by the model)."""
    mean = covs.mean(axis=0)
    for _ in range(max_iter):
        root, invroot = spd_power(mean, 0.5), spd_power(mean, -0.5)
        tangent = np.mean([spd_log(invroot @ c @ invroot) for c in covs], axis=0)
        if np.linalg.norm(tangent, "fro") < tol:
            break
        values, vectors = eigh(tangent)
        mean = root @ ((vectors * np.exp(values)) @ vectors.T) @ root
        mean = (mean + mean.T) / 2
    return mean


def riemann_distance(a: np.ndarray, b: np.ndarray) -> float:
    values = eigh(b, a, eigvals_only=True)
    return float(np.linalg.norm(np.log(np.maximum(values, 1e-12))))


def oas_covariances(x: np.ndarray) -> np.ndarray:
    return np.stack([OAS().fit(trial.T.astype(np.float64)).covariance_ for trial in x])


def tangent_vectors(covs: np.ndarray) -> np.ndarray:
    """Log-Euclidean vectors: upper triangle of log(C), off-diagonal scaled by sqrt(2)."""
    n = covs.shape[1]
    rows, cols = np.triu_indices(n)
    weight = np.where(rows == cols, 1.0, np.sqrt(2.0))
    return np.stack([spd_log(c)[rows, cols] * weight for c in covs])


# ---------------------------------------------------------------- data container
@dataclass
class SubjectData:
    subject: int
    sessions: list            # [(x [trials, channels, samples] float32, y int), ...] chronological
    fit_sessions: int         # leading sessions used to fit the frozen RA reference


@dataclass
class View:
    name: str                 # e.g. "BCI IV-2a", "Zhou2016 (unfiltered)"
    sfreq: float
    preprocessing: str        # human-readable description
    subjects: list            # [SubjectData]


# ---------------------------------------------------------------- per-subject metrics
def h1_correlation(x: np.ndarray) -> dict:
    """Mean |r| between channels and the common-mode energy share, averaged over trials."""
    xc = x - x.mean(axis=-1, keepdims=True)
    cov = np.einsum("nct,ndt->ncd", xc, xc, optimize=True) / x.shape[-1]
    std = np.sqrt(np.einsum("ncc->nc", cov))
    corr = cov / (std[:, :, None] * std[:, None, :])
    n_ch = x.shape[1]
    off = ~np.eye(n_ch, dtype=bool)
    mean_abs_corr = float(np.abs(corr[:, off]).mean())
    # Energy of the channel average, as a share of total energy: what CAR would remove.
    common = cov.sum(axis=(1, 2)) / n_ch / np.einsum("ncc->n", cov)
    common_share = float(common.mean())
    return {
        "h1_mean_abs_corr": mean_abs_corr,
        "h1_common_mode_share": common_share,
        # 0 = independent equal-variance channels, 1 = identical channels.
        "h1_common_mode_excess": float((common_share - 1 / n_ch) / (1 - 1 / n_ch)),
    }


def mean_psd(x: np.ndarray, sfreq: float) -> tuple[np.ndarray, np.ndarray]:
    nperseg = int(min(x.shape[-1], 2 * sfreq))
    freqs, psd = welch(x.astype(np.float64), fs=sfreq, nperseg=nperseg, axis=-1)
    return freqs, psd.mean(axis=(0, 1))


def h2_band_shares(freqs: np.ndarray, psd: np.ndarray) -> dict:
    def power(lo, hi):
        keep = (freqs >= lo) & (freqs < hi)
        return float(trapezoid(psd[keep], freqs[keep])) if keep.sum() > 1 else 0.0

    total = power(*TOTAL_BAND)
    shares = {f"h2_share_{name}": power(lo, hi) / total for name, (lo, hi) in BANDS.items()}
    shares["h2_share_outside_8-30"] = 1.0 - shares["h2_share_mubeta_8-30"]
    return shares


def h3_reference(subject: SubjectData, rng: np.random.Generator) -> tuple[dict, list]:
    fit_x = np.concatenate([x for x, _ in subject.sessions[:subject.fit_sessions]])
    eval_x = np.concatenate([x for x, _ in subject.sessions[subject.fit_sessions:]])
    n_ch = fit_x.shape[1]
    fit_covs = oas_covariances(fit_x)
    eval_covs = oas_covariances(eval_x)
    reference = riemann_mean(fit_covs)
    halves = []
    for _ in range(SPLIT_HALF_REPEATS):
        order = rng.permutation(len(fit_covs))
        a, b = order[: len(order) // 2], order[len(order) // 2:]
        halves.append(riemann_distance(riemann_mean(fit_covs[a]), riemann_mean(fit_covs[b])))
    values = eigh(reference, eigvals_only=True)
    metrics = {
        "h3_fit_trials": int(len(fit_covs)),
        "h3_trials_per_channel": float(len(fit_covs) / n_ch),
        "h3_split_half_distance": float(np.mean(halves) / np.sqrt(n_ch)),
        "h3_reference_log10_condition": float(np.log10(values.max() / max(values.min(), 1e-12))),
        "h3_session_shift": float(riemann_distance(reference, riemann_mean(eval_covs)) / np.sqrt(n_ch)),
    }
    return metrics, [fit_covs, eval_covs, reference]


# ---------------------------------------------------------------- H4: variance decomposition
def variance_decomposition(vectors: np.ndarray, subjects: np.ndarray, labels: np.ndarray) -> tuple[dict, pd.DataFrame]:
    """Sums of squares of tangent vectors split into subject, class, subject x class, residual."""
    grand = vectors.mean(axis=0)
    total = float(((vectors - grand) ** 2).sum())

    def between(groups):
        ss = 0.0
        for g in np.unique(groups, axis=0):
            mask = np.all(groups == g, axis=1) if groups.ndim == 2 else groups == g
            ss += mask.sum() * float(((vectors[mask].mean(axis=0) - grand) ** 2).sum())
        return ss

    ss_subject = between(subjects)
    ss_class = between(labels)
    ss_cell = between(np.stack([subjects, labels], axis=1))
    ss_interaction = max(ss_cell - ss_subject - ss_class, 0.0)
    summary = {
        "eta2_subject": ss_subject / total,
        "eta2_class": ss_class / total,
        "eta2_subject_x_class": ss_interaction / total,
        "eta2_residual": (total - ss_cell) / total,
    }
    summary["subject_over_interaction"] = summary["eta2_subject"] / max(summary["eta2_subject_x_class"], 1e-12)
    summary["class_over_interaction"] = summary["eta2_class"] / max(summary["eta2_subject_x_class"], 1e-12)

    # Per subject: offset of its mean (removable by RA) vs its class-specific deviation (not removable).
    p = vectors.shape[1]
    class_means = {k: vectors[labels == k].mean(axis=0) for k in np.unique(labels)}
    rows = []
    for s in np.unique(subjects):
        ms = subjects == s
        subject_mean = vectors[ms].mean(axis=0)
        offset = float(np.linalg.norm(subject_mean - grand) / np.sqrt(p))
        dev = [np.sum((vectors[ms & (labels == k)].mean(axis=0) - subject_mean - class_means[k] + grand) ** 2)
               for k in class_means if np.any(ms & (labels == k))]
        interaction = float(np.sqrt(np.mean(dev) / p))
        rows.append({"subject": int(s), "h4_subject_offset": offset,
                     "h4_subject_x_class": interaction,
                     "h4_offset_over_interaction": offset / max(interaction, 1e-12)})
    return summary, pd.DataFrame(rows)


# ---------------------------------------------------------------- one view
def analyse_view(view: View, seed: int = 0, log=print) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows, psd_rows = [], []
    vec_raw, vec_shape, vec_ra, subj_ids, labels = [], [], [], [], []
    for sd in view.subjects:
        x_all = np.concatenate([x for x, _ in sd.sessions])
        y_all = np.concatenate([y for _, y in sd.sessions])
        row = {"view": view.name, "subject": sd.subject, "n_channels": x_all.shape[1],
               "n_classes": int(len(np.unique(y_all))), "n_trials": int(len(y_all))}
        row.update(h1_correlation(x_all))
        freqs, psd = mean_psd(x_all, view.sfreq)
        row.update(h2_band_shares(freqs, psd))
        keep = (freqs >= TOTAL_BAND[0]) & (freqs <= TOTAL_BAND[1])
        total = trapezoid(psd[keep], freqs[keep])
        psd_rows += [{"view": view.name, "subject": sd.subject, "freq": float(f), "relative_psd": float(v / total)}
                     for f, v in zip(freqs[keep], psd[keep])]
        h3, (fit_covs, eval_covs, reference) = h3_reference(sd, rng)
        row.update(h3)
        rows.append(row)

        covs = np.concatenate([fit_covs, eval_covs])
        whiten = spd_power(reference, -0.5)
        traces = np.einsum("ncc->n", covs)
        vec_raw.append(tangent_vectors(covs))
        vec_shape.append(tangent_vectors(covs / traces[:, None, None] * covs.shape[1]))
        vec_ra.append(tangent_vectors(np.einsum("ij,njk,kl->nil", whiten, covs, whiten, optimize=True)))
        fit_y = np.concatenate([y for _, y in sd.sessions[:sd.fit_sessions]])
        eval_y = np.concatenate([y for _, y in sd.sessions[sd.fit_sessions:]])
        labels.append(np.concatenate([fit_y, eval_y]))
        subj_ids.append(np.full(len(covs), sd.subject))
        log(f"  {view.name} S{sd.subject:02d}: corr={row['h1_mean_abs_corr']:.2f} "
            f"outside8-30={row['h2_share_outside_8-30']:.2f} split-half={row['h3_split_half_distance']:.3f}")

    subjects = np.concatenate(subj_ids)
    labels = np.concatenate(labels)
    decomposition, per_subject = [], None
    for kind, vectors in (("before RA", vec_raw), ("before RA, trace-normalized", vec_shape),
                          ("after frozen RA", vec_ra)):
        summary, table = variance_decomposition(np.concatenate(vectors), subjects, labels)
        decomposition.append({"view": view.name, "covariance": kind, **summary})
        if kind == "before RA":
            per_subject = table
    metrics = pd.DataFrame(rows).merge(per_subject, on="subject")
    return metrics, pd.DataFrame(decomposition), pd.DataFrame(psd_rows)


# ---------------------------------------------------------------- summaries
SUMMARY_METRICS = [
    "h1_mean_abs_corr", "h1_common_mode_share", "h1_common_mode_excess",
    "h2_share_low_0.5-4", "h2_share_theta_4-8", "h2_share_mubeta_8-30", "h2_share_high_30-45",
    "h2_share_outside_8-30",
    "h3_trials_per_channel", "h3_split_half_distance", "h3_reference_log10_condition", "h3_session_shift",
    "h4_subject_offset", "h4_subject_x_class", "h4_offset_over_interaction",
]


def summarise(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for view, group in metrics.groupby("view", sort=False):
        row = {"view": view, "n_subjects": len(group), "n_channels": int(group["n_channels"].iloc[0])}
        for metric in SUMMARY_METRICS:
            q1, med, q3 = group[metric].quantile([0.25, 0.5, 0.75])
            row[metric] = f"{med:.3f} [{q1:.3f}-{q3:.3f}]"
        # Between-subject spread of the H1/H2 shares (coefficient of variation).
        for metric in ("h1_common_mode_share", "h2_share_outside_8-30", "h2_share_low_0.5-4"):
            values = group[metric]
            row[f"{metric}_between_subject_cv"] = f"{values.std(ddof=0) / values.mean():.3f}"
        rows.append(row)
    return pd.DataFrame(rows).set_index("view").T


def natural_key(value) -> tuple:
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", str(value)))


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=float))
