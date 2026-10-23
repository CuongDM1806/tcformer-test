"""Mechanism analysis for subject-wise Riemannian alignment.

This module is intentionally independent of the training model.  It downloads
the public MOABB recordings, reproduces the chronological session split and
the paper's epoch preprocessing, then asks whether RA improves *conditional*
(same-class) cross-subject alignment on sessions not used to fit the reference.

The companion notebook ``ra_mechanism_evidence.ipynb`` is the user-facing
entry point.  Generated data are cached outside the repository by default.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/tcformer-ra-mpl")
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from mne_bids import read_raw_bids
from scipy.linalg import eigh
from scipy.stats import mannwhitneyu, wilcoxon
from sklearn.covariance import OAS

from moabb.datasets import Zhou2016
from moabb.datasets.bnci.bnci_2014 import (
    _load_data_001_2014,
    _load_data_004_2014,
)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    subjects: tuple[int, ...]
    labels: tuple[str, ...]
    tmin: float
    duration: float
    fit_sessions: int
    l_freq: float | None = None
    h_freq: float | None = None


SPECS = {
    "BCI IV-2a": DatasetSpec(
        "BCI IV-2a", tuple(range(1, 10)),
        ("left_hand", "right_hand", "feet", "tongue"), 2.0, 4.0, 1,
    ),
    "BCI IV-2b": DatasetSpec(
        "BCI IV-2b", tuple(range(1, 10)),
        ("left_hand", "right_hand"), 3.0, 4.0, 3,
    ),
    "Zhou2016": DatasetSpec(
        "Zhou2016", tuple(range(1, 5)),
        ("left_hand", "right_hand", "feet"), 0.5, 4.5, 2, 8.0, 30.0,
    ),
}

VARIANTS = ("none", "scalar", "diagonal", "full")
COLOURS = {"none": "#a3a29c", "scalar": "#eda100", "diagonal": "#1baf7a", "full": "#2a78d6"}


def spd_power(matrix: np.ndarray, exponent: float) -> np.ndarray:
    values, vectors = eigh(matrix)
    return (vectors * np.maximum(values, 1e-10) ** exponent) @ vectors.T


def riemann_mean(covariances: np.ndarray, tol: float = 1e-7, max_iter: int = 50) -> np.ndarray:
    """Affine-invariant mean, matching the implementation used by the model."""
    mean = np.mean(covariances, axis=0)
    for _ in range(max_iter):
        root = spd_power(mean, 0.5)
        invroot = spd_power(mean, -0.5)
        tangent = np.zeros_like(mean)
        for covariance in covariances:
            values, vectors = eigh(invroot @ covariance @ invroot)
            tangent += (vectors * np.log(np.maximum(values, 1e-10))) @ vectors.T
        tangent /= len(covariances)
        if np.linalg.norm(tangent, "fro") < tol:
            break
        values, vectors = eigh(tangent)
        mean = root @ ((vectors * np.exp(values)) @ vectors.T) @ root
        mean = (mean + mean.T) / 2
    return mean


def riemann_distance(a: np.ndarray, b: np.ndarray) -> float:
    values = eigh(b, a, eigvals_only=True)
    return float(np.linalg.norm(np.log(np.maximum(values, 1e-10))))


def distance_parts(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    logs = np.log(np.maximum(eigh(b, a, eigvals_only=True), 1e-10))
    return float(np.sqrt(np.mean(logs**2))), float(abs(logs.mean())), float(logs.std())


def covariances(x: np.ndarray) -> np.ndarray:
    return np.stack([OAS().fit(trial.T.astype(np.float64)).covariance_ for trial in x])


def whitener(reference: np.ndarray, variant: str) -> np.ndarray:
    n_channels = reference.shape[0]
    if variant == "none":
        return np.eye(n_channels)
    if variant == "scalar":
        scale = float(np.exp(np.mean(np.log(np.maximum(eigh(reference, eigvals_only=True), 1e-10)))))
        return np.eye(n_channels) / np.sqrt(scale)
    if variant == "diagonal":
        return np.diag(1.0 / np.sqrt(np.maximum(np.diag(reference), 1e-10)))
    if variant == "full":
        return spd_power(reference, -0.5)
    raise ValueError(variant)


def congruence(covs: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.einsum("ij,njk,kl->nil", w, covs, w, optimize=True)


def _natural_key(value: str) -> tuple:
    import re
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", str(value)))


def load_raw_sessions(dataset: str, subject: int, data_root: Path) -> dict[str, dict[str, mne.io.BaseRaw]]:
    """Load raw sessions directly through MOABB while keeping data under data_root."""
    if dataset == "BCI IV-2a":
        return _load_data_001_2014(subject, path=str(data_root), update_path=False, verbose=False)
    if dataset == "BCI IV-2b":
        return _load_data_004_2014(subject, path=str(data_root), update_path=False, verbose=False)

    ds = Zhou2016()
    paths = ds.bids_paths(subject, path=str(data_root), update_path=False, verbose=False)
    sessions: dict[str, dict[str, mne.io.BaseRaw]] = {}
    for path in paths:
        raw = read_raw_bids(path, extra_params=ds._get_read_extra_params(subject), verbose=False)
        session = path.session or "0"
        run = path.run or "0"
        sessions.setdefault(session, {})[run] = raw
    return sessions


def _events(raw: mne.io.BaseRaw, labels: tuple[str, ...]) -> tuple[np.ndarray, dict[str, int]]:
    stim_channels = [name for name, kind in zip(raw.ch_names, raw.get_channel_types()) if kind == "stim"]
    if stim_channels:
        events = mne.find_events(raw, stim_channel=stim_channels[0], shortest_event=1, verbose=False)
        return events, {label: idx + 1 for idx, label in enumerate(labels)}

    descriptions = set(raw.annotations.description)
    missing = set(labels) - descriptions
    if missing:
        raise RuntimeError(f"Missing annotations {sorted(missing)}; found {sorted(descriptions)}")
    event_id = {label: idx + 1 for idx, label in enumerate(labels)}
    events, _ = mne.events_from_annotations(raw, event_id=event_id, verbose=False)
    return events, event_id


def epoch_raw(raw: mne.io.BaseRaw, spec: DatasetSpec) -> tuple[np.ndarray, np.ndarray, list[str]]:
    events, event_id = _events(raw, spec.labels)
    eeg = raw.copy().pick("eeg").load_data()
    eeg.apply_function(lambda values: values * 1e6, channel_wise=False, verbose=False)
    if spec.l_freq is not None or spec.h_freq is not None:
        eeg.filter(spec.l_freq, spec.h_freq, verbose=False)
    if eeg.info["sfreq"] != 250:
        eeg, events = eeg.resample(250, events=events, verbose=False)
    sfreq = float(eeg.info["sfreq"])
    epochs = mne.Epochs(
        eeg, events, event_id=event_id, tmin=spec.tmin,
        tmax=spec.tmin + spec.duration - 1.0 / sfreq,
        baseline=None, preload=True, reject_by_annotation=False, verbose=False,
    )
    inverse = {code: idx for idx, code in enumerate(event_id.values())}
    y = np.asarray([inverse[code] for code in epochs.events[:, 2]], dtype=np.int16)
    return epochs.get_data(copy=False).astype(np.float32), y, list(epochs.ch_names)


def load_subject_epochs(dataset: str, subject: int, data_root: Path, cache_root: Path):
    """Return chronological session arrays; cache the expensive raw preprocessing."""
    spec = SPECS[dataset]
    slug = dataset.lower().replace(" ", "_").replace("-", "")
    cache_file = cache_root / f"{slug}_s{subject:02d}.npz"
    if cache_file.exists():
        saved = np.load(cache_file, allow_pickle=False)
        session_count = int(saved["session_count"])
        sessions = [(saved[f"x_{i}"], saved[f"y_{i}"]) for i in range(session_count)]
        return sessions, saved["channels"].astype(str).tolist()

    raw_sessions = load_raw_sessions(dataset, subject, data_root)
    sessions = []
    channels = None
    for session_name in sorted(raw_sessions, key=_natural_key):
        xs, ys = [], []
        for run_name in sorted(raw_sessions[session_name], key=_natural_key):
            x, y, run_channels = epoch_raw(raw_sessions[session_name][run_name], spec)
            channels = run_channels if channels is None else channels
            if run_channels != channels:
                raise RuntimeError(f"Channel mismatch in {dataset} subject {subject}")
            xs.append(x)
            ys.append(y)
        sessions.append((np.concatenate(xs), np.concatenate(ys)))

    cache_root.mkdir(parents=True, exist_ok=True)
    payload = {"session_count": np.asarray(len(sessions)), "channels": np.asarray(channels)}
    for idx, (x, y) in enumerate(sessions):
        payload[f"x_{idx}"] = x
        payload[f"y_{idx}"] = y
    np.savez_compressed(cache_file, **payload)
    return sessions, channels


def _select_trials(sessions, channels, channel_names=None, label_ids=None):
    channel_idx = np.arange(len(channels)) if channel_names is None else np.asarray([channels.index(c) for c in channel_names])
    selected = []
    for x, y in sessions:
        keep = np.ones(len(y), dtype=bool) if label_ids is None else np.isin(y, label_ids)
        x_sel = x[keep][:, channel_idx, :]
        y_sel = y[keep]
        if label_ids is not None:
            remap = {old: new for new, old in enumerate(label_ids)}
            y_sel = np.asarray([remap[int(v)] for v in y_sel], dtype=np.int16)
        selected.append((x_sel, y_sel))
    return selected


def build_configurations(data_root: Path, cache_root: Path):
    """Yield analysis configurations, including controlled reductions of 2a."""
    configurations = {
        "2a-22ch-4class": ("BCI IV-2a", None, None),
        "2a-22ch-LR": ("BCI IV-2a", None, (0, 1)),
        "2a-3ch-4class": ("BCI IV-2a", ("C3", "Cz", "C4"), None),
        "2a-3ch-LR": ("BCI IV-2a", ("C3", "Cz", "C4"), (0, 1)),
        "2b-3ch-LR": ("BCI IV-2b", None, None),
        "Zhou-14ch-3class": ("Zhou2016", None, None),
    }
    raw_cache = {}
    for config_name, (dataset, channel_names, label_ids) in configurations.items():
        spec = SPECS[dataset]
        subject_data = []
        for subject in spec.subjects:
            key = (dataset, subject)
            if key not in raw_cache:
                print(f"Loading {dataset}, subject {subject} ...", flush=True)
                raw_cache[key] = load_subject_epochs(dataset, subject, data_root, cache_root)
            sessions, channels = raw_cache[key]
            subject_data.append((subject, _select_trials(sessions, channels, channel_names, label_ids)))
        yield config_name, dataset, spec.fit_sessions, subject_data


def analyse_configuration(config_name: str, dataset: str, fit_session_count: int, subject_data):
    """Compute marginal and same-class LOSO geometry for all RA variants."""
    subject_stats = {}
    for subject, sessions in subject_data:
        fit_x = np.concatenate([x for x, _ in sessions[:fit_session_count]])
        fit_y = np.concatenate([y for _, y in sessions[:fit_session_count]])
        eval_x = np.concatenate([x for x, _ in sessions[fit_session_count:]])
        eval_y = np.concatenate([y for _, y in sessions[fit_session_count:]])
        fit_cov, eval_cov = covariances(fit_x), covariances(eval_x)
        reference = riemann_mean(fit_cov)
        n_classes = len(np.unique(fit_y))
        # The affine-invariant mean is congruence-equivariant.  Compute every
        # raw mean once, then transform the means exactly instead of repeating
        # the iterative mean for all four normalization variants.
        raw_fit = reference
        raw_eval = riemann_mean(eval_cov)
        raw_fit_class = [riemann_mean(fit_cov[fit_y == label]) for label in range(n_classes)]
        raw_eval_class = [riemann_mean(eval_cov[eval_y == label]) for label in range(n_classes)]
        stats = {"n_channels": fit_x.shape[1], "n_classes": n_classes, "variants": {}}
        for variant in VARIANTS:
            w = whitener(reference, variant)
            stats["variants"][variant] = {
                "fit": w @ raw_fit @ w,
                "eval": w @ raw_eval @ w,
                "fit_class": [w @ value @ w for value in raw_fit_class],
                "eval_class": [w @ value @ w for value in raw_eval_class],
                "eval_trials": congruence(eval_cov, w),
                "eval_y": eval_y,
            }
        subject_stats[subject] = stats

    subjects = list(subject_stats)
    rows = []
    for subject in subjects:
        n_channels = subject_stats[subject]["n_channels"]
        n_classes = subject_stats[subject]["n_classes"]
        norm = np.sqrt(n_channels)
        for variant in VARIANTS:
            own = subject_stats[subject]["variants"][variant]
            others = [subject_stats[s]["variants"][variant] for s in subjects if s != subject]
            source = riemann_mean(np.stack([item["fit"] for item in others]))
            source_classes = [riemann_mean(np.stack([item["fit_class"][label] for item in others])) for label in range(n_classes)]
            marginal = riemann_distance(source, own["eval"]) / norm
            conditional = np.mean([
                riemann_distance(source_classes[label], own["eval_class"][label]) / norm
                for label in range(n_classes)
            ])
            between = np.mean([
                riemann_distance(source_classes[a], source_classes[b]) / norm
                for a, b in combinations(range(n_classes), 2)
            ])
            predictions = np.asarray([
                np.argmin([riemann_distance(centroid, trial) for centroid in source_classes])
                for trial in own["eval_trials"]
            ])
            mdm_accuracy = float(np.mean(predictions == own["eval_y"]))
            session = riemann_distance(own["fit"], own["eval"]) / norm
            row = {
                "config": config_name, "dataset": dataset, "subject": subject,
                "variant": variant, "n_channels": n_channels, "n_classes": n_classes,
                "marginal_shift": marginal, "same_class_shift": conditional,
                "source_class_separation": between,
                "conditional_quality": between / max(conditional, 1e-12),
                "mdm_accuracy": mdm_accuracy,
                "session_shift": session,
            }
            if variant == "none":
                _, scale, shape = distance_parts(source, own["eval"])
                row.update(raw_scale=scale, raw_shape=shape,
                           raw_shape_fraction=shape**2 / max(scale**2 + shape**2, 1e-12))
            rows.append(row)
    return pd.DataFrame(rows)


def paired_p(frame: pd.DataFrame, metric: str, a: str = "none", b: str = "full") -> float:
    pivot = frame.pivot(index="subject", columns="variant", values=metric)
    try:
        return float(wilcoxon(pivot[a], pivot[b], alternative="two-sided", method="exact").pvalue)
    except ValueError:
        return 1.0


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    means = results.groupby(["config", "dataset", "variant"], as_index=False).agg(
        marginal_shift=("marginal_shift", "mean"),
        same_class_shift=("same_class_shift", "mean"),
        source_class_separation=("source_class_separation", "mean"),
        conditional_quality=("conditional_quality", "mean"),
        mdm_accuracy=("mdm_accuracy", "mean"),
        session_shift=("session_shift", "mean"),
        n_subjects=("subject", "nunique"),
        n_channels=("n_channels", "first"),
        n_classes=("n_classes", "first"),
    )
    wide = means.pivot(index=["config", "dataset", "n_subjects", "n_channels", "n_classes"], columns="variant")
    records = []
    for index, values in wide.iterrows():
        config, dataset, n_subjects, n_channels, n_classes = index
        record = dict(config=config, dataset=dataset, n_subjects=n_subjects,
                      n_channels=n_channels, n_classes=n_classes)
        for metric in ("marginal_shift", "same_class_shift", "source_class_separation", "conditional_quality", "mdm_accuracy", "session_shift"):
            for variant in VARIANTS:
                record[f"{metric}_{variant}"] = float(values[(metric, variant)])
        record["marginal_reduction_full_pct"] = 100 * (1 - record["marginal_shift_full"] / record["marginal_shift_none"])
        record["same_class_reduction_full_pct"] = 100 * (1 - record["same_class_shift_full"] / record["same_class_shift_none"])
        record["quality_change_full_pct"] = 100 * (record["conditional_quality_full"] / record["conditional_quality_none"] - 1)
        record["mdm_gain_full_pp"] = 100 * (record["mdm_accuracy_full"] - record["mdm_accuracy_none"])
        subset = results[results.config == config]
        record["p_same_class_none_vs_full"] = paired_p(subset, "same_class_shift")
        record["p_quality_none_vs_full"] = paired_p(subset, "conditional_quality")
        record["mdm_full_minus_scalar_pp"] = 100 * (record["mdm_accuracy_full"] - record["mdm_accuracy_scalar"])
        record["mdm_full_minus_diagonal_pp"] = 100 * (record["mdm_accuracy_full"] - record["mdm_accuracy_diagonal"])
        record["p_mdm_none_vs_full"] = paired_p(subset, "mdm_accuracy")
        record["p_mdm_scalar_vs_full"] = paired_p(subset, "mdm_accuracy", "scalar", "full")
        record["p_mdm_diagonal_vs_full"] = paired_p(subset, "mdm_accuracy", "diagonal", "full")
        records.append(record)
    return pd.DataFrame(records).sort_values("config").reset_index(drop=True)


def controlled_tests(results: pd.DataFrame) -> dict:
    """Paired subject tests isolating channel count and class count inside 2a."""
    full = results[results.variant == "full"].set_index(["config", "subject"])
    raw = results[results.variant == "none"].set_index(["config", "subject"])
    reduction = 100 * (1 - full.same_class_shift / raw.same_class_shift)
    mdm_gain = 100 * (full.mdm_accuracy - raw.mdm_accuracy)

    def compare(config_a, config_b):
        a = reduction.loc[config_a]
        b = reduction.loc[config_b]
        return {
            "config_a": config_a, "config_b": config_b,
            "mean_reduction_a_pct": float(a.mean()),
            "mean_reduction_b_pct": float(b.mean()),
            "mean_paired_difference_pct": float((a - b).mean()),
            "wilcoxon_p": float(wilcoxon(a, b, alternative="two-sided", method="exact").pvalue),
        }

    def compare_mdm(config_a, config_b):
        a = mdm_gain.loc[config_a]
        b = mdm_gain.loc[config_b]
        return {
            "config_a": config_a, "config_b": config_b,
            "mean_mdm_gain_a_pp": float(a.mean()),
            "mean_mdm_gain_b_pp": float(b.mean()),
            "mean_paired_difference_pp": float((a - b).mean()),
            "wilcoxon_p": float(wilcoxon(a, b, alternative="two-sided", method="auto").pvalue),
        }

    def compare_mdm_independent(config_a, config_b):
        a = mdm_gain.loc[config_a]
        b = mdm_gain.loc[config_b]
        return {
            "config_a": config_a, "config_b": config_b,
            "mean_mdm_gain_a_pp": float(a.mean()),
            "mean_mdm_gain_b_pp": float(b.mean()),
            "mean_difference_pp": float(a.mean() - b.mean()),
            "mann_whitney_p": float(mannwhitneyu(a, b, alternative="two-sided", method="exact").pvalue),
        }

    return {
        "channel_effect_4class": compare("2a-22ch-4class", "2a-3ch-4class"),
        "channel_effect_LR": compare("2a-22ch-LR", "2a-3ch-LR"),
        "class_effect_22ch": compare("2a-22ch-4class", "2a-22ch-LR"),
        "class_effect_3ch": compare("2a-3ch-4class", "2a-3ch-LR"),
        "mdm_channel_effect_4class": compare_mdm("2a-22ch-4class", "2a-3ch-4class"),
        "mdm_channel_effect_LR": compare_mdm("2a-22ch-LR", "2a-3ch-LR"),
        "mdm_2a_vs_2b_matched_design": compare_mdm_independent("2a-3ch-LR", "2b-3ch-LR"),
    }


def plot_summary(summary: pd.DataFrame, output_dir: Path) -> None:
    order = ["2a-22ch-4class", "2a-3ch-4class", "2a-22ch-LR", "2a-3ch-LR", "2b-3ch-LR", "Zhou-14ch-3class"]
    plot_data = summary.set_index("config").loc[order]
    x = np.arange(len(order))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    width = 0.19
    for idx, variant in enumerate(VARIANTS):
        axes[0].bar(x + (idx - 1.5) * width, plot_data[f"same_class_shift_{variant}"], width,
                    label=variant.capitalize(), color=COLOURS[variant])
    axes[0].set_ylabel(r"Same-class target-to-source distance ($d/\sqrt{C}$)")
    axes[0].set_title("Lower is better: class-conditional alignment")
    axes[0].set_xticks(x, order, rotation=28, ha="right")
    axes[0].legend(frameon=False, ncol=2)

    axes[1].bar(x, plot_data["mdm_gain_full_pp"], color="#2a78d6")
    axes[1].axhline(0, color="#777777", lw=0.8)
    axes[1].set_ylabel("MDM accuracy gain (percentage points)")
    axes[1].set_title("Does alignment transfer class information?")
    axes[1].set_xticks(x, order, rotation=28, ha="right")

    axes[2].bar(x, plot_data["mdm_full_minus_scalar_pp"],
                color=["#2a78d6" if value >= 0 else "#eb6834"
                       for value in plot_data["mdm_full_minus_scalar_pp"]])
    axes[2].axhline(0, color="#777777", lw=0.8)
    axes[2].set_ylabel("Full RA minus scalar normalization (pp)")
    axes[2].set_title("Spatial-whitening bonus")
    axes[2].set_xticks(x, order, rotation=28, ha="right")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(output_dir / "ra_mechanism_summary.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / "ra_mechanism_summary.pdf", bbox_inches="tight")
    plt.close(fig)


def run_analysis(data_root="/tmp/tcformer-ra-data", cache_root="/tmp/tcformer-ra-epochs",
                 output_dir="analysis/ra_mechanism_outputs"):
    data_root, cache_root, output_dir = map(Path, (data_root, cache_root, output_dir))
    for path in (data_root, cache_root, output_dir):
        path.mkdir(parents=True, exist_ok=True)
    frames = []
    for config_name, dataset, fit_sessions, subject_data in build_configurations(data_root, cache_root):
        print(f"Analysing {config_name} ...", flush=True)
        frames.append(analyse_configuration(config_name, dataset, fit_sessions, subject_data))
    results = pd.concat(frames, ignore_index=True)
    summary = summarise(results)
    tests = controlled_tests(results)
    results.to_csv(output_dir / "ra_mechanism_subject_results.csv", index=False)
    summary.to_csv(output_dir / "ra_mechanism_summary.csv", index=False)
    (output_dir / "ra_mechanism_controlled_tests.json").write_text(json.dumps(tests, indent=2))
    plot_summary(summary, output_dir)
    print(f"Saved results to {output_dir.resolve()}")
    return results, summary, tests


if __name__ == "__main__":
    run_analysis()
