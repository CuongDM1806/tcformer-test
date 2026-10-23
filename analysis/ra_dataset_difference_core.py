"""Targeted tests of why Riemannian alignment behaves differently on 2a/2b.

The analysis deliberately separates three questions:

1. Does the heterogeneous 2b reference (screening sessions 1--2 plus feedback
   session 3) weaken RA relative to a feedback-matched session-3 reference?
2. Is the RA classification gain larger on 2a after matching both the number
   of channels and the left/right task?
3. Does RA make the left/right class-contrast direction more consistent across
   subjects on 2a than on 2b?

All target labels are used only for retrospective measurement.  No neural
network is trained.  Trial covariances use OAS, and classification uses LOSO
Riemannian minimum-distance-to-class-mean (MDM).
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.stats import wilcoxon

from ra_mechanism_core import (
    congruence,
    covariances,
    distance_parts,
    riemann_distance,
    riemann_mean,
    whitener,
)


CACHE_ROOT = Path("/tmp/tcformer-ra-epochs")
VARIANTS = ("none", "scalar", "full")


CONFIGS = {
    "2a-22ch-LR": {
        "dataset": "bci_iv2a",
        "subjects": range(1, 10),
        "channels": None,
        "labels": (0, 1),
        "strategies": {
            "standard": {"reference": (0,), "centroids": (0,), "evaluation": (1,)},
        },
    },
    "2a-3ch-LR": {
        "dataset": "bci_iv2a",
        "subjects": range(1, 10),
        "channels": ("C3", "Cz", "C4"),
        "labels": (0, 1),
        "strategies": {
            "standard": {"reference": (0,), "centroids": (0,), "evaluation": (1,)},
        },
    },
    "2b-3ch-LR": {
        "dataset": "bci_iv2b",
        "subjects": range(1, 10),
        "channels": None,
        "labels": (0, 1),
        "strategies": {
            # Current pipeline: screening sessions 1--2 plus feedback session 3.
            "pooled-S1S2S3": {"reference": (0, 1, 2), "centroids": (0, 1, 2), "evaluation": (3, 4)},
            # Causal isolation: change only the RA reference; keep all labeled
            # source trials used to form class centroids exactly the same.
            "feedback-S3-reference": {"reference": (2,), "centroids": (0, 1, 2), "evaluation": (3, 4)},
            # Practical matched-regime diagnostic: both reference and source
            # class centroids use the feedback session closest to evaluation.
            "feedback-S3-only": {"reference": (2,), "centroids": (2,), "evaluation": (3, 4)},
            "screening-S1S2-only": {"reference": (0, 1), "centroids": (0, 1), "evaluation": (3, 4)},
        },
    },
}


def matrix_log(matrix: np.ndarray) -> np.ndarray:
    values, vectors = eigh(matrix)
    return (vectors * np.log(np.maximum(values, 1e-10))) @ vectors.T


def sym_vector(matrix: np.ndarray) -> np.ndarray:
    """Frobenius-isometric vectorization of a symmetric matrix."""
    diagonal = np.diag(matrix)
    upper = matrix[np.triu_indices_from(matrix, k=1)] * np.sqrt(2.0)
    return np.concatenate([diagonal, upper])


def contrast_vector(class_means: list[np.ndarray]) -> np.ndarray:
    vector = sym_vector(matrix_log(class_means[0]) - matrix_log(class_means[1]))
    return vector / max(np.linalg.norm(vector), 1e-12)


def concatenate_sessions(session_data, indices):
    x = np.concatenate([session_data[index][0] for index in indices])
    y = np.concatenate([session_data[index][1] for index in indices])
    return x, y


def load_cached_subject(dataset: str, subject: int, channels, labels):
    path = CACHE_ROOT / f"{dataset}_s{subject:02d}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing epoch cache: {path}. Run ra_mechanism_evidence.ipynb first.")
    saved = np.load(path, allow_pickle=False)
    all_channels = saved["channels"].astype(str).tolist()
    channel_idx = np.arange(len(all_channels)) if channels is None else np.asarray(
        [all_channels.index(channel) for channel in channels]
    )
    remap = {old: new for new, old in enumerate(labels)}
    sessions = []
    for index in range(int(saved["session_count"])):
        x, y = saved[f"x_{index}"], saved[f"y_{index}"]
        keep = np.isin(y, labels)
        selected_x = x[keep][:, channel_idx, :]
        selected_y = np.asarray([remap[int(value)] for value in y[keep]], dtype=np.int16)
        sessions.append((selected_x, selected_y))
    return sessions


def prepare_subject(session_data, strategy):
    reference_x, _ = concatenate_sessions(session_data, strategy["reference"])
    centroid_x, centroid_y = concatenate_sessions(session_data, strategy["centroids"])
    evaluation_x, evaluation_y = concatenate_sessions(session_data, strategy["evaluation"])

    reference_cov = covariances(reference_x)
    centroid_cov = covariances(centroid_x)
    evaluation_cov = covariances(evaluation_x)
    reference_mean = riemann_mean(reference_cov)
    evaluation_mean = riemann_mean(evaluation_cov)
    _, scale_shift, shape_shift = distance_parts(reference_mean, evaluation_mean)

    variants = {}
    for variant in VARIANTS:
        w = whitener(reference_mean, variant)
        transformed_centroids = congruence(centroid_cov, w)
        class_means = [
            riemann_mean(transformed_centroids[centroid_y == label]) for label in (0, 1)
        ]
        transformed_evaluation = congruence(evaluation_cov, w)
        evaluation_class_means = [
            riemann_mean(transformed_evaluation[evaluation_y == label]) for label in (0, 1)
        ]
        variants[variant] = {
            "class_means": class_means,
            "contrast": contrast_vector(class_means),
            "evaluation_cov": transformed_evaluation,
            "evaluation_y": evaluation_y,
            "evaluation_class_means": evaluation_class_means,
        }
    return variants, scale_shift, shape_shift


def mean_pairwise_cosine(vectors: list[np.ndarray]) -> float:
    return float(np.mean([np.dot(vectors[a], vectors[b]) for a, b in combinations(range(len(vectors)), 2)]))


def analyse_configuration(config_name: str, spec: dict):
    session_cache = {
        subject: load_cached_subject(spec["dataset"], subject, spec["channels"], spec["labels"])
        for subject in spec["subjects"]
    }
    result_rows, contrast_rows, geometry_rows = [], [], []
    contrast_vectors = {}
    for strategy_name, strategy in spec["strategies"].items():
        prepared = {}
        for subject, session_data in session_cache.items():
            variants, scale_shift, shape_shift = prepare_subject(session_data, strategy)
            prepared[subject] = variants
            geometry_rows.append({
                "config": config_name,
                "strategy": strategy_name,
                "subject": subject,
                "reference_eval_scale_shift": scale_shift,
                "reference_eval_shape_shift": shape_shift,
            })

        for variant in VARIANTS:
            contrast_vectors[(strategy_name, variant)] = [
                prepared[subject][variant]["contrast"] for subject in spec["subjects"]
            ]
            contrast_rows.append({
                "config": config_name,
                "strategy": strategy_name,
                "variant": variant,
                "mean_pairwise_contrast_cosine": mean_pairwise_cosine(
                    contrast_vectors[(strategy_name, variant)]
                ),
            })
            for target in spec["subjects"]:
                source_subjects = [subject for subject in spec["subjects"] if subject != target]
                source_classes = [
                    riemann_mean(np.stack([
                        prepared[subject][variant]["class_means"][label]
                        for subject in source_subjects
                    ]))
                    for label in (0, 1)
                ]
                target_data = prepared[target][variant]
                predictions = np.asarray([
                    np.argmin([riemann_distance(class_mean, trial) for class_mean in source_classes])
                    for trial in target_data["evaluation_cov"]
                ])
                same_class_shift = np.mean([
                    riemann_distance(source_classes[label], target_data["evaluation_class_means"][label])
                    / np.sqrt(target_data["evaluation_cov"].shape[1])
                    for label in (0, 1)
                ])
                result_rows.append({
                    "config": config_name,
                    "strategy": strategy_name,
                    "subject": target,
                    "variant": variant,
                    "mdm_accuracy": float(np.mean(predictions == target_data["evaluation_y"])),
                    "same_class_shift": float(same_class_shift),
                })
    return (
        pd.DataFrame(result_rows),
        pd.DataFrame(contrast_rows),
        pd.DataFrame(geometry_rows),
        contrast_vectors,
    )


def exact_permutation_mean_greater(a: np.ndarray, b: np.ndarray) -> float:
    """Exact one-sided independent permutation test: mean(a) > mean(b)."""
    combined = np.concatenate([a, b])
    n_a = len(a)
    observed = float(np.mean(a) - np.mean(b))
    exceed, total = 0, 0
    all_indices = set(range(len(combined)))
    for selected_tuple in combinations(range(len(combined)), n_a):
        selected = set(selected_tuple)
        other = list(all_indices - selected)
        statistic = float(np.mean(combined[list(selected)]) - np.mean(combined[other]))
        exceed += statistic >= observed - 1e-12
        total += 1
    return exceed / total


def contrast_interaction_permutation(a_none, a_full, b_none, b_full) -> tuple[float, float, float]:
    """Subject-label permutation for delta pairwise contrast consistency."""
    pairs = list(zip(a_none, a_full)) + list(zip(b_none, b_full))
    n_a = len(a_none)

    def delta(indices):
        none = [pairs[index][0] for index in indices]
        full = [pairs[index][1] for index in indices]
        return mean_pairwise_cosine(full) - mean_pairwise_cosine(none)

    a_indices = list(range(n_a))
    b_indices = list(range(n_a, len(pairs)))
    delta_a, delta_b = delta(a_indices), delta(b_indices)
    observed = delta_a - delta_b
    all_indices = set(range(len(pairs)))
    exceed, total = 0, 0
    for selected_tuple in combinations(range(len(pairs)), n_a):
        selected = list(selected_tuple)
        other = list(all_indices - set(selected))
        statistic = delta(selected) - delta(other)
        exceed += statistic >= observed - 1e-12
        total += 1
    return delta_a, delta_b, exceed / total


def bootstrap_mean_ci(values, rng, iterations=20_000):
    values = np.asarray(values)
    samples = rng.choice(values, size=(iterations, len(values)), replace=True).mean(axis=1)
    return np.quantile(samples, [0.025, 0.975]).tolist()


def holm_adjust(p_values):
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        value = (len(p_values) - rank) * p_values[index]
        running = max(running, value)
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def build_summary(results, contrasts, geometries, vectors):
    pivot = results.pivot_table(
        index=["config", "strategy", "subject"], columns="variant", values="mdm_accuracy"
    )
    pivot["full_gain_pp"] = 100 * (pivot["full"] - pivot["none"])
    pivot["scalar_gain_pp"] = 100 * (pivot["scalar"] - pivot["none"])
    pivot["full_minus_scalar_pp"] = 100 * (pivot["full"] - pivot["scalar"])
    pivot = pivot.reset_index()

    rng = np.random.default_rng(20261023)
    summary_rows = []
    for (config, strategy), frame in pivot.groupby(["config", "strategy"], sort=False):
        result_frame = results[(results.config == config) & (results.strategy == strategy)]
        none_shift = result_frame[result_frame.variant == "none"].set_index("subject").same_class_shift
        full_shift = result_frame[result_frame.variant == "full"].set_index("subject").same_class_shift
        contrast_frame = contrasts[(contrasts.config == config) & (contrasts.strategy == strategy)].set_index("variant")
        geometry = geometries[(geometries.config == config) & (geometries.strategy == strategy)]
        gain = frame.full_gain_pp.to_numpy()
        summary_rows.append({
            "config": config,
            "strategy": strategy,
            "mdm_none_pct": 100 * frame.none.mean(),
            "mdm_scalar_pct": 100 * frame.scalar.mean(),
            "mdm_full_pct": 100 * frame.full.mean(),
            "full_gain_pp": gain.mean(),
            "full_gain_ci95_low": bootstrap_mean_ci(gain, rng)[0],
            "full_gain_ci95_high": bootstrap_mean_ci(gain, rng)[1],
            "p_full_gain_two_sided": float(wilcoxon(frame.full, frame.none, method="exact").pvalue),
            "full_minus_scalar_pp": frame.full_minus_scalar_pp.mean(),
            "same_class_reduction_pct": 100 * (1 - full_shift.mean() / none_shift.mean()),
            "contrast_cosine_none": contrast_frame.loc["none", "mean_pairwise_contrast_cosine"],
            "contrast_cosine_full": contrast_frame.loc["full", "mean_pairwise_contrast_cosine"],
            "contrast_cosine_delta": (
                contrast_frame.loc["full", "mean_pairwise_contrast_cosine"]
                - contrast_frame.loc["none", "mean_pairwise_contrast_cosine"]
            ),
            "reference_eval_scale_shift": geometry.reference_eval_scale_shift.mean(),
            "reference_eval_shape_shift": geometry.reference_eval_shape_shift.mean(),
        })
    summary = pd.DataFrame(summary_rows)

    def gains(config, strategy):
        return pivot[(pivot.config == config) & (pivot.strategy == strategy)].set_index("subject").full_gain_pp

    pooled = gains("2b-3ch-LR", "pooled-S1S2S3")
    feedback_reference = gains("2b-3ch-LR", "feedback-S3-reference")
    two_a_22 = gains("2a-22ch-LR", "standard")
    two_a_3 = gains("2a-3ch-LR", "standard")

    protocol_difference = feedback_reference - pooled
    protocol_p = float(wilcoxon(
        protocol_difference, alternative="greater", method="exact"
    ).pvalue) if np.any(protocol_difference != 0) else 1.0
    interaction_22_p = exact_permutation_mean_greater(two_a_22.to_numpy(), pooled.to_numpy())
    interaction_3_p = exact_permutation_mean_greater(two_a_3.to_numpy(), pooled.to_numpy())

    key_a = ("standard", "none")
    key_af = ("standard", "full")
    key_b = ("pooled-S1S2S3", "none")
    key_bf = ("pooled-S1S2S3", "full")
    contrast_delta_a, contrast_delta_b, contrast_p = contrast_interaction_permutation(
        vectors["2a-3ch-LR"][key_a], vectors["2a-3ch-LR"][key_af],
        vectors["2b-3ch-LR"][key_b], vectors["2b-3ch-LR"][key_bf],
    )

    primary_p = [protocol_p, interaction_3_p, contrast_p]
    adjusted = holm_adjust(primary_p)
    tests = {
        "H1_2b_feedback_matched_reference_improves_full_RA_gain": {
            "mean_gain_pooled_pp": float(pooled.mean()),
            "mean_gain_feedback_S3_reference_pp": float(feedback_reference.mean()),
            "mean_paired_improvement_pp": float(protocol_difference.mean()),
            "wilcoxon_one_sided_p": protocol_p,
            "holm_p_three_primary_tests": adjusted[0],
        },
        "H2_RA_gain_2a_exceeds_2b_original_22ch": {
            "mean_gain_2a_pp": float(two_a_22.mean()),
            "mean_gain_2b_pp": float(pooled.mean()),
            "difference_pp": float(two_a_22.mean() - pooled.mean()),
            "exact_independent_permutation_one_sided_p": interaction_22_p,
            "note": "Descriptive original-design test; not in the three-test matched primary family.",
        },
        "H2_RA_gain_2a_exceeds_2b_matched_3ch_LR": {
            "mean_gain_2a_pp": float(two_a_3.mean()),
            "mean_gain_2b_pp": float(pooled.mean()),
            "difference_pp": float(two_a_3.mean() - pooled.mean()),
            "exact_independent_permutation_one_sided_p": interaction_3_p,
            "holm_p_three_primary_tests": adjusted[1],
        },
        "H3_RA_improves_cross_subject_class_contrast_more_on_2a_matched_3ch_LR": {
            "contrast_cosine_delta_2a": contrast_delta_a,
            "contrast_cosine_delta_2b": contrast_delta_b,
            "difference_in_deltas": contrast_delta_a - contrast_delta_b,
            "exact_subject_label_permutation_one_sided_p": contrast_p,
            "holm_p_three_primary_tests": adjusted[2],
        },
    }
    return pivot, summary, tests


def make_figure(summary, pivot, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))

    strategy_order = [
        "pooled-S1S2S3", "feedback-S3-reference", "feedback-S3-only", "screening-S1S2-only"
    ]
    labels = ["S1–3 pooled", "S3 ref only", "S3 only", "S1–2 only"]
    data = [
        pivot[(pivot.config == "2b-3ch-LR") & (pivot.strategy == strategy)].full_gain_pp
        for strategy in strategy_order
    ]
    axes[0].boxplot(data, tick_labels=labels, showmeans=True)
    axes[0].axhline(0, color="#777777", lw=0.8)
    axes[0].set_ylabel("Full RA MDM gain (pp)")
    axes[0].set_title("2b: reference/session ablation")
    axes[0].tick_params(axis="x", rotation=22)

    configs = [("2a-22ch-LR", "standard"), ("2a-3ch-LR", "standard"), ("2b-3ch-LR", "pooled-S1S2S3")]
    labels = ["2a 22ch LR", "2a 3ch LR", "2b 3ch LR"]
    values = [
        summary[(summary.config == config) & (summary.strategy == strategy)].iloc[0].full_gain_pp
        for config, strategy in configs
    ]
    axes[1].bar(labels, values, color=["#2a78d6", "#59a5d8", "#eb6834"])
    axes[1].axhline(0, color="#777777", lw=0.8)
    axes[1].set_ylabel("Full RA MDM gain (pp)")
    axes[1].set_title("Dataset effect after matching task/channels")
    axes[1].tick_params(axis="x", rotation=22)

    contrast_configs = [("2a-3ch-LR", "standard"), ("2b-3ch-LR", "pooled-S1S2S3")]
    x = np.arange(2)
    none_values, full_values = [], []
    for config, strategy in contrast_configs:
        row = summary[(summary.config == config) & (summary.strategy == strategy)].iloc[0]
        none_values.append(row.contrast_cosine_none)
        full_values.append(row.contrast_cosine_full)
    axes[2].bar(x - 0.18, none_values, 0.36, label="Before RA", color="#a3a29c")
    axes[2].bar(x + 0.18, full_values, 0.36, label="Full RA", color="#2a78d6")
    axes[2].set_xticks(x, ["2a 3ch LR", "2b 3ch LR"])
    axes[2].set_ylabel("Mean pairwise class-contrast cosine")
    axes[2].set_title("Cross-subject label geometry")
    axes[2].legend(frameon=False)

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#dddddd", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(output_dir / "ra_dataset_difference_summary.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / "ra_dataset_difference_summary.pdf", bbox_inches="tight")
    plt.close(fig)


def run_analysis(output_dir="analysis/ra_dataset_difference_outputs"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results, all_contrasts, all_geometries = [], [], []
    all_vectors = {}
    for config_name, spec in CONFIGS.items():
        print(f"Analysing {config_name} ...", flush=True)
        results, contrasts, geometries, vectors = analyse_configuration(config_name, spec)
        all_results.append(results)
        all_contrasts.append(contrasts)
        all_geometries.append(geometries)
        all_vectors[config_name] = vectors
    results = pd.concat(all_results, ignore_index=True)
    contrasts = pd.concat(all_contrasts, ignore_index=True)
    geometries = pd.concat(all_geometries, ignore_index=True)
    subject_gains, summary, tests = build_summary(results, contrasts, geometries, all_vectors)

    results.to_csv(output_dir / "mdm_subject_results.csv", index=False)
    subject_gains.to_csv(output_dir / "mdm_subject_gains.csv", index=False)
    contrasts.to_csv(output_dir / "class_contrast_consistency.csv", index=False)
    geometries.to_csv(output_dir / "reference_eval_geometry.csv", index=False)
    summary.to_csv(output_dir / "ra_dataset_difference_summary.csv", index=False)
    (output_dir / "ra_dataset_difference_tests.json").write_text(json.dumps(tests, indent=2))
    make_figure(summary, subject_gains, output_dir)
    print(json.dumps(tests, indent=2))
    print(f"Saved outputs to {output_dir.resolve()}")
    return results, subject_gains, contrasts, geometries, summary, tests


if __name__ == "__main__":
    run_analysis()
