"""Rotation (Procrustes) alignment after Riemannian alignment (RA).

RA whitens every subject by its own reference, which removes the subject-specific channel
mixing A_s only up to an orthogonal matrix: after RA the subject's data are U_s z, with an
unknown rotation U_s (docs/RA_GIAI_THICH.md, section 3.4). This module estimates a rotation
per subject so that its class means match a common template, as the last step of Riemannian
Procrustes Analysis (Rodrigues et al., 2019), in the log-Euclidean tangent space at the identity.

Rotation acts on covariances by congruence, C -> U C U^T, so log(U C U^T) = U log(C) U^T,
and on signals by x -> U x.

* Source subjects (labelled): generalized Procrustes. The template is the mean of the
  sources' rotated class means; each source is rotated onto it; repeated a few times.
* Target subject (unlabelled): its class means are taken with pseudo-labels (nearest template
  class mean), the rotation is fitted, and pseudo-labels are recomputed; target labels are never used.

The rotation minimizes sum_k w_k ||U A_k U^T - B_k||_F^2 over orthogonal U by Riemannian
gradient descent on the orthogonal group (U <- expm(-t Omega) U, Omega skew-symmetric),
starting at the identity, with backtracking.

Only numpy, scipy and scikit-learn. The same core is used by the proxy notebook of the analysis
(analysis/ra_rotation.py on feature/hada-full-mamba) and here, by the BCI IV-2a data module.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh, expm


def spd_log(matrix: np.ndarray) -> np.ndarray:
    values, vectors = eigh(matrix)
    return (vectors * np.log(np.maximum(values, 1e-10))) @ vectors.T


def log_covariances(covs: np.ndarray) -> np.ndarray:
    return np.stack([spd_log(c) for c in covs])


def class_means(logs: np.ndarray, labels: np.ndarray, classes) -> dict:
    return {k: logs[labels == k].mean(axis=0) for k in classes if np.any(labels == k)}


def _cost(rotation, pairs):
    return sum(w * float(np.sum((rotation @ a @ rotation.T - b) ** 2)) for a, b, w in pairs)


def fit_rotation(pairs, iterations: int = 100, tolerance: float = 1e-5) -> np.ndarray:
    """Orthogonal U minimizing sum_k w_k ||U A_k U^T - B_k||^2; pairs = [(A_k, B_k, w_k)]."""
    n = pairs[0][0].shape[0]
    rotation, cost, step = np.eye(n), _cost(np.eye(n), pairs), 1.0
    for _ in range(iterations):
        grad = sum(4.0 * w * (rotation @ a @ rotation.T - b) @ rotation @ a for a, b, w in pairs)
        omega = grad @ rotation.T - rotation @ grad.T          # skew-symmetric
        norm = np.linalg.norm(omega)
        if norm < 1e-12:
            break
        while step > 1e-8:
            candidate = expm(-step * omega / norm) @ rotation
            new_cost = _cost(candidate, pairs)
            if new_cost < cost:
                break
            step /= 2.0
        else:
            break
        improvement = cost - new_cost
        rotation, cost, step = candidate, new_cost, step * 2.0
        if improvement < tolerance * max(cost, 1e-12):
            break
    return rotation


def rotate_logs(logs: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    return np.einsum("ij,njk,lk->nil", rotation, logs, rotation, optimize=True)


def source_template(source_logs: list, source_labels: list, classes, iterations: int = 3):
    """Generalized Procrustes over the labelled sources: (template {class: mean}, [rotation per source])."""
    means = [class_means(L, y, classes) for L, y in zip(source_logs, source_labels)]
    n = source_logs[0].shape[1]
    rotations = [np.eye(n) for _ in means]

    def template_of(rotations):
        return {k: np.mean([u @ m[k] @ u.T for u, m in zip(rotations, means) if k in m], axis=0)
                for k in classes}

    for _ in range(iterations):
        template = template_of(rotations)
        rotations = [fit_rotation([(m[k], template[k], 1.0) for k in m]) for m in means]
    return template_of(rotations), rotations


def predict_nearest(logs: np.ndarray, template: dict) -> np.ndarray:
    keys = list(template)
    means = np.stack([template[k] for k in keys])
    dist = ((logs[:, None] - means[None]) ** 2).sum(axis=(-2, -1))
    return np.asarray(keys)[dist.argmin(axis=1)]


def target_rotation(target_logs: np.ndarray, template: dict, iterations: int = 5) -> np.ndarray:
    """Rotation of an unlabelled subject onto the template, with pseudo-labels."""
    rotation = np.eye(target_logs.shape[1])
    for _ in range(iterations):
        pseudo = predict_nearest(rotate_logs(target_logs, rotation), template)
        if len(np.unique(pseudo)) < 2:
            break
        means = class_means(target_logs, pseudo, template)
        pairs = [(means[k], template[k], float(np.mean(pseudo == k))) for k in means]
        rotation = fit_rotation(pairs)
    return rotation


def misalignment(logs_by_subject: list, labels_by_subject: list, classes) -> dict:
    """Diagnostic with all labels: class-mean mismatch to the template before/after rotation.

    rotation_share = share of the mismatch that rotation removes. Large values mean that after
    RA much of the between-subject difference is a rotation.
    """
    means = [class_means(L, y, classes) for L, y in zip(logs_by_subject, labels_by_subject)]
    plain = {k: np.mean([m[k] for m in means if k in m], axis=0) for k in classes}
    before = float(np.mean([sum(np.sum((m[k] - plain[k]) ** 2) for k in m) for m in means]))
    template, rotations = source_template(logs_by_subject, labels_by_subject, classes)
    after = float(np.mean([sum(np.sum((u @ m[k] @ u.T - template[k]) ** 2) for k in m)
                           for u, m in zip(rotations, means)]))
    return {"mismatch_before": before, "mismatch_after": after,
            "rotation_share": 1.0 - after / max(before, 1e-12)}


# ---------------------------------------------------------------- signals (network data pipeline)
def trial_logs(X: np.ndarray) -> np.ndarray:
    """Log of the OAS covariance of every trial of X [n, C, T] (as RA estimates covariances)."""
    from sklearn.covariance import OAS
    return log_covariances(np.stack([OAS().fit(trial.T.astype(np.float64)).covariance_ for trial in X]))


def rotate_signals(X: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    return np.einsum("cd,ndt->nct", rotation, X, optimize=True).astype(X.dtype, copy=False)


def align_rotations(train_arrays, val_arrays, X_target, X_test):
    """Rotate RA-aligned data of one LOSO fold onto the template of the labelled sources.

    train_arrays / val_arrays: [(X, y)] per source subject (same order). Each source is rotated
    with its labelled training trials; the target's rotation is fitted on its unlabelled X_target
    (pseudo-labels) and applied to X_target and X_test. Target labels are not used.
    """
    logs = [trial_logs(X) for X, _ in train_arrays]
    labels = [np.asarray(y) for _, y in train_arrays]
    classes = np.unique(np.concatenate(labels))
    template, rotations = source_template(logs, labels, classes)
    target = target_rotation(trial_logs(X_target), template)
    train = [(rotate_signals(X, u), y) for (X, y), u in zip(train_arrays, rotations)]
    val = [(rotate_signals(X, u), y) for (X, y), u in zip(val_arrays, rotations)]
    return train, val, rotate_signals(X_target, target), rotate_signals(X_test, target)
