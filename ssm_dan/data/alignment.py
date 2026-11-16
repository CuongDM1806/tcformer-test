"""Riemannian alignment (RA) of EEG trials.

Each trial covariance is estimated with OAS; the reference is the
affine-invariant Riemannian mean of a set of trials, and every trial is
whitened by the inverse square root of that reference. Labels are never used.
"""
import numpy as np
from scipy.linalg import eigh
from sklearn.covariance import OAS

RA_REFERENCES = ("calibration", "per_session")


def spd_power(matrix, exponent, epsilon=1e-8):
    """Raise a symmetric positive-definite matrix to a real power."""
    eigenvalues, eigenvectors = eigh(matrix)
    eigenvalues = np.maximum(eigenvalues, epsilon)
    return (eigenvectors * (eigenvalues ** exponent)) @ eigenvectors.T


def riemannian_mean(covariances, tolerance=1e-7, max_iterations=50):
    """Affine-invariant Riemannian mean of SPD matrices (fixed-point iteration)."""
    mean = np.mean(covariances, axis=0)
    for _ in range(max_iterations):
        mean_sqrt = spd_power(mean, 0.5)
        mean_invsqrt = spd_power(mean, -0.5)
        tangent = np.zeros_like(mean)

        for covariance in covariances:
            centered = mean_invsqrt @ covariance @ mean_invsqrt
            eigenvalues, eigenvectors = eigh(centered)
            eigenvalues = np.maximum(eigenvalues, 1e-8)
            tangent += (eigenvectors * np.log(eigenvalues)) @ eigenvectors.T

        tangent /= len(covariances)
        if np.linalg.norm(tangent, ord="fro") < tolerance:
            break

        eigenvalues, eigenvectors = eigh(tangent)
        tangent_exp = (eigenvectors * np.exp(eigenvalues)) @ eigenvectors.T
        mean = mean_sqrt @ tangent_exp @ mean_sqrt
        mean = 0.5 * (mean + mean.T)

    return mean


def whitener(X):
    """Inverse square root of the OAS/Riemannian reference of trials X [n, C, T]."""
    covariances = np.stack(
        [OAS().fit(trial.T.astype(np.float64)).covariance_ for trial in X],
        axis=0,
    )
    return spd_power(riemannian_mean(covariances), -0.5)


def align(X_reference, *other_arrays):
    """Fit RA on ``X_reference`` and apply the same whitening to every array."""
    matrix = whitener(X_reference)

    def transform(array):
        aligned = np.einsum("cd,ndt->nct", matrix, array, optimize=True)
        return aligned.astype(array.dtype, copy=False)

    return (transform(X_reference), *(transform(array) for array in other_arrays))


def per_session(preprocessing) -> bool:
    """Whether RA uses one reference per recording session.

    ``ra_reference: calibration`` (default) fits each subject's reference on its
    earlier (calibration) sessions and applies it unchanged to its later
    sessions. ``ra_reference: per_session`` whitens every recording session by
    its own label-free reference, the target's evaluation session included.
    """
    mode = preprocessing.get("ra_reference", "calibration")
    if mode not in RA_REFERENCES:
        raise ValueError(f"ra_reference must be one of {RA_REFERENCES}, got {mode!r}.")
    return mode == "per_session"
