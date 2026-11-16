"""Riemannian alignment: the aligned reference is the identity; per-session mode switch."""
import numpy as np
import pytest

from ssm_dan.data import alignment


def _trials(n=60, channels=6, samples=250, seed=0):
    rng = np.random.default_rng(seed)
    mixing = rng.normal(size=(channels, channels))
    return np.einsum("cd,ndt->nct", mixing, rng.normal(size=(n, channels, samples)))


def test_whitener_maps_the_reference_to_identity():
    from sklearn.covariance import OAS
    X = _trials()
    reference = alignment.riemannian_mean(np.stack([OAS().fit(t.T).covariance_ for t in X]))
    matrix = alignment.whitener(X)
    assert np.allclose(matrix @ reference @ matrix, np.eye(X.shape[1]), atol=1e-6)


def test_same_whitening_for_every_array():
    X, Y = _trials(seed=1), _trials(seed=2)
    aligned_X, aligned_Y = alignment.align(X, Y)
    matrix = alignment.whitener(X)
    assert np.allclose(aligned_Y, np.einsum("cd,ndt->nct", matrix, Y))
    assert aligned_X.dtype == X.dtype


def test_reference_mode():
    assert not alignment.per_session({})
    assert alignment.per_session({"ra_reference": "per_session"})
    with pytest.raises(ValueError):
        alignment.per_session({"ra_reference": "session"})
