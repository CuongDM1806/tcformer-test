import numpy as np

from datamodules.base import BaseDataModule


def _trials(seed, n_trials=8, n_channels=4, n_times=64):
    rng = np.random.default_rng(seed)
    mixing = rng.normal(size=(n_channels, n_channels))
    signals = rng.normal(size=(n_trials, n_channels, n_times))
    return np.einsum("cd,ndt->nct", mixing, signals).astype(np.float32)


def test_frozen_ra_applies_to_each_test_trial_independently():
    adaptation = _trials(1)
    test = _trials(2, n_trials=5)

    whitener = BaseDataModule._fit_riemannian_whitener(adaptation)
    batch_aligned = BaseDataModule._apply_riemannian_whitener(whitener, test)
    trial_aligned = np.stack(
        [
            BaseDataModule._apply_riemannian_whitener(whitener, trial)
            for trial in test
        ]
    )

    np.testing.assert_allclose(batch_aligned, trial_aligned, rtol=1e-6, atol=1e-6)


def test_test_session_cannot_change_adaptation_whitener():
    adaptation = _trials(3)
    test_a = _trials(4, n_trials=5)
    test_b = _trials(5, n_trials=5) * 100.0

    whitener_before = BaseDataModule._fit_riemannian_whitener(adaptation)
    BaseDataModule._apply_riemannian_whitener(whitener_before, test_a)
    BaseDataModule._apply_riemannian_whitener(whitener_before, test_b)
    whitener_after = BaseDataModule._fit_riemannian_whitener(adaptation)

    np.testing.assert_allclose(whitener_before, whitener_after, rtol=0, atol=0)
