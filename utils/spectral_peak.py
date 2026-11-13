"""Individual mu-rhythm peak frequency, used to check the learned spectral warp."""

import numpy as np


def mu_peak_hz(
    X,
    sfreq: float,
    band=(7.0, 14.0),
    fit_range=(3.0, 30.0),
    max_trials: int = 600,
    chunk: int = 64,
    seed: int = 0,
) -> float:
    """Peak frequency of the mu band above the 1/f background.

    ``X`` is (trials, channels, samples). The Hann-windowed power spectrum is
    averaged over trials and channels; a straight line in log-log coordinates
    is fitted over ``fit_range`` and removed, and the largest residual inside
    ``band`` (refined by a parabola through its neighbours) is returned.
    Only unlabeled EEG is used.
    """
    X = np.asarray(X)
    rng = np.random.default_rng(seed)
    if len(X) > max_trials:
        X = X[np.sort(rng.choice(len(X), max_trials, replace=False))]
    n_samples = X.shape[-1]
    window = np.hanning(n_samples).astype(np.float64)
    power = np.zeros(n_samples // 2 + 1)
    for begin in range(0, len(X), chunk):
        block = X[begin:begin + chunk].astype(np.float64) * window
        power += (np.abs(np.fft.rfft(block, axis=-1)) ** 2).sum(axis=(0, 1))
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sfreq)

    fit = (freqs >= fit_range[0]) & (freqs <= fit_range[1])
    log_f, log_p = np.log10(freqs[fit]), np.log10(power[fit] + 1e-30)
    slope, intercept = np.polyfit(log_f, log_p, 1)
    residual = np.log10(power + 1e-30) - (
        slope * np.log10(np.maximum(freqs, 1e-6)) + intercept
    )
    inside = np.flatnonzero((freqs >= band[0]) & (freqs <= band[1]))
    peak = inside[np.argmax(residual[inside])]
    if 0 < peak < len(freqs) - 1:
        left, center, right = residual[peak - 1:peak + 2]
        denominator = left - 2.0 * center + right
        if denominator < 0:
            offset = 0.5 * (left - right) / denominator
            return float(freqs[peak] + offset * (freqs[1] - freqs[0]))
    return float(freqs[peak])
