"""Fused Gromov-Wasserstein channel alignment (label-free, per subject or session).

Each subject's functional channel structure (1 - magnitude-squared coherence, 8-30 Hz,
cross-spectra averaged over trials) is matched by fused GW to a template: the mean
structure of the LOSO source subjects' training sessions (the target never enters it).
The fused term is the squared 3D distance between electrodes of the standard 10-05
montage, so signal only moves towards neighbouring electrodes.  The coupling T gives a
mapping A = n * T^T; aligned channel j = sum_i A_ji x_i (rows of A sum to one).

Modes (``preprocessing.<dataset>.channel_alignment``):
  gw_chnorm  per-channel scaling (std on the reference trials), then GW. Replaces RA:
             requires ``riemannian_alignment: false``.
  gw_ra      GW, then the usual RA. Requires ``riemannian_alignment: true``.
The reference trials follow ``ra_reference``: "calibration" fits the mapping on each
subject's first session and applies it to the later one; "per_session" fits every
session on itself.
"""
from __future__ import annotations

import numpy as np

MODES = ("gw_chnorm", "gw_ra")

BCIC2A_CHANNELS = ["Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2",
                   "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz"]


def electrode_distance_mm(channel_names):
    import mne
    positions = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
    missing = [name for name in channel_names if name not in positions]
    if missing:
        raise ValueError(f"no standard position for channels {missing}")
    xyz = np.array([positions[name] for name in channel_names]) * 1000.0
    return np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)


def coherence_distance(X, sfreq, band=(8.0, 30.0)):
    """1 - mean magnitude-squared coherence over `band`; X is [trials, channels, time]."""
    X = np.asarray(X, dtype=np.float64)
    spectrum = np.fft.rfft(X * np.hanning(X.shape[-1]), axis=-1)
    freqs = np.fft.rfftfreq(X.shape[-1], 1.0 / sfreq)
    spectrum = spectrum[..., (freqs >= band[0]) & (freqs <= band[1])]
    cross = np.einsum("ncf,ndf->fcd", spectrum, spectrum.conj()) / len(X)
    power = np.real(np.einsum("fcc->fc", cross))
    coherence = np.abs(cross) ** 2 / (power[:, :, None] * power[:, None, :])
    distance = 1.0 - coherence.mean(axis=0)
    np.fill_diagonal(distance, 0.0)
    return distance


def channel_scale(X):
    """1 / per-channel standard deviation over all trials and samples."""
    X = np.asarray(X, dtype=np.float64)
    centered = X - X.mean(axis=-1, keepdims=True)
    return 1.0 / np.sqrt((centered ** 2).mean(axis=(0, 2)) + 1e-12)


class GWChannelAligner:
    def __init__(self, mode, channel_names, sfreq, alpha=0.5, band=(8.0, 30.0)):
        if mode not in MODES:
            raise ValueError(f"channel_alignment must be one of {MODES}, got {mode!r}")
        self.mode, self.sfreq, self.alpha, self.band = mode, sfreq, float(alpha), band
        self.distance_mm = electrode_distance_mm(channel_names)
        self.geometry = (self.distance_mm / self.distance_mm.max()) ** 2
        self.template = None

    def fit_template(self, source_references):
        self.template = np.mean([coherence_distance(X, self.sfreq, self.band)
                                 for X in source_references], axis=0)
        return self

    def transform(self, X_reference, *arrays):
        """Fit the mapping on X_reference (unlabelled) and apply it to it and `arrays`."""
        import ot
        if self.template is None:
            raise RuntimeError("fit_template must be called first")
        n = self.template.shape[0]
        weights = np.full(n, 1.0 / n)
        coupling = ot.gromov.fused_gromov_wasserstein(
            self.geometry, coherence_distance(X_reference, self.sfreq, self.band), self.template,
            weights, weights, loss_fun="square_loss", alpha=self.alpha)
        mapping = n * np.asarray(coupling).T
        if self.mode == "gw_chnorm":
            mapping = mapping * channel_scale(X_reference)[None, :]
        info = dict(diag_mass=float(np.trace(coupling)),
                    displacement_mm=float((coupling * self.distance_mm).sum()))

        def apply(X):
            return np.einsum("cd,ndt->nct", mapping, X, optimize=True).astype(X.dtype, copy=False)

        return (apply(X_reference), *(apply(X) for X in arrays)), info


def align_loso(preprocessing, channel_names, train_subjects, train_arrays, val_arrays,
               X_target, X_test, per_session, log=print):
    """Apply GW channel alignment to a LOSO split; returns the same structures."""
    mode = preprocessing["channel_alignment"]
    use_ra = bool(preprocessing.get("riemannian_alignment", False))
    if (mode == "gw_chnorm") == use_ra:
        raise ValueError(f"channel_alignment={mode} needs riemannian_alignment="
                         f"{mode == 'gw_ra'}")
    aligner = GWChannelAligner(mode, channel_names, preprocessing["sfreq"],
                               alpha=preprocessing.get("gw_alpha", 0.5))
    aligner.fit_template([X for X, _ in train_arrays])
    log(f"GW channel alignment: mode={mode}, alpha={aligner.alpha}, "
        f"reference={'per session' if per_session else 'calibration session'}, "
        f"template = {len(train_arrays)} source subjects")

    def report(name, info):
        log(f"  GW {name}: diag_mass={info['diag_mass']:.3f} "
            f"displacement={info['displacement_mm']:.2f} mm")

    new_train, new_val = [], []
    for subject, (X_tr, y_tr), (X_va, y_va) in zip(train_subjects, train_arrays, val_arrays):
        if per_session:
            (X_tr,), info = aligner.transform(X_tr)
            report(f"S{subject} T", info)
            (X_va,), info = aligner.transform(X_va)
            report(f"S{subject} E", info)
        else:
            (X_tr, X_va), info = aligner.transform(X_tr, X_va)
            report(f"S{subject}", info)
        new_train.append((X_tr, y_tr))
        new_val.append((X_va, y_va))
    if per_session:
        (X_target,), info = aligner.transform(X_target)
        report("target T", info)
        (X_test,), info = aligner.transform(X_test)
        report("target E", info)
    else:
        (X_target, X_test), info = aligner.transform(X_target, X_test)
        report("target", info)
    return new_train, new_val, X_target, X_test
