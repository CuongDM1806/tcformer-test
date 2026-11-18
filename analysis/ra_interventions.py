"""Claim 1, item 5 (+6): interventions on the data, scored with covariance proxy classifiers.

For one dataset at a time, one data property is switched off (or on) and the RA gain
Delta_RA = acc(with RA) - acc(without RA) is measured again with two proxy classifiers:

MDM    nearest class mean in log-Euclidean tangent space (no learned weights)
TS-LR  tangent-space vectors -> standardisation -> logistic regression

Both use the same tangent vectors (upper triangle of log(C), off-diagonal scaled by sqrt(2)),
so the log-Euclidean distance is the Euclidean distance between vectors.

Protocol (as the paper): leave one subject out. Source subjects contribute their labelled
fit sessions (the training sessions); the target subject is scored on its later sessions.
RA settings, each applied to every subject:
  none         no alignment
  frozen       whitened by the Riemannian mean of the subject's fit sessions
  per_session  every session whitened by its own Riemannian mean
No network is trained. Used by ra_interventions.ipynb (Kaggle, CPU only).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import ra_data_characteristics as rdc

RA_SETTINGS = ("none", "frozen", "per_session")
CLASSIFIERS = ("MDM", "TS-LR")


@dataclass
class SubjectCovs:
    subject: int
    sessions: list            # [(covs [trials, C, C], y), ...] chronological
    fit_sessions: int


# ---------------------------------------------------------------- signal transforms
def common_average_reference(x: np.ndarray) -> np.ndarray:
    """Subtract the channel mean at every sample (what the OpenBMI pipeline applies)."""
    return x - x.mean(axis=1, keepdims=True)


def to_covariances(view: rdc.View, transform=None, log=print) -> list[SubjectCovs]:
    out = []
    for sd in view.subjects:
        sessions = []
        for x, y in sd.sessions:
            x = transform(x) if transform is not None else x
            sessions.append((rdc.oas_covariances(x), y))
        out.append(SubjectCovs(sd.subject, sessions, sd.fit_sessions))
    log(f"  covariances: {view.name}, {len(out)} subjects, "
        f"{out[0].sessions[0][0].shape[1]} channels{' (transformed)' if transform else ''}")
    return out


# ---------------------------------------------------------------- alignment
def whiten(covs: np.ndarray, reference: np.ndarray) -> np.ndarray:
    w = rdc.spd_power(reference, -0.5)
    return np.einsum("ij,njk,kl->nil", w, covs, w, optimize=True)


def aligned_split(sc: SubjectCovs, setting: str, ref_trials: int | None,
                  rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Tangent vectors and labels of the fit sessions (train) and later sessions (test)."""
    sessions = sc.sessions
    if setting == "frozen":
        fit = np.concatenate([c for c, _ in sessions[:sc.fit_sessions]])
        if ref_trials is not None and ref_trials < len(fit):
            fit = fit[rng.choice(len(fit), ref_trials, replace=False)]
        reference = rdc.riemann_mean(fit)
        sessions = [(whiten(c, reference), y) for c, y in sessions]
    elif setting == "per_session":
        sessions = [(whiten(c, rdc.riemann_mean(c)), y) for c, y in sessions]
    elif setting != "none":
        raise ValueError(setting)
    train, test = sessions[:sc.fit_sessions], sessions[sc.fit_sessions:]
    return (rdc.tangent_vectors(np.concatenate([c for c, _ in train])), np.concatenate([y for _, y in train]),
            rdc.tangent_vectors(np.concatenate([c for c, _ in test])), np.concatenate([y for _, y in test]))


# ---------------------------------------------------------------- classifiers
def mdm_predict(x_train, y_train, x_test):
    classes = np.unique(y_train)
    means = np.stack([x_train[y_train == k].mean(axis=0) for k in classes])
    dist = ((x_test[:, None, :] - means[None]) ** 2).sum(axis=-1)
    return classes[dist.argmin(axis=1)]


def tslr_predict(x_train, y_train, x_test):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    return model.fit(x_train, y_train).predict(x_test)


PREDICT = {"MDM": mdm_predict, "TS-LR": tslr_predict}


def loso_accuracy(subjects: list[SubjectCovs], setting: str, ref_trials: int | None = None,
                  repeats: int = 1, seed: int = 0) -> pd.DataFrame:
    """Per-target accuracy (%) of both classifiers, averaged over `repeats` reference draws."""
    rows = []
    for r in range(repeats):
        rng = np.random.default_rng(seed + r)
        parts = {sc.subject: aligned_split(sc, setting, ref_trials, rng) for sc in subjects}
        for target in parts:
            x_tr = np.concatenate([parts[s][0] for s in parts if s != target])
            y_tr = np.concatenate([parts[s][1] for s in parts if s != target])
            _, _, x_te, y_te = parts[target]
            for name in CLASSIFIERS:
                acc = 100.0 * float(np.mean(PREDICT[name](x_tr, y_tr, x_te) == y_te))
                rows.append({"subject": target, "classifier": name, "repeat": r, "acc": acc})
    return (pd.DataFrame(rows).groupby(["subject", "classifier"], as_index=False)["acc"].mean())


def ra_gain_table(dataset: str, intervention: str, subjects: list[SubjectCovs],
                  ref_trials: int | None = None, repeats: int = 1, log=print) -> pd.DataFrame:
    """Accuracy for the three RA settings and the RA gains, per subject and classifier.

    ref_trials limits the trials of the frozen reference only; the other settings are unchanged.
    """
    tables = []
    for setting in RA_SETTINGS:
        reps = repeats if (setting == "frozen" and ref_trials is not None) else 1
        t = loso_accuracy(subjects, setting, ref_trials if setting == "frozen" else None, reps)
        tables.append(t.rename(columns={"acc": f"acc_{setting}"}))
    table = tables[0].merge(tables[1], on=["subject", "classifier"]).merge(tables[2], on=["subject", "classifier"])
    table["gain_frozen"] = table["acc_frozen"] - table["acc_none"]
    table["gain_per_session"] = table["acc_per_session"] - table["acc_none"]
    table.insert(0, "intervention", intervention)
    table.insert(0, "dataset", dataset)
    for name, g in table.groupby("classifier"):
        log(f"  {dataset} | {intervention:<22} | {name:<5}: none {g.acc_none.mean():5.2f}  "
            f"frozen {g.acc_frozen.mean():5.2f} ({g.gain_frozen.mean():+5.2f})  "
            f"per-session {g.acc_per_session.mean():5.2f} ({g.gain_per_session.mean():+5.2f})")
    return table


# ---------------------------------------------------------------- summary
def _p(values: np.ndarray) -> float:
    values = values[np.abs(values) > 1e-9]
    if len(values) == 0:
        return float("nan")
    return float(wilcoxon(values, alternative="two-sided", method="exact" if len(values) <= 25 else "auto").pvalue)


def summarise(results: pd.DataFrame, baseline: dict[str, str]) -> pd.DataFrame:
    """Mean gains per (dataset, intervention, classifier) and the paired change vs the dataset baseline.

    p_gain:   Wilcoxon of the per-subject RA gain against 0.
    p_change: Wilcoxon of the per-subject gain difference (intervention - baseline).
    """
    rows = []
    for (dataset, intervention, clf), g in results.groupby(["dataset", "intervention", "classifier"], sort=False):
        base = results[(results.dataset == dataset) & (results.intervention == baseline[dataset])
                       & (results.classifier == clf)].set_index("subject")
        g = g.set_index("subject")
        row = {"dataset": dataset, "intervention": intervention, "classifier": clf, "n": len(g),
               "acc_none": g.acc_none.mean(), "acc_frozen": g.acc_frozen.mean(),
               "acc_per_session": g.acc_per_session.mean()}
        for kind in ("frozen", "per_session"):
            gain = g[f"gain_{kind}"]
            row[f"gain_{kind}"] = gain.mean()
            row[f"p_gain_{kind}"] = _p(gain.to_numpy())
            if intervention != baseline[dataset]:
                change = (gain - base.loc[g.index, f"gain_{kind}"]).to_numpy()
                row[f"change_vs_baseline_{kind}"] = change.mean()
                row[f"p_change_{kind}"] = _p(change)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- H3 on OpenBMI: fewer channels
def subset_channels(subjects: list[SubjectCovs], channels: np.ndarray) -> list[SubjectCovs]:
    """The covariance of a channel subset is the corresponding sub-matrix."""
    ix = np.ix_(channels, channels)
    return [SubjectCovs(sc.subject, [(c[:, ix[0], ix[1]], y) for c, y in sc.sessions], sc.fit_sessions)
            for sc in subjects]


def ra_gain_random_channels(dataset: str, intervention: str, subjects: list[SubjectCovs], n_channels: int,
                            draws: int = 5, seed: int = 0, log=print) -> pd.DataFrame:
    """RA gains on `draws` random subsets of `n_channels` channels (same subset for every subject), averaged.

    Fewer channels means more reference trials per channel (H3) without changing the trials.
    """
    rng = np.random.default_rng(seed)
    n_all = subjects[0].sessions[0][0].shape[1]
    tables = []
    for d in range(draws):
        channels = np.sort(rng.choice(n_all, n_channels, replace=False))
        tables.append(ra_gain_table(dataset, intervention, subset_channels(subjects, channels), log=lambda m: None))
        log(f"  {dataset} | {intervention}: draw {d + 1}/{draws} done")
    table = (pd.concat(tables).groupby(["dataset", "intervention", "subject", "classifier"], as_index=False)
             .mean(numeric_only=True))
    for name, g in table.groupby("classifier"):
        log(f"  {dataset} | {intervention:<22} | {name:<5}: none {g.acc_none.mean():5.2f}  "
            f"frozen {g.acc_frozen.mean():5.2f} ({g.gain_frozen.mean():+5.2f})  "
            f"per-session {g.acc_per_session.mean():5.2f} ({g.gain_per_session.mean():+5.2f})")
    return table
