"""Claim 1, item 12: does a rotation remain after RA, and does removing it add to the RA gain?

Proxy classifiers (MDM, TS-LR) as in ra_interventions.py, same protocol (leave one subject out,
labelled fit sessions of the sources for training, target scored on its later sessions).
Settings, each with and without rotation alignment (ra_rotation.py):

  none             no alignment
  frozen           RA, reference of the subject's fit sessions
  frozen+rot       frozen, then Procrustes rotation onto the source template
  per_session      RA, one reference per session
  per_session+rot  per_session, then rotation

The target's rotation is fitted on its unlabelled fit sessions with pseudo-labels and applied to
its later sessions; target labels are used only for scoring. Sources are rotated with their labels.

Hypothesis (docs/CLAIM1_RA_BCI2A.md, 6a): RA leaves a subject-specific rotation with C(C-1)/2
degrees of freedom; its effect grows with the channel count, so the extra gain of rotation
(frozen+rot - frozen) should grow from 3 to 22 channels of BCI IV-2a.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import ra_data_characteristics as rdc
import ra_interventions as rai
import ra_rotation as rr

SETTINGS = ("none", "frozen", "frozen+rot", "per_session", "per_session+rot")
# Diagnostic only (item 15): the target rotation is fitted with the TRUE labels of its fit sessions
# (session T on BCI IV-2a), then applied to its later sessions. Not a valid method.
ORACLE_SETTINGS = ("frozen+oracle", "per_session+oracle")


def tangent_from_logs(logs: np.ndarray) -> np.ndarray:
    n = logs.shape[1]
    rows, cols = np.triu_indices(n)
    weight = np.where(rows == cols, 1.0, np.sqrt(2.0))
    return logs[:, rows, cols] * weight


def aligned_logs(sc: rai.SubjectCovs, base: str):
    """Log-covariances of the fit sessions (train) and later sessions (test) after RA `base`."""
    sessions = sc.sessions
    if base == "frozen":
        reference = rdc.riemann_mean(np.concatenate([c for c, _ in sessions[:sc.fit_sessions]]))
        sessions = [(rai.whiten(c, reference), y) for c, y in sessions]
    elif base == "per_session":
        sessions = [(rai.whiten(c, rdc.riemann_mean(c)), y) for c, y in sessions]
    train, test = sessions[:sc.fit_sessions], sessions[sc.fit_sessions:]
    return (rr.log_covariances(np.concatenate([c for c, _ in train])), np.concatenate([y for _, y in train]),
            rr.log_covariances(np.concatenate([c for c, _ in test])), np.concatenate([y for _, y in test]))


def loso(subjects: list[rai.SubjectCovs], setting: str) -> pd.DataFrame:
    oracle = setting.endswith("+oracle")
    base = setting.replace("+rot", "").replace("+oracle", "")
    rotate = setting.endswith("+rot") or oracle
    parts = {sc.subject: aligned_logs(sc, base) for sc in subjects}
    classes = np.unique(np.concatenate([p[1] for p in parts.values()]))
    rows = []
    for target in parts:
        sources = [s for s in parts if s != target]
        src_logs = [parts[s][0] for s in sources]
        src_y = [parts[s][1] for s in sources]
        tgt_fit, y_fit, tgt_test, y_test = parts[target]
        if rotate:
            template, rotations = rr.source_template(src_logs, src_y, classes)
            src_logs = [rr.rotate_logs(L, u) for L, u in zip(src_logs, rotations)]
            if oracle:
                means = rr.class_means(tgt_fit, y_fit, classes)
                u_target = rr.fit_rotation([(means[k], template[k], 1.0) for k in means])
            else:
                u_target = rr.target_rotation(tgt_fit, template)
            tgt_test = rr.rotate_logs(tgt_test, u_target)
        x_tr = tangent_from_logs(np.concatenate(src_logs))
        y_tr = np.concatenate(src_y)
        x_te = tangent_from_logs(tgt_test)
        for name in rai.CLASSIFIERS:
            acc = 100.0 * float(np.mean(rai.PREDICT[name](x_tr, y_tr, x_te) == y_test))
            rows.append({"subject": target, "classifier": name, "acc": acc})
    return pd.DataFrame(rows)


def rotation_table(dataset: str, view: str, subjects: list[rai.SubjectCovs], log=print,
                   oracle: bool = False) -> pd.DataFrame:
    table = None
    for setting in SETTINGS + (ORACLE_SETTINGS if oracle else ()):
        t = loso(subjects, setting).rename(columns={"acc": f"acc_{setting}"})
        table = t if table is None else table.merge(t, on=["subject", "classifier"])
    for base in ("frozen", "per_session"):
        table[f"gain_{base}"] = table[f"acc_{base}"] - table["acc_none"]
        table[f"gain_{base}+rot"] = table[f"acc_{base}+rot"] - table["acc_none"]
        table[f"extra_rot_{base}"] = table[f"acc_{base}+rot"] - table[f"acc_{base}"]
        if f"acc_{base}+oracle" in table:
            table[f"extra_oracle_{base}"] = table[f"acc_{base}+oracle"] - table[f"acc_{base}"]
    table.insert(0, "n_channels", subjects[0].sessions[0][0].shape[1])
    table.insert(0, "view", view)
    table.insert(0, "dataset", dataset)
    for name, g in table.groupby("classifier"):
        log(f"  {dataset} | {view:<18} | {name:<5}: none {g.acc_none.mean():5.2f} | "
            f"frozen {g['acc_frozen'].mean():5.2f} +rot {g['acc_frozen+rot'].mean():5.2f} "
            f"({g.extra_rot_frozen.mean():+5.2f}) | per-session {g['acc_per_session'].mean():5.2f} "
            f"+rot {g['acc_per_session+rot'].mean():5.2f} ({g.extra_rot_per_session.mean():+5.2f})"
            + (f" | oracle: frozen {g.extra_oracle_frozen.mean():+5.2f}, per-session "
               f"{g.extra_oracle_per_session.mean():+5.2f}" if "extra_oracle_frozen" in g else ""))
    return table


def rotation_diagnostic(dataset: str, view: str, subjects: list[rai.SubjectCovs]) -> dict:
    """Share of the class-mean mismatch between subjects (after frozen RA) that is a rotation.

    Uses the labels of every subject: a description of the data, not a classifier.
    """
    parts = [aligned_logs(sc, "frozen") for sc in subjects]
    logs = [np.concatenate([p[0], p[2]]) for p in parts]
    labels = [np.concatenate([p[1], p[3]]) for p in parts]
    classes = np.unique(np.concatenate(labels))
    out = rr.misalignment(logs, labels, classes)
    return {"dataset": dataset, "view": view, "n_channels": subjects[0].sessions[0][0].shape[1], **out}


def _p(values) -> float:
    values = np.asarray(values)
    values = values[np.abs(values) > 1e-9]
    if len(values) == 0:
        return float("nan")
    return float(wilcoxon(values, method="exact" if len(values) <= 25 else "auto").pvalue)


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["dataset", "view", "n_channels", "classifier"]
    for key, g in results.groupby(keys, sort=False):
        # Random channel subsets: average the draws per subject first.
        g = g.groupby("subject").mean(numeric_only=True)
        row = dict(zip(keys, key), n=len(g))
        for setting in SETTINGS + ORACLE_SETTINGS:
            if f"acc_{setting}" in g:
                row[f"acc_{setting}"] = g[f"acc_{setting}"].mean()
        for base in ("frozen", "per_session"):
            row[f"gain_{base}"] = g[f"gain_{base}"].mean()
            row[f"gain_{base}+rot"] = g[f"gain_{base}+rot"].mean()
            row[f"extra_rot_{base}"] = g[f"extra_rot_{base}"].mean()
            row[f"p_extra_rot_{base}"] = _p(g[f"extra_rot_{base}"])
            if f"extra_oracle_{base}" in g:
                row[f"extra_oracle_{base}"] = g[f"extra_oracle_{base}"].mean()
                row[f"p_extra_oracle_{base}"] = _p(g[f"extra_oracle_{base}"])
        rows.append(row)
    return pd.DataFrame(rows)
