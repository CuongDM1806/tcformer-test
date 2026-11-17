"""Session arrays for the baselines that run outside the Lightning pipeline.

Same windows and chronological session split as the LOSO data modules:
(source training sessions / source validation sessions = target evaluation
sessions) BCI IV-2a 1 / 2, BCI IV-2b 1-3 / 4-5, Zhou2016 1-2 / 3, OpenBMI 1 / 2.
"""
import numpy as np

from ..data import get_datamodule_cls
from ..data.base import BaseDataModule
from ..data.bcic2a import train_test_sessions
from ..data.bcic2b import five_sessions
from ..data.loaders import load_bcic4, load_zhou2016
from ..data.zhou2016 import three_sessions

SPLITS = {
    "bcic2a": dict(train=[0], test=[1]),
    "bcic2b": dict(train=[0, 1, 2], test=[3, 4]),
    "zhou2016": dict(train=[0, 1], test=[2]),
    "openbmi": dict(train=[0], test=[1]),
}


def _subject_sessions(dataset, windows, subject):
    by_subject = windows.split("subject")
    subject_ds = by_subject.get(str(subject), by_subject.get(subject))
    if dataset == "bcic2a":
        return list(train_test_sessions(subject_ds))
    if dataset == "bcic2b":
        return five_sessions(subject_ds)
    return three_sessions(subject_ds)


def load_sessions(dataset, preprocessing):
    """{subject: [(X float64, y int64) per session]} in chronological order."""
    if dataset not in SPLITS:
        raise ValueError(f"{dataset} is not supported by this baseline runner.")
    subject_ids = get_datamodule_cls(dataset).all_subject_ids
    if dataset == "openbmi":
        from ..data.openbmi_cache import load_openbmi_subject

        # One preprocessed file per subject (data/openbmi*/openbmi_SXX.npz): sessions 1 and 2.
        return {subject: [(X.astype(np.float64), y.astype(np.int64))
                          for X, y in load_openbmi_subject(subject, preprocessing)]
                for subject in subject_ids}
    if dataset == "zhou2016":
        windows = load_zhou2016(subject_ids, preprocessing)
    else:
        windows = load_bcic4(subject_ids, dataset[-2:], preprocessing)
    sessions = {}
    for subject in subject_ids:
        arrays = []
        for session in _subject_sessions(dataset, windows, subject):
            X, y = BaseDataModule.to_arrays(session)
            arrays.append((X.astype(np.float64), y.astype(np.int64)))
        sessions[subject] = arrays
    return sessions


def concat(arrays, idx):
    return (np.concatenate([arrays[i][0] for i in idx]),
            np.concatenate([arrays[i][1] for i in idx]))
