"""BCI Competition IV-2b under the chronological LOSO protocol.

Sources: sessions 1-3 (training) and 4-5 (validation). Target: sessions 1-3
(unlabeled Stage-1 adaptation) and 4-5 (IM-TTA input and evaluation).
"""
from typing import Optional

import numpy as np

from . import alignment
from .base import BaseDataModule, ordered_sessions
from .loaders import load_bcic4

CALIBRATION_SESSIONS = (0, 1, 2)
EVALUATION_SESSIONS = (3, 4)


def five_sessions(windows_dataset):
    split = windows_dataset.split("session")
    preferred = [f"session_{idx}" for idx in range(5)]
    if all(key in split for key in preferred):
        return [split[key] for key in preferred]
    sessions = ordered_sessions(windows_dataset)
    if len(sessions) < 5:
        raise KeyError(f"Expected 5 BCI IV-2b sessions, got {list(split.keys())}")
    return sessions[:5]


class BCICIV2bLOSO(BaseDataModule):
    all_subject_ids = list(range(1, 10))
    class_names = ["hand(L)", "hand(R)"]
    channels = 3
    classes = 2
    primary_test_label = "SESSIONS 4-5"
    auxiliary_test_label = "SESSIONS 1-5"

    def prepare_data(self) -> None:
        if self._setup_complete or self.dataset is not None:
            return
        self.dataset = load_bcic4(self.all_subject_ids, "2b", self.preprocessing)

    def setup(self, stage: Optional[str] = None) -> None:
        if self._setup_complete:
            return
        if self.dataset is None:
            self.prepare_data()
        by_subject = self.dataset.split("subject")
        source_ids = [s for s in self.all_subject_ids if s != self.subject_id]
        source_sessions = [five_sessions(by_subject[str(s)]) for s in source_ids]
        target_sessions = five_sessions(by_subject[str(self.subject_id)])

        # One array per (subject, session), in subject-major order.
        train_arrays = [self.to_arrays(sessions[k])
                        for sessions in source_sessions for k in CALIBRATION_SESSIONS]
        val_arrays = [self.to_arrays(sessions[k])
                      for sessions in source_sessions for k in EVALUATION_SESSIONS]
        test_arrays = [self.to_arrays(target_sessions[k]) for k in EVALUATION_SESSIONS]
        target_arrays = [self.to_arrays(target_sessions[k]) for k in CALIBRATION_SESSIONS]
        y_test = np.concatenate([arr[1] for arr in test_arrays], axis=0)
        X_test = np.concatenate([arr[0] for arr in test_arrays], axis=0)
        X_target = np.concatenate([arr[0] for arr in target_arrays], axis=0)

        if self.preprocessing.get("riemannian_alignment", False):
            if alignment.per_session(self.preprocessing):
                print(f"Per-session RA for BCI IV-2b LOSO target {self.subject_id} "
                      "(reference = each session of each subject)", flush=True)

                def align_sessions(arrays):
                    return [(alignment.align(X_s)[0], y_s) for X_s, y_s in arrays]

                train_arrays = align_sessions(train_arrays)
                val_arrays = align_sessions(val_arrays)
                test_arrays = align_sessions(test_arrays)
                target_arrays = align_sessions(target_arrays)
                X_test = np.concatenate([arr[0] for arr in test_arrays], axis=0)
                X_target = np.concatenate([arr[0] for arr in target_arrays], axis=0)
            else:
                # One reference per subject, fitted on its sessions 1-3 and
                # applied unchanged to its sessions 4-5.
                print(f"Calibration RA for BCI IV-2b LOSO target {self.subject_id} "
                      "(references = sessions 1-3)", flush=True)
                aligned_train, aligned_val = [], []
                n_cal, n_eval = len(CALIBRATION_SESSIONS), len(EVALUATION_SESSIONS)
                for index in range(len(source_ids)):
                    cal = train_arrays[n_cal * index:n_cal * (index + 1)]
                    ev = val_arrays[n_eval * index:n_eval * (index + 1)]
                    X_s = np.concatenate([arr[0] for arr in cal], axis=0)
                    y_s = np.concatenate([arr[1] for arr in cal], axis=0)
                    X_v = np.concatenate([arr[0] for arr in ev], axis=0)
                    y_v = np.concatenate([arr[1] for arr in ev], axis=0)
                    X_s, X_v = alignment.align(X_s, X_v)
                    aligned_train.append((X_s, y_s))
                    aligned_val.append((X_v, y_v))
                train_arrays, val_arrays = aligned_train, aligned_val
                X_target, X_test = alignment.align(X_target, X_test)

        X = np.concatenate([arr[0] for arr in train_arrays], axis=0)
        y = np.concatenate([arr[1] for arr in train_arrays], axis=0)
        X_val = np.concatenate([arr[0] for arr in val_arrays], axis=0)
        y_val = np.concatenate([arr[1] for arr in val_arrays], axis=0)
        self._build(X, y, X_val, y_val, X_target, X_test, y_test)
        self._setup_complete = True
        self.dataset = None
