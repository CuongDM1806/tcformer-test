"""Zhou2016 under the chronological LOSO protocol.

Sources: sessions 1-2 (training) and 3 (validation). Target: sessions 1-2
(unlabeled Stage-1 adaptation) and 3 (IM-TTA input and evaluation).
"""
from typing import Optional

import numpy as np

from . import alignment
from .base import BaseDataModule, ordered_sessions
from .loaders import load_zhou2016


def three_sessions(subject_dataset):
    sessions = ordered_sessions(subject_dataset)
    if len(sessions) != 3:
        raise RuntimeError("Zhou2016 must contain exactly three sessions per subject; "
                           f"found {len(sessions)}.")
    return sessions


class Zhou2016LOSO(BaseDataModule):
    all_subject_ids = list(range(1, 5))
    class_names = ["hand(L)", "hand(R)", "feet"]
    channels = 14
    classes = 3
    primary_test_label = "SESSION 3"
    auxiliary_test_label = "ALL 3 SESSIONS"

    def prepare_data(self) -> None:
        if self._setup_complete or self.dataset is not None:
            return
        self.dataset = load_zhou2016(self.all_subject_ids, self.preprocessing)

    @staticmethod
    def _subject(by_subject, subject_id):
        dataset = by_subject.get(str(subject_id), by_subject.get(subject_id))
        if dataset is None:
            raise KeyError(f"Zhou2016 subject {subject_id} is absent: {list(by_subject)}")
        return dataset

    def setup(self, stage: Optional[str] = None) -> None:
        if self._setup_complete:
            return
        if self.dataset is None:
            self.prepare_data()
        by_subject = self.dataset.split("subject")
        source_ids = [s for s in self.all_subject_ids if s != self.subject_id]

        use_ra = self.preprocessing.get("riemannian_alignment", False)
        per_session_ra = use_ra and alignment.per_session(self.preprocessing)
        calibration_ra = use_ra and not per_session_ra
        if per_session_ra:
            print(f"Per-session RA for Zhou2016 LOSO target {self.subject_id} "
                  "(reference = each session of each subject)", flush=True)

        train_arrays, val_arrays = [], []
        for source_id in source_ids:
            sessions = [self.to_arrays(s)
                        for s in three_sessions(self._subject(by_subject, source_id))]
            if per_session_ra:
                sessions = [(alignment.align(X_s)[0], y_s) for X_s, y_s in sessions]
            X_s = np.concatenate([sessions[0][0], sessions[1][0]], axis=0)
            y_s = np.concatenate([sessions[0][1], sessions[1][1]], axis=0)
            X_v, y_v = sessions[2]
            if calibration_ra:
                X_s, X_v = alignment.align(X_s, X_v)
            train_arrays.append((X_s, y_s))
            val_arrays.append((X_v, y_v))

        target = [self.to_arrays(s)
                  for s in three_sessions(self._subject(by_subject, self.subject_id))]
        if per_session_ra:
            target = [(alignment.align(X_s)[0], y_s) for X_s, y_s in target]
        X_target = np.concatenate([target[0][0], target[1][0]], axis=0)
        y_target = np.concatenate([target[0][1], target[1][1]], axis=0)
        X_test, y_test = target[2]
        if calibration_ra:
            # Fitted on the adaptation sessions 1-2, applied unchanged to session 3.
            X_target, X_test = alignment.align(X_target, X_test)

        X = np.concatenate([arr[0] for arr in train_arrays], axis=0)
        y = np.concatenate([arr[1] for arr in train_arrays], axis=0)
        X_val = np.concatenate([arr[0] for arr in val_arrays], axis=0)
        y_val = np.concatenate([arr[1] for arr in val_arrays], axis=0)

        timepoints = int(round(self.preprocessing["sfreq"]
                               * float(self.preprocessing.get("trial_duration", 5.0))))
        for name, array in (("source train", X), ("source validation", X_val),
                            ("target adaptation", X_target), ("target test", X_test)):
            if array.ndim != 3 or array.shape[1:] != (self.channels, timepoints):
                raise RuntimeError(f"Zhou2016 {name} has shape {tuple(array.shape)}; "
                                   f"expected [trials, {self.channels}, {timepoints}].")

        self._build(X, y, X_val, y_val, X_target, X_test, y_test, y_target=y_target)
        self._setup_complete = True
        self.dataset = None
        start = float(self.preprocessing.get("start", 0.0))
        print(f"Zhou2016 LOSO target S{self.subject_id:02d} | "
              f"window={start:.1f}-{start + timepoints / self.preprocessing['sfreq']:.1f} s | "
              f"source_train={len(y)} | source_val={len(y_val)} | "
              f"target_adaptation={len(X_target)} | target_test={len(y_test)}", flush=True)
