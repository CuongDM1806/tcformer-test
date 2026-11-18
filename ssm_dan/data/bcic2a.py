"""BCI Competition IV-2a under the chronological LOSO protocol.

Sources: session T (training) and session E (validation). Target: session T
(unlabeled Stage-1 adaptation) and session E (IM-TTA input and evaluation).
"""
from typing import Optional

import numpy as np

from . import alignment, rotation
from .base import BaseDataModule, ordered_sessions
from .loaders import load_bcic4


def train_test_sessions(windows_dataset):
    """(session T, session E) of one subject."""
    split = windows_dataset.split("session")
    if "session_T" in split and "session_E" in split:
        return split["session_T"], split["session_E"]
    keys = list(split.keys())
    train_keys = [key for key in keys if "train" in str(key).lower()]
    test_keys = [key for key in keys
                 if "test" in str(key).lower() or "eval" in str(key).lower()]
    if train_keys and test_keys:
        return split[train_keys[0]], split[test_keys[0]]
    sessions = ordered_sessions(windows_dataset)
    if len(sessions) < 2:
        raise KeyError(f"Expected at least 2 BCI IV-2a sessions, got {keys}")
    return sessions[0], sessions[1]


class BCICIV2aLOSO(BaseDataModule):
    all_subject_ids = list(range(1, 10))
    class_names = ["feet", "hand(L)", "hand(R)", "tongue"]
    channels = 22
    classes = 4
    primary_test_label = "SESSION E"
    auxiliary_test_label = "SESSIONS T+E"

    def prepare_data(self) -> None:
        # Lightning may call the data hooks again for testing; do not reload.
        if self._setup_complete or self.dataset is not None:
            return
        self.dataset = load_bcic4(self.all_subject_ids, "2a", self.preprocessing)

    def setup(self, stage: Optional[str] = None) -> None:
        if self._setup_complete:
            return
        if self.dataset is None:
            self.prepare_data()
        by_subject = self.dataset.split("subject")
        source_ids = [s for s in self.all_subject_ids if s != self.subject_id]
        source_sessions = [train_test_sessions(by_subject[str(s)]) for s in source_ids]
        target_train, target_test = train_test_sessions(by_subject[str(self.subject_id)])

        train_arrays = [self.to_arrays(sessions[0]) for sessions in source_sessions]
        val_arrays = [self.to_arrays(sessions[1]) for sessions in source_sessions]
        X_test, y_test = self.to_arrays(target_test)
        X_target, y_target = self.to_arrays(target_train)

        if self.preprocessing.get("riemannian_alignment", False):
            if alignment.per_session(self.preprocessing):
                print(f"Per-session RA for LOSO target {self.subject_id} "
                      "(reference = each session of each subject)", flush=True)
                train_arrays = [(alignment.align(X)[0], y) for X, y in train_arrays]
                val_arrays = [(alignment.align(X)[0], y) for X, y in val_arrays]
                (X_target,) = alignment.align(X_target)
                (X_test,) = alignment.align(X_test)
            else:
                # One reference per subject, fitted on its session T only.
                print(f"Calibration RA for LOSO target {self.subject_id} "
                      "(reference = each subject's session T)", flush=True)
                aligned_train, aligned_val = [], []
                for (X, y), (X_val, y_val) in zip(train_arrays, val_arrays):
                    X, X_val = alignment.align(X, X_val)
                    aligned_train.append((X, y))
                    aligned_val.append((X_val, y_val))
                train_arrays, val_arrays = aligned_train, aligned_val
                X_target, X_test = alignment.align(X_target, X_test)

        if self.preprocessing.get("ra_rotation", False):
            # Claim 1, item 13: rotation (Procrustes) alignment after the calibration RA.
            if not self.preprocessing.get("riemannian_alignment", False) or alignment.per_session(self.preprocessing):
                raise ValueError("ra_rotation needs riemannian_alignment with ra_reference: calibration.")
            print(f"Rotation alignment for LOSO target {self.subject_id} "
                  "(sources: labelled template; target: pseudo-labels on session T)", flush=True)
            train_arrays, val_arrays, X_target, X_test = rotation.align_rotations(
                train_arrays, val_arrays, X_target, X_test)

        X = np.concatenate([arr[0] for arr in train_arrays], axis=0)
        y = np.concatenate([arr[1] for arr in train_arrays], axis=0)
        X_val = np.concatenate([arr[0] for arr in val_arrays], axis=0)
        y_val = np.concatenate([arr[1] for arr in val_arrays], axis=0)
        self._build(X, y, X_val, y_val, X_target, X_test, y_test, y_target=y_target)
        self._setup_complete = True
        # The tensors are self-contained; release the Braindecode windows.
        self.dataset = None
