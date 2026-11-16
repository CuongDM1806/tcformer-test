"""OpenBMI (Lee2019_MI) under the chronological LOSO protocol.

Each subject has one labeled offline MI run per session (sessions on different
days, 100 trials each). Sources: session 1 (training) and session 2
(validation). Target: session 1 (unlabeled Stage-1 adaptation) and session 2
(IM-TTA input and evaluation). The preprocessed (and aligned) arrays of all
subjects are built once per process and shared by every fold.
"""
from typing import Optional

import numpy as np

from . import alignment
from .base import BaseDataModule
from .openbmi_cache import OPENBMI_CHANNELS, load_openbmi_subject


class OpenBMILOSO(BaseDataModule):
    all_subject_ids = list(range(1, 55))
    class_names = ["hand(L)", "hand(R)"]
    channels = OPENBMI_CHANNELS
    classes = 2
    primary_test_label = "SESSION 2"
    auxiliary_test_label = "SESSIONS 1+2"
    _arrays_cache = None
    _cache_signature = None

    def __init__(self, preprocessing: dict, subject_id: int):
        super().__init__(preprocessing, subject_id)
        if subject_id not in self.all_subject_ids:
            raise ValueError(f"OpenBMI target subject must be in 1..54, got {subject_id}.")

    def _signature(self):
        keys = ("sfreq", "low_cut", "high_cut", "start", "trial_duration",
                "common_average_reference", "riemannian_alignment", "ra_reference")
        return tuple(self.preprocessing.get(key) for key in keys)

    def prepare_data(self) -> None:
        signature = self._signature()
        if type(self)._arrays_cache is not None and type(self)._cache_signature == signature:
            return
        arrays = {}
        for subject_id in self.all_subject_ids:
            (X_1, y_1), (X_2, y_2) = load_openbmi_subject(subject_id, self.preprocessing)
            if self.preprocessing.get("riemannian_alignment", True):
                if alignment.per_session(self.preprocessing):
                    (X_1,) = alignment.align(X_1)
                    (X_2,) = alignment.align(X_2)
                else:
                    # Reference fitted on session 1 only, applied to session 2.
                    X_1, X_2 = alignment.align(X_1, X_2)
            arrays[subject_id] = ((X_1, y_1), (X_2, y_2))
        type(self)._arrays_cache = arrays
        type(self)._cache_signature = signature

    def setup(self, stage: Optional[str] = None) -> None:
        if self._setup_complete:
            return
        if type(self)._arrays_cache is None or type(self)._cache_signature != self._signature():
            self.prepare_data()
        arrays = type(self)._arrays_cache
        source_ids = [s for s in self.all_subject_ids if s != self.subject_id]
        X = np.concatenate([arrays[s][0][0] for s in source_ids], axis=0)
        y = np.concatenate([arrays[s][0][1] for s in source_ids], axis=0)
        X_val = np.concatenate([arrays[s][1][0] for s in source_ids], axis=0)
        y_val = np.concatenate([arrays[s][1][1] for s in source_ids], axis=0)
        X_target, y_target = arrays[self.subject_id][0]   # labels: auxiliary view only
        X_test, y_test = arrays[self.subject_id][1]
        self._build(X, y, X_val, y_val, X_target, X_test, y_test, y_target=y_target)
        self._setup_complete = True
        print(f"OpenBMI LOSO target S{self.subject_id:02d} | source_subjects={len(source_ids)} | "
              f"source_train={len(y)} | source_val={len(y_val)} | "
              f"target_adaptation={len(X_target)} | target_test={len(y_test)}", flush=True)

    def test_dataloader(self):
        """The target's 100-trial session 2 as a single batch.

        IM-TTA makes BatchNorm use batch statistics; a batch of 48 would leave
        a last batch of only 4 trials.
        """
        return self._evaluation_loader(self.test_dataset, batch_size=len(self.test_dataset))
