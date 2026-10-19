from typing import Optional

import numpy as np
from torch.utils.data import DataLoader

from .base import BaseDataModule
from utils.load_openbmi import OPENBMI_CHANNELS, load_openbmi_subject


class OpenBMIMILOSO(BaseDataModule):
    """Chronological subject-independent OpenBMI (Lee2019_MI) split.

    Every subject has one labeled offline MI run per recording session
    (session 1 and session 2, recorded on different days; 100 trials each).
    For each held-out target, all other subjects are sources: their session 1
    is labeled training data and their session 2 is labeled source-only
    validation data (checkpoint selection). The target's session 1 is the
    unlabeled Stage-1 adaptation partition; its session 2 is used without
    labels for test-time adaptation and is the primary test set. RA is fitted
    per subject on session 1 only and applied unchanged to session 2. A
    sessions-1+2 target view is retained only as an auxiliary diagnostic.

    The preprocessed and RA-aligned arrays of all subjects are built once per
    process and shared by every fold.
    """

    all_subject_ids = list(range(1, 55))
    class_names = ["hand(L)", "hand(R)"]
    channels = OPENBMI_CHANNELS
    classes = 2
    primary_test_label = "SESSION 2"
    auxiliary_test_label = "SESSIONS 1+2"
    all_sessions_test_label = auxiliary_test_label
    _arrays_cache = None
    _cache_signature = None

    def __init__(self, preprocessing_dict: dict, subject_id: int):
        super().__init__(preprocessing_dict, subject_id)
        configured_ids = preprocessing_dict.get("dataset_subject_ids", self.all_subject_ids)
        self.dataset_subject_ids = [int(value) for value in configured_ids]
        if len(set(self.dataset_subject_ids)) != len(self.dataset_subject_ids):
            raise ValueError("OpenBMI dataset_subject_ids must be unique.")
        if not set(self.dataset_subject_ids).issubset(self.all_subject_ids):
            raise ValueError("OpenBMI subject IDs must be within 1..54.")
        if len(self.dataset_subject_ids) < 2:
            raise ValueError("OpenBMI LOSO requires at least two subjects.")
        if subject_id not in self.dataset_subject_ids:
            raise ValueError(f"Target subject {subject_id} is absent from dataset_subject_ids.")
        self._setup_complete = False

    def _signature(self):
        keys = ("sfreq", "low_cut", "high_cut", "start", "trial_duration",
                "common_average_reference", "riemannian_alignment")
        return (tuple(self.dataset_subject_ids),
                *(self.preprocessing_dict.get(key) for key in keys))

    def prepare_data(self) -> None:
        signature = self._signature()
        if type(self)._arrays_cache is not None and type(self)._cache_signature == signature:
            return
        arrays = {}
        for subject_id in self.dataset_subject_ids:
            (X_1, y_1), (X_2, y_2) = load_openbmi_subject(subject_id, self.preprocessing_dict)
            if self.preprocessing_dict.get("riemannian_alignment", True):
                # Reference fitted on session 1 only (labels unused).
                X_1, X_2 = BaseDataModule._riemannian_align_many(X_1, X_2)
            arrays[subject_id] = ((X_1, y_1), (X_2, y_2))
        type(self)._arrays_cache = arrays
        type(self)._cache_signature = signature

    def setup(self, stage: Optional[str] = None) -> None:
        if self._setup_complete:
            return
        if type(self)._arrays_cache is None or type(self)._cache_signature != self._signature():
            self.prepare_data()
        arrays = type(self)._arrays_cache

        source_ids = [value for value in self.dataset_subject_ids if value != self.subject_id]
        X_train = np.concatenate([arrays[s][0][0] for s in source_ids], axis=0)
        y_train = np.concatenate([arrays[s][0][1] for s in source_ids], axis=0)
        X_val = np.concatenate([arrays[s][1][0] for s in source_ids], axis=0)
        y_val = np.concatenate([arrays[s][1][1] for s in source_ids], axis=0)
        X_target, y_target_aux = arrays[self.subject_id][0]   # labels: auxiliary view only
        X_test, y_test = arrays[self.subject_id][1]

        if self.preprocessing_dict.get("z_scale", True):
            # Statistics of the labeled source training sessions only.
            X_train, X_val, X_target, X_test = BaseDataModule._z_scale_many(
                X_train, X_val, X_target, X_test)

        self.train_dataset = BaseDataModule._make_tensor_dataset(X_train, y_train)
        self.val_dataset = BaseDataModule._make_tensor_dataset(X_val, y_val)
        self.target_dataset = BaseDataModule._make_unlabeled_dataset(X_target)
        self.test_dataset = BaseDataModule._make_tensor_dataset(X_test, y_test)
        self.all_target_dataset = BaseDataModule._make_tensor_dataset(
            np.concatenate((X_target, X_test), axis=0),
            np.concatenate((y_target_aux, y_test), axis=0))
        self.dataset = None
        self._setup_complete = True

        trial_start = float(self.preprocessing_dict.get("start", 0.0))
        trial_duration = float(self.preprocessing_dict.get("trial_duration", 4.0))
        print(f"OpenBMI LOSO target S{self.subject_id:02d} | "
              f"window={trial_start:.1f}-{trial_start + trial_duration:.1f} s "
              f"({X_test.shape[-1]} samples) | source_subjects={len(source_ids)} | "
              f"source_train(session 1)={len(y_train)} | source_val(session 2)={len(y_val)} | "
              f"target_adaptation(session 1, unlabeled)={len(X_target)} | "
              f"target_test(session 2)={len(y_test)}", flush=True)

    def val_dataloader(self) -> DataLoader:
        """Validate on session 2 of the source subjects only."""
        num_workers = self.preprocessing_dict.get("test_num_workers", 0)
        return DataLoader(
            self.val_dataset,
            batch_size=self.preprocessing_dict["batch_size"],
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            **({"prefetch_factor": 2} if num_workers > 0 else {}),
        )
