from typing import Optional

import numpy as np
from torch.utils.data import DataLoader

from .base import BaseDataModule
from utils.load_openbmi import load_openbmi


def _ordered_sessions(windows_dataset):
    sessions = windows_dataset.split("session")

    def sort_key(item):
        key, _ = item
        digits = "".join(character for character in str(key) if character.isdigit())
        return (int(digits) if digits else 999, str(key))

    ordered = [dataset for _, dataset in sorted(sessions.items(), key=sort_key)]
    if len(ordered) != 2:
        raise RuntimeError(
            f"OpenBMI MI requires exactly two sessions, got {list(sessions)}."
        )
    return ordered[0], ordered[1]


class OpenBMIMILOSO(BaseDataModule):
    """OpenBMI MI cross-subject adaptation with a strict session split.

    For every source subject, session 1 supplies labeled training trials and
    session 2 supplies source-only validation trials. For the held-out target,
    session 1 is exposed without labels during HADA training and session 2 is
    the primary test set. RA references are always fitted on session 1 only.
    """

    all_subject_ids = list(range(1, 55))
    class_names = ["hand(L)", "hand(R)"]
    channels = 62
    classes = 2
    primary_test_label = "SESSION 2"
    _arrays_cache = None
    _cache_signature = None

    def __init__(self, preprocessing_dict: dict, subject_id: int):
        super().__init__(preprocessing_dict, subject_id)
        configured_ids = preprocessing_dict.get(
            "dataset_subject_ids", self.all_subject_ids
        )
        self.dataset_subject_ids = [int(value) for value in configured_ids]
        if len(set(self.dataset_subject_ids)) != len(self.dataset_subject_ids):
            raise ValueError("OpenBMI dataset_subject_ids must be unique.")
        if not set(self.dataset_subject_ids).issubset(self.all_subject_ids):
            raise ValueError("OpenBMI subject IDs must be within 1..54.")
        if len(self.dataset_subject_ids) < 2:
            raise ValueError("OpenBMI LOSO requires at least two subjects.")
        if subject_id not in self.dataset_subject_ids:
            raise ValueError(
                f"Target subject {subject_id} is absent from dataset_subject_ids."
            )
        self._setup_complete = False

    def _signature(self):
        keys = (
            "sfreq",
            "low_cut",
            "high_cut",
            "start",
            "trial_duration",
            "common_average_reference",
            "riemannian_alignment",
        )
        return (
            tuple(self.dataset_subject_ids),
            *(self.preprocessing_dict.get(key) for key in keys),
        )

    def prepare_data(self) -> None:
        signature = self._signature()
        if (
            type(self)._arrays_cache is not None
            and type(self)._cache_signature == signature
        ):
            return

        arrays = {}
        expected_timepoints = int(
            round(
                self.preprocessing_dict["sfreq"]
                * self.preprocessing_dict.get("trial_duration", 4.0)
            )
        )
        for subject_id in self.dataset_subject_ids:
            print(f"Loading OpenBMI MI S{subject_id:02d}", flush=True)
            windows = load_openbmi([subject_id], self.preprocessing_dict)
            session_1, session_2 = _ordered_sessions(windows)
            X_session_1, y_session_1 = BaseDataModule._dataset_to_arrays(session_1)
            X_session_2, y_session_2 = BaseDataModule._dataset_to_arrays(session_2)

            for session_id, X, y in (
                (1, X_session_1, y_session_1),
                (2, X_session_2, y_session_2),
            ):
                observed_classes, class_counts = np.unique(y, return_counts=True)
                if X.shape[1:] != (self.channels, expected_timepoints):
                    raise RuntimeError(
                        f"OpenBMI S{subject_id:02d} session {session_id} has "
                        f"shape {tuple(X.shape)}; expected "
                        f"[trials, {self.channels}, {expected_timepoints}]."
                    )
                if not np.array_equal(observed_classes, np.arange(self.classes)):
                    raise RuntimeError(
                        f"OpenBMI S{subject_id:02d} session {session_id} has "
                        f"classes {observed_classes.tolist()}, expected [0, 1]."
                    )
                print(
                    f"  session={session_id} | trials={len(y)} | "
                    f"class_counts={class_counts.tolist()} | shape={tuple(X.shape)}",
                    flush=True,
                )

            X_session_1 = X_session_1.astype(np.float32, copy=False)
            X_session_2 = X_session_2.astype(np.float32, copy=False)
            if self.preprocessing_dict.get("riemannian_alignment", True):
                print(
                    f"  RA S{subject_id:02d}: fit session 1, transform sessions 1+2",
                    flush=True,
                )
                X_session_1, X_session_2 = BaseDataModule._riemannian_align_many(
                    X_session_1, X_session_2
                )

            arrays[subject_id] = (
                (X_session_1, y_session_1.astype(np.int64, copy=False)),
                (X_session_2, y_session_2.astype(np.int64, copy=False)),
            )
            del windows, session_1, session_2

        type(self)._arrays_cache = arrays
        type(self)._cache_signature = signature

    def setup(self, stage: Optional[str] = None) -> None:
        if self._setup_complete:
            return
        if (
            type(self)._arrays_cache is None
            or type(self)._cache_signature != self._signature()
        ):
            self.prepare_data()
        arrays = type(self)._arrays_cache

        source_ids = [
            value for value in self.dataset_subject_ids if value != self.subject_id
        ]
        train_arrays = [arrays[value][0] for value in source_ids]
        val_arrays = [arrays[value][1] for value in source_ids]
        X_target, y_target = arrays[self.subject_id][0]
        X_test, y_test = arrays[self.subject_id][1]

        X_train = np.concatenate([item[0] for item in train_arrays], axis=0)
        y_train = np.concatenate([item[1] for item in train_arrays], axis=0)
        X_val = np.concatenate([item[0] for item in val_arrays], axis=0)
        y_val = np.concatenate([item[1] for item in val_arrays], axis=0)

        if self.preprocessing_dict.get("z_scale", True):
            X_train, X_val, X_target, X_test = BaseDataModule._z_scale_many(
                X_train, X_val, X_target, X_test
            )

        self.train_dataset = BaseDataModule._make_tensor_dataset(X_train, y_train)
        self.val_dataset = BaseDataModule._make_tensor_dataset(X_val, y_val)
        self.target_dataset = BaseDataModule._make_unlabeled_dataset(X_target)
        self.test_dataset = BaseDataModule._make_tensor_dataset(X_test, y_test)
        self.all_target_dataset = BaseDataModule._make_tensor_dataset(
            np.concatenate((X_target, X_test), axis=0),
            np.concatenate((y_target, y_test), axis=0),
        )
        self.dataset = None
        self._setup_complete = True

        print(
            f"OpenBMI LOSO target S{self.subject_id:02d} | "
            f"source_subjects={len(source_ids)} | source_train={len(y_train)} | "
            f"source_val={len(y_val)} | target_unlabeled={len(y_target)} | "
            f"test_session_2={len(y_test)}",
            flush=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Keep target labels out of validation during UDA training."""
        num_workers = self.preprocessing_dict.get("test_num_workers", 0)
        return DataLoader(
            self.val_dataset,
            batch_size=self.preprocessing_dict["batch_size"],
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            **({"prefetch_factor": 2} if num_workers > 0 else {}),
        )
