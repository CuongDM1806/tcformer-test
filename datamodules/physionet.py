from typing import Optional

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .base import BaseDataModule
from utils.load_physionet import load_physionet


class PhysioNetMILOSO(BaseDataModule):
    """First-20-subject PhysioNet motor-imagery LOSO benchmark.

    PhysioNet has no separate train/evaluation session. In each fold, the
    target subject's complete imagery set is exposed without labels to HADA
    and is scored only after training (and optional IM-TTA). Each of the 19
    source subjects contributes a stratified 95/5 train/validation split.
    """

    all_subject_ids = list(range(1, 21))
    class_names = ["hand(L)", "hand(R)", "both hands", "feet"]
    channels = 64
    classes = 4
    primary_test_label = "IMAGERY TEST SET"

    # Materializing all 20 subjects once avoids decoding and windowing the
    # same EDF files again for every fold in a single LOSO process.
    _subject_arrays_cache = None

    def __init__(self, preprocessing_dict: dict, subject_id: int):
        super().__init__(preprocessing_dict, subject_id)
        if subject_id not in self.all_subject_ids:
            raise ValueError(
                f"PhysioNet target subject must be in 1..20, got {subject_id}."
            )
        self._setup_complete = False

    def prepare_data(self) -> None:
        if self._setup_complete or type(self)._subject_arrays_cache is not None:
            return
        if self.dataset is None:
            self.dataset = load_physionet(
                self.all_subject_ids, self.preprocessing_dict
            )

    def _materialize_subject_arrays(self):
        if type(self)._subject_arrays_cache is not None:
            return type(self)._subject_arrays_cache

        split_by_subject = self.dataset.split("subject")
        arrays = {}
        for subject_id in self.all_subject_ids:
            subject_dataset = split_by_subject.get(str(subject_id))
            if subject_dataset is None:
                subject_dataset = split_by_subject.get(subject_id)
            if subject_dataset is None:
                raise KeyError(
                    f"PhysioNet subject {subject_id} is absent; available keys: "
                    f"{list(split_by_subject)}"
                )
            X, y = BaseDataModule._dataset_to_arrays(subject_dataset)
            expected_classes = np.arange(self.classes)
            observed_classes, class_counts = np.unique(y, return_counts=True)
            expected_timepoints = int(
                round(
                    self.preprocessing_dict["sfreq"]
                    * self.preprocessing_dict.get("trial_duration", 4.1)
                )
            )
            if (
                X.ndim != 3
                or X.shape[1] != self.channels
                or X.shape[2] != expected_timepoints
            ):
                raise RuntimeError(
                    f"PhysioNet S{subject_id:03d} has invalid EEG shape "
                    f"{tuple(X.shape)}; expected "
                    f"[trials, {self.channels}, {expected_timepoints}]."
                )
            if not np.array_equal(observed_classes, expected_classes):
                raise RuntimeError(
                    f"PhysioNet S{subject_id:03d} has classes "
                    f"{observed_classes.tolist()}; expected "
                    f"{expected_classes.tolist()}."
                )
            arrays[subject_id] = (
                X.astype(np.float32, copy=False),
                y.astype(np.int64, copy=False),
            )
            print(
                f"PhysioNet S{subject_id:03d} | trials={len(y)} | "
                f"class_counts={class_counts.tolist()} | shape={tuple(X.shape)}",
                flush=True,
            )

        if set(arrays) != set(self.all_subject_ids):
            raise RuntimeError("PhysioNet loader did not materialize subjects 1..20.")
        type(self)._subject_arrays_cache = arrays
        return arrays

    def setup(self, stage: Optional[str] = None) -> None:
        if self._setup_complete:
            return
        if type(self)._subject_arrays_cache is None:
            if self.dataset is None:
                self.prepare_data()
            subject_arrays = self._materialize_subject_arrays()
        else:
            subject_arrays = type(self)._subject_arrays_cache

        seed = int(self.preprocessing_dict.get("seed", 0))
        validation_fraction = float(
            self.preprocessing_dict.get("validation_fraction", 0.05)
        )
        if not 0.0 < validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1.")

        source_ids = [
            subject_id
            for subject_id in self.all_subject_ids
            if subject_id != self.subject_id
        ]
        if len(source_ids) != 19 or self.subject_id in source_ids:
            raise RuntimeError("Invalid PhysioNet LOSO source/target separation.")

        train_arrays = []
        val_arrays = []
        for source_id in source_ids:
            X_source, y_source = subject_arrays[source_id]
            indices = np.arange(len(y_source))
            class_count = np.unique(y_source).size
            validation_size = max(
                int(np.ceil(len(y_source) * validation_fraction)), class_count
            )
            if len(y_source) - validation_size < class_count:
                raise ValueError(
                    f"PhysioNet S{source_id:03d} has too few trials for a "
                    f"stratified {validation_fraction:.1%} validation split."
                )
            train_indices, val_indices = train_test_split(
                indices,
                test_size=validation_size,
                random_state=seed + source_id,
                stratify=y_source,
            )
            if np.intersect1d(train_indices, val_indices).size:
                raise RuntimeError(
                    f"PhysioNet S{source_id:03d} train/validation overlap."
                )
            if len(train_indices) + len(val_indices) != len(indices):
                raise RuntimeError(
                    f"PhysioNet S{source_id:03d} split lost source trials."
                )
            X_train = X_source[train_indices]
            y_train = y_source[train_indices]
            X_val = X_source[val_indices]
            y_val = y_source[val_indices]

            if self.preprocessing_dict.get("riemannian_alignment", False):
                X_train, X_val = BaseDataModule._riemannian_align_many(
                    X_train, X_val
                )
            train_arrays.append((X_train, y_train))
            val_arrays.append((X_val, y_val))

        X_target, y_target = subject_arrays[self.subject_id]
        X_target = X_target.copy()
        y_target = y_target.copy()
        if self.preprocessing_dict.get("riemannian_alignment", False):
            # The reference uses target EEG only; target labels are not needed.
            X_target = BaseDataModule._riemannian_align_many(X_target)[0]

        X_train = np.concatenate([item[0] for item in train_arrays], axis=0)
        y_train = np.concatenate([item[1] for item in train_arrays], axis=0)
        X_val = np.concatenate([item[0] for item in val_arrays], axis=0)
        y_val = np.concatenate([item[1] for item in val_arrays], axis=0)

        if self.preprocessing_dict.get("z_scale", False):
            X_train, X_val, X_target = BaseDataModule._z_scale_many(
                X_train, X_val, X_target
            )

        self.train_dataset = BaseDataModule._make_tensor_dataset(X_train, y_train)
        self.val_dataset = BaseDataModule._make_tensor_dataset(X_val, y_val)
        self.target_dataset = BaseDataModule._make_unlabeled_dataset(X_target)
        self.test_dataset = BaseDataModule._make_tensor_dataset(X_target, y_target)
        self.all_target_dataset = None
        self._setup_complete = True
        self.dataset = None

        print(
            f"PhysioNet LOSO target S{self.subject_id:03d} | "
            f"source_subjects={len(source_ids)} | train={len(y_train)} | "
            f"val={len(y_val)} | target_unlabeled/test={len(y_target)}",
            flush=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Validate only on held-out labeled source trials.

        This override is essential for strict UDA: target labels must not be
        consumed by validation during ``fit``.
        """
        num_workers = self.preprocessing_dict.get("test_num_workers", 0)
        return DataLoader(
            self.val_dataset,
            batch_size=self.preprocessing_dict["batch_size"],
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            **({"prefetch_factor": 2} if num_workers > 0 else {}),
        )
