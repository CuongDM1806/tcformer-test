from typing import Optional

import numpy as np
from torch.utils.data.dataloader import DataLoader

from .base import BaseDataModule
from utils.load_bcic4 import load_bcic4


def _ordered_session_items(splitted_ds):
    def sort_key(item):
        key, _ = item
        digits = "".join(ch for ch in str(key) if ch.isdigit())
        return (int(digits) if digits else 999, str(key))

    return sorted(splitted_ds.items(), key=sort_key)


def _get_ordered_sessions(windows_dataset):
    splitted_ds = windows_dataset.split("session")
    preferred = [f"session_{idx}" for idx in range(5)]
    if all(key in splitted_ds for key in preferred):
        return [splitted_ds[key] for key in preferred]

    sessions = [dataset for _, dataset in _ordered_session_items(splitted_ds)]
    if len(sessions) < 5:
        raise KeyError(f"Expected at least 5 BCIC IV-2b sessions, got {list(splitted_ds.keys())}")
    return sessions[:5]


class BCICIV2b(BaseDataModule):
    all_subject_ids = list(range(1, 10))
    class_names = ["hand(L)", "hand(R)"]
    channels = 3
    classes = 2

    def __init__(self, preprocessing_dict, subject_id):
        super().__init__(preprocessing_dict, subject_id)

    def prepare_data(self) -> None:
        self.dataset = load_bcic4(subject_ids=[self.subject_id], dataset="2b",
                                  preprocessing_dict=self.preprocessing_dict)

    def setup(self, stage: Optional[str] = None) -> None:
        if self.dataset is None:
            self.prepare_data()
        # split the data
        sessions = _get_ordered_sessions(self.dataset)
        train_datasets = [sessions[session] for session in [0, 1, 2]]
        test_datasets = [sessions[session] for session in [3, 4]]

        # load the data
        train_arrays = [BaseDataModule._dataset_to_arrays(ds) for ds in train_datasets]
        test_arrays = [BaseDataModule._dataset_to_arrays(ds) for ds in test_datasets]
        X = np.concatenate([arr[0] for arr in train_arrays], axis=0)
        y = np.concatenate([arr[1] for arr in train_arrays], axis=0)
        X_test = np.concatenate([arr[0] for arr in test_arrays], axis=0)
        y_test = np.concatenate([arr[1] for arr in test_arrays], axis=0)

        # scale data
        if self.preprocessing_dict["z_scale"]:
            X, X_test = BaseDataModule._z_scale(X, X_test)

        # make datasets
        self.train_dataset = BaseDataModule._make_tensor_dataset(X, y)
        self.test_dataset = BaseDataModule._make_tensor_dataset(X_test, y_test)


class BCICIV2bLOSO(BCICIV2b):
    val_dataset = None

    def __init__(self, preprocessing_dict: dict, subject_id: int):
        super(BCICIV2bLOSO, self).__init__(preprocessing_dict, subject_id)
        self._setup_complete = False

    def prepare_data(self) -> None:
        if self._setup_complete or self.dataset is not None:
            return
        self.dataset = load_bcic4(
            subject_ids=self.all_subject_ids, dataset="2b",
            preprocessing_dict=self.preprocessing_dict)

    def setup(self, stage: Optional[str] = None) -> None:
        if self._setup_complete:
            return
        if self.dataset is None:
            self.prepare_data()
        # split the data
        splitted_ds = self.dataset.split("subject")
        train_subjects = [
            subj_id for subj_id in self.all_subject_ids if subj_id != self.subject_id]
        train_datasets = [
            _get_ordered_sessions(splitted_ds[str(subj_id)])[session]
            for subj_id in train_subjects for session in [0, 1, 2]]
        val_datasets = [
            _get_ordered_sessions(splitted_ds[str(subj_id)])[session]
            for subj_id in train_subjects for session in [3, 4]]
        test_datasets = [
            _get_ordered_sessions(splitted_ds[str(self.subject_id)])[session]
            for session in [3, 4]]
        target_datasets = [
            _get_ordered_sessions(splitted_ds[str(self.subject_id)])[session]
            for session in [0, 1, 2]]

        # load the data
        train_arrays = [BaseDataModule._dataset_to_arrays(ds) for ds in train_datasets]
        val_arrays = [BaseDataModule._dataset_to_arrays(ds) for ds in val_datasets]
        test_arrays = [BaseDataModule._dataset_to_arrays(ds) for ds in test_datasets]
        target_arrays = [BaseDataModule._dataset_to_arrays(ds) for ds in target_datasets]
        X_test = np.concatenate([arr[0] for arr in test_arrays], axis=0)
        y_test = np.concatenate([arr[1] for arr in test_arrays], axis=0)
        X_target = np.concatenate([arr[0] for arr in target_arrays], axis=0)

        if self.preprocessing_dict.get("riemannian_alignment", False):
            # Fit one whitening reference per subject. Only sessions 1-3 are
            # used to estimate it; sessions 4-5 never leak into the reference.
            print(
                f"Applying strict per-subject RA for BCIC IV-2b LOSO target "
                f"{self.subject_id}",
                flush=True,
            )
            aligned_train_arrays = []
            aligned_val_arrays = []
            for source_id in train_subjects:
                source_sessions = _get_ordered_sessions(splitted_ds[str(source_id)])
                source_train = [
                    BaseDataModule._dataset_to_arrays(source_sessions[idx])
                    for idx in (0, 1, 2)
                ]
                source_val = [
                    BaseDataModule._dataset_to_arrays(source_sessions[idx])
                    for idx in (3, 4)
                ]
                source_X = np.concatenate([arr[0] for arr in source_train], axis=0)
                source_y = np.concatenate([arr[1] for arr in source_train], axis=0)
                source_X_val = np.concatenate([arr[0] for arr in source_val], axis=0)
                source_y_val = np.concatenate([arr[1] for arr in source_val], axis=0)
                source_X, source_X_val = BaseDataModule._riemannian_align_many(
                    source_X, source_X_val
                )
                aligned_train_arrays.append((source_X, source_y))
                aligned_val_arrays.append((source_X_val, source_y_val))
            train_arrays = aligned_train_arrays
            val_arrays = aligned_val_arrays
            X_target, X_test = BaseDataModule._riemannian_align_many(
                X_target, X_test
            )

        X = np.concatenate([arr[0] for arr in train_arrays], axis=0)
        y = np.concatenate([arr[1] for arr in train_arrays], axis=0)
        X_val = np.concatenate([arr[0] for arr in val_arrays], axis=0)
        y_val = np.concatenate([arr[1] for arr in val_arrays], axis=0)

        # scale data
        if self.preprocessing_dict["z_scale"]:
            X, X_val, X_target, X_test = BaseDataModule._z_scale_many(
                X, X_val, X_target, X_test
            )

        self.train_dataset = BaseDataModule._make_tensor_dataset(X, y)
        self.val_dataset = BaseDataModule._make_tensor_dataset(X_val, y_val)
        self.target_dataset = BaseDataModule._make_unlabeled_dataset(X_target)
        self.test_dataset = BaseDataModule._make_tensor_dataset(X_test, y_test)
        self._setup_complete = True
        self.dataset = None

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset,
                          batch_size=self.preprocessing_dict["batch_size"])
