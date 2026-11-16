"""Shared LOSO data-module logic: loaders, z-scoring and array helpers.

Every dataset module builds five tensor datasets in ``setup``:

- ``train_dataset``: labeled source training trials
- ``val_dataset``: labeled source validation trials (checkpoint selection)
- ``target_dataset``: unlabeled target trials for Stage-1 adaptation
- ``test_dataset``: target evaluation trials (labels used only for scoring)
- ``all_target_dataset``: optional auxiliary view over every target session
"""
import os
from typing import Dict, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, TensorDataset

from .interaug import interaug


class InterAugCollate:
    """Collate labeled source trials and apply S&R augmentation if enabled.

    A module-level class so that DataLoader workers can pickle it.
    """

    def __init__(self, preprocessing):
        self.preprocessing = preprocessing

    def __call__(self, batch):
        xs, ys = zip(*batch)
        x = torch.stack(xs)
        y = torch.tensor(ys, dtype=torch.long)
        if self.preprocessing.get("interaug", False):
            x, y = interaug([x, y])
        return x, y


class UnlabeledDataset(Dataset):
    """Expose EEG only, so target labels cannot enter the training step."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        return item[0] if isinstance(item, (tuple, list)) else item


class BaseDataModule(pl.LightningDataModule):
    # Set by every dataset module.
    all_subject_ids: list = []
    class_names: list = []
    channels: int = 0
    classes: int = 0
    primary_test_label = "EVALUATION SESSION"
    auxiliary_test_label = "ALL TARGET SESSIONS"

    def __init__(self, preprocessing: Dict, subject_id: int):
        super().__init__()
        self.preprocessing = preprocessing
        self.subject_id = subject_id
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.target_dataset = None
        self.test_dataset = None
        self.all_target_dataset = None
        self._setup_complete = False

    # ------------------------------------------------------------------ loaders
    def _source_train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.preprocessing["batch_size"],
            shuffle=True,
            num_workers=self.preprocessing.get("num_workers", os.cpu_count() // 2),
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
            collate_fn=InterAugCollate(self.preprocessing),
        )

    def _target_train_dataloader(self) -> DataLoader:
        return DataLoader(
            UnlabeledDataset(self.target_dataset),
            batch_size=self.preprocessing["batch_size"],
            shuffle=True,
            num_workers=self.preprocessing.get("num_workers", os.cpu_count() // 2),
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
        )

    def train_dataloader(self):
        source_loader = self._source_train_dataloader()
        if not self.preprocessing.get("domain_adaptation", False):
            return source_loader
        if self.target_dataset is None:
            raise RuntimeError(
                "Domain adaptation is enabled, but no unlabeled target split is available."
            )
        return {"source": source_loader, "target": self._target_train_dataloader()}

    def _evaluation_loader(self, dataset, batch_size=None) -> DataLoader:
        # Evaluation sets are small and deterministic; single-process loading
        # avoids duplicating the parent process in worker processes.
        num_workers = self.preprocessing.get("test_num_workers", 0)
        return DataLoader(
            dataset,
            batch_size=batch_size or self.preprocessing["batch_size"],
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            **({"prefetch_factor": 2} if num_workers > 0 else {}),
        )

    def val_dataloader(self) -> DataLoader:
        """Labeled source validation trials only (checkpoint selection)."""
        if self.val_dataset is None:
            raise RuntimeError("No source validation split is available.")
        return self._evaluation_loader(self.val_dataset)

    def test_dataloader(self) -> DataLoader:
        """Target evaluation trials; also the unlabeled IM-TTA input."""
        return self._evaluation_loader(self.test_dataset)

    def all_target_dataloader(self) -> DataLoader:
        """Auxiliary view over every target session."""
        return self._evaluation_loader(self.all_target_dataset)

    # ------------------------------------------------------------ array helpers
    @staticmethod
    def z_scale(X_train, *other_arrays):
        """Per-channel z-score fitted on the labeled source training trials."""
        _, channels, timepoints = X_train.shape
        train_2d = X_train.transpose(1, 0, 2).reshape(channels, -1).T
        scaler = StandardScaler().fit(train_2d)

        def transform(array):
            samples = array.shape[0]
            array_2d = array.transpose(1, 0, 2).reshape(channels, -1).T
            return scaler.transform(array_2d).T.reshape(
                channels, samples, timepoints
            ).transpose(1, 0, 2)

        return (transform(X_train), *(transform(array) for array in other_arrays))

    @staticmethod
    def tensor_dataset(X, y):
        return TensorDataset(torch.Tensor(X), torch.Tensor(y).type(torch.LongTensor))

    @staticmethod
    def unlabeled_dataset(X):
        return TensorDataset(torch.Tensor(X))

    @staticmethod
    def to_arrays(dataset):
        """(X, y) arrays of a Braindecode windows dataset (or a concat of them)."""
        if hasattr(dataset, "datasets"):
            arrays = [BaseDataModule.to_arrays(ds) for ds in dataset.datasets]
            X = np.concatenate([arr[0] for arr in arrays], axis=0)
            y = np.concatenate([arr[1] for arr in arrays], axis=0)
            return X, y

        if hasattr(dataset, "windows"):
            return dataset.windows.load_data()._data, np.array(dataset.y)

        X, y = [], []
        for idx in range(len(dataset)):
            item = dataset[idx]
            if isinstance(item, dict):
                x_item = item.get("X", item.get("x", item.get("data")))
                y_item = item.get("y", item.get("target", item.get("label")))
            else:
                x_item, y_item = item[0], item[1]
            if torch.is_tensor(x_item):
                x_item = x_item.detach().cpu().numpy()
            if torch.is_tensor(y_item):
                y_item = y_item.detach().cpu().numpy()
            X.append(x_item)
            y.append(y_item)
        return np.stack(X, axis=0), np.asarray(y)

    def _build(self, X, y, X_val, y_val, X_target, X_test, y_test,
               y_target=None) -> None:
        """z-score with source-training statistics and create the datasets."""
        if self.preprocessing.get("z_scale", False):
            X, X_val, X_target, X_test = self.z_scale(X, X_val, X_target, X_test)
        self.train_dataset = self.tensor_dataset(X, y)
        self.val_dataset = self.tensor_dataset(X_val, y_val)
        self.target_dataset = self.unlabeled_dataset(X_target)
        self.test_dataset = self.tensor_dataset(X_test, y_test)
        if y_target is not None:
            self.all_target_dataset = self.tensor_dataset(
                np.concatenate((X_target, X_test), axis=0),
                np.concatenate((y_target, y_test), axis=0),
            )

    def prepare_data(self) -> None:
        raise NotImplementedError

    def setup(self, stage: Optional[str] = None) -> None:
        raise NotImplementedError


def ordered_sessions(windows_dataset):
    """Sessions of a Braindecode dataset sorted by the number in their name."""
    split = windows_dataset.split("session")

    def sort_key(item):
        key, _ = item
        digits = "".join(ch for ch in str(key) if ch.isdigit())
        return (int(digits) if digits else 999, str(key))

    return [dataset for _, dataset in sorted(split.items(), key=sort_key)]
