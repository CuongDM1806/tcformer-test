from typing import Dict, Optional

import numpy as np
import pytorch_lightning as pl
from scipy.linalg import eigh
from sklearn.covariance import OAS
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset
import os

from utils.interaug import interaug


class InterAugCollate:
    # Module-level callable so DataLoader workers can pickle it under Windows spawn.
    def __init__(self, preproc):
        self.preproc = preproc

    def __call__(self, batch):
        xs, ys = zip(*batch)
        x = torch.stack(xs)
        y = torch.tensor(ys, dtype=torch.long)
        if self.preproc.get("interaug", False):
            x, y = interaug([x, y])
        return x, y


def make_collate_fn(preproc):
    return InterAugCollate(preproc)


class BaseDataModule(pl.LightningDataModule):
    dataset = None
    train_dataset = None
    test_dataset = None
    target_dataset = None

    def __init__(self, preprocessing_dict: Dict, subject_id: int):
        super(BaseDataModule, self).__init__()
        self.preprocessing_dict = preprocessing_dict
        self.subject_id = subject_id

    def prepare_data(self) -> None:
        raise NotImplementedError

    def setup(self, stage: Optional[str] = None) -> None:
        raise NotImplementedError

    def _source_train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset,
                          batch_size=self.preprocessing_dict["batch_size"],
                          shuffle=True,
                          num_workers=self.preprocessing_dict.get("num_workers", os.cpu_count() // 2),
                          pin_memory=True,
                          persistent_workers=True,          # ↩︎ keeps workers alive between epochs
                          prefetch_factor=4,                 # ↩︎ each worker preloads 4 future batches                          
                          collate_fn=make_collate_fn(self.preprocessing_dict)  # 👈 new
                    )

    def _target_train_dataloader(self) -> DataLoader:
        return DataLoader(
            UnlabeledDataset(self.target_dataset),
            batch_size=self.preprocessing_dict["batch_size"],
            shuffle=True,
            num_workers=self.preprocessing_dict.get("num_workers", os.cpu_count() // 2),
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
        )

    def train_dataloader(self):
        source_loader = self._source_train_dataloader()
        if not self.preprocessing_dict.get("domain_adaptation", False):
            return source_loader
        if self.target_dataset is None:
            raise RuntimeError(
                "Domain adaptation is enabled, but no unlabeled target split is available."
            )
        return {"source": source_loader, "target": self._target_train_dataloader()}

    def val_dataloader(self) -> DataLoader:
        return self.test_dataloader()

    def test_dataloader(self) -> DataLoader:
        # Test sets are small.  On Windows, worker processes can duplicate a
        # substantial part of the parent process and cause a RAM spike after
        # training, so testing is single-process unless explicitly overridden.
        test_num_workers = self.preprocessing_dict.get("test_num_workers", 0)
        return DataLoader(self.test_dataset,
                          batch_size=self.preprocessing_dict["batch_size"],
                          num_workers=test_num_workers,
                          pin_memory=True,
                          persistent_workers=test_num_workers > 0,
                          **({"prefetch_factor": 2} if test_num_workers > 0 else {}),
                        )

    @staticmethod
    # Method 1 (per-channel & per-timepoint) across samples
    # def _z_scale(X, X_test):
    #     for ch_idx in range(X.shape[1]):
    #         sc = StandardScaler()
    #         X[:, ch_idx, :] = sc.fit_transform(X[:, ch_idx, :])
    #         X_test[:, ch_idx, :] = sc.transform(X_test[:, ch_idx, :])
    #     return X, X_test
    # Method 2 Per-channel across all samples and timepoints
    def _z_scale(X, X_test):
        return BaseDataModule._z_scale_many(X, X_test)

    # @staticmethod
    # # Method 1 (per-channel & per-timepoint) across samples
    # def _z_scale_tvt(X_train, X_val, X_test):
    #     for ch in range(X_train.shape[1]):
    #         sc = StandardScaler()
    #         X_train[:, ch, :] = sc.fit_transform(X_train[:, ch, :])
    #         X_val[:, ch, :] = sc.transform(X_val[:, ch, :])
    #         X_test[:, ch, :] = sc.transform(X_test[:, ch, :])
    #     return X_train, X_val, X_test

    # Method 2 Per-channel across all samples and timepoints
    def _z_scale_tvt(X, X_val, X_test):
        return BaseDataModule._z_scale_many(X, X_val, X_test)

    @staticmethod
    def _z_scale_many(X_train, *other_arrays):
        """Fit on labeled source training data and transform every other split."""
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
    def _spd_power(matrix, exponent, epsilon=1e-8):
        """Raise a symmetric positive-definite matrix to a real power."""
        eigenvalues, eigenvectors = eigh(matrix)
        eigenvalues = np.maximum(eigenvalues, epsilon)
        return (eigenvectors * (eigenvalues ** exponent)) @ eigenvectors.T

    @staticmethod
    def _riemannian_mean(covariances, tolerance=1e-7, max_iterations=50):
        """Compute the affine-invariant Riemannian mean of SPD matrices."""
        mean = np.mean(covariances, axis=0)
        for _ in range(max_iterations):
            mean_sqrt = BaseDataModule._spd_power(mean, 0.5)
            mean_invsqrt = BaseDataModule._spd_power(mean, -0.5)
            tangent = np.zeros_like(mean)

            for covariance in covariances:
                centered = mean_invsqrt @ covariance @ mean_invsqrt
                eigenvalues, eigenvectors = eigh(centered)
                eigenvalues = np.maximum(eigenvalues, 1e-8)
                tangent += (
                    eigenvectors * np.log(eigenvalues)
                ) @ eigenvectors.T

            tangent /= len(covariances)
            if np.linalg.norm(tangent, ord="fro") < tolerance:
                break

            eigenvalues, eigenvectors = eigh(tangent)
            tangent_exp = (
                eigenvectors * np.exp(eigenvalues)
            ) @ eigenvectors.T
            mean = mean_sqrt @ tangent_exp @ mean_sqrt
            mean = 0.5 * (mean + mean.T)

        return mean

    @staticmethod
    def _riemannian_whitener(X):
        """Fit an OAS/Riemannian reference and return its inverse square root."""
        covariances = np.stack(
            [OAS().fit(trial.T.astype(np.float64)).covariance_ for trial in X],
            axis=0,
        )
        reference = BaseDataModule._riemannian_mean(covariances)
        return BaseDataModule._spd_power(reference, -0.5)

    @staticmethod
    def _riemannian_align_many(X_reference, *other_arrays):
        """Fit RA on reference trials and apply the same whitening to all splits."""
        whitener = BaseDataModule._riemannian_whitener(X_reference)

        def transform(array):
            aligned = np.einsum("cd,ndt->nct", whitener, array, optimize=True)
            return aligned.astype(array.dtype, copy=False)

        return (
            transform(X_reference),
            *(transform(array) for array in other_arrays),
        )
    
    @staticmethod
    def _make_tensor_dataset(X, y):
        return TensorDataset(torch.Tensor(X), torch.Tensor(y).type(torch.LongTensor))
        # return TensorDataset(torch.tensor(X), torch.tensor(y).long())

    @staticmethod
    def _make_unlabeled_dataset(X):
        return TensorDataset(torch.Tensor(X))

    @staticmethod
    def _dataset_to_arrays(dataset):
        if hasattr(dataset, "datasets"):
            arrays = [BaseDataModule._dataset_to_arrays(ds) for ds in dataset.datasets]
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


class UnlabeledDataset(torch.utils.data.Dataset):
    """Expose EEG only, so target labels cannot enter the training step."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        return item[0] if isinstance(item, (tuple, list)) else item
        
    # @staticmethod
    # def _make_tensor_dataset(X, y, preprocessing_dict=None, mode="train"):
    #     if preprocessing_dict and mode == "train":
    #         return AugmentedTensorDataset(
    #             X, y,
    #             interaug=preprocessing_dict.get("interaug", False),
    #         )
    #     return TensorDataset(torch.tensor(X), torch.tensor(y).long())


# from utils.interaug import interaug
# class AugmentedTensorDataset(TensorDataset):
#     def __init__(self, X, y, interaug=False):
#         super().__init__(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
#         self.interaug = interaug

#     def __getitem__(self, index):
#         x, y = super().__getitem__(index)
        
#         if self.interaug:
#             x, y = interaug([x, y])

#         return x, y
