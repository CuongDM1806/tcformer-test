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

from utils.interaug import interaug, interaug_vectorized


class InterAugCollate:
    # Module-level callable so DataLoader workers can pickle it under Windows spawn.
    def __init__(self, preproc):
        self.preproc = preproc

    def __call__(self, batch):
        xs, ys = zip(*batch)
        x = torch.stack(xs)
        y = torch.tensor(ys, dtype=torch.long)
        if self.preproc.get("interaug", False):
            # "vectorized" gives the same output as "loop" under the same seed.
            if self.preproc.get("interaug_impl", "loop") == "vectorized":
                x, y = interaug_vectorized([x, y])
            else:
                x, y = interaug([x, y])
        return x, y


def make_collate_fn(preproc):
    return InterAugCollate(preproc)


class SubjectInterAugCollate:
    """Collate (x, y, subject) items; S&R augmentation stays within one subject.

    Segments are recombined only among same-class trials of the same subject,
    so every augmented trial still belongs to exactly one subject and can use
    that subject's predicted time scale.
    """

    def __init__(self, preproc):
        self.preproc = preproc

    def __call__(self, batch):
        xs, ys, ss = zip(*batch)
        x, y = torch.stack(xs), torch.as_tensor(ys, dtype=torch.long)
        s = torch.as_tensor(ss, dtype=torch.long)
        if not self.preproc.get("interaug", False):
            return x, y, s
        augment = interaug_vectorized if self.preproc.get("interaug_impl", "loop") == "vectorized" else interaug
        parts = []
        for subject in torch.unique(s):
            members = s == subject
            x_aug, y_aug = augment([x[members], y[members]])
            parts.append((x_aug, y_aug, torch.full_like(y_aug, int(subject))))
        x = torch.cat([p[0] for p in parts])
        y = torch.cat([p[1] for p in parts])
        s = torch.cat([p[2] for p in parts])
        order = torch.randperm(len(x))
        return x[order], y[order], s[order]


class SubjectBatchSampler(torch.utils.data.Sampler):
    """Episodic batches: `subjects_per_batch` subjects x `trials_per_subject` trials.

    Every trial is used at most once per epoch; the random numbers come from
    torch, so the order follows the global seed.
    """

    def __init__(self, subject_ids, subjects_per_batch: int, trials_per_subject: int):
        self.subject_ids = np.asarray(subject_ids)
        self.subjects_per_batch = int(subjects_per_batch)
        self.trials_per_subject = int(trials_per_subject)
        self.by_subject = {
            int(subject): np.flatnonzero(self.subject_ids == subject)
            for subject in np.unique(self.subject_ids)
        }
        if len(self.by_subject) < self.subjects_per_batch:
            raise ValueError("Fewer source subjects than subjects_per_batch.")

    def __len__(self):
        chunks = sum(len(i) // self.trials_per_subject for i in self.by_subject.values())
        return chunks // self.subjects_per_batch

    def __iter__(self):
        queues = {
            subject: list(indices[torch.randperm(len(indices)).numpy()])
            for subject, indices in self.by_subject.items()
        }
        for _ in range(len(self)):
            available = [s for s, q in queues.items() if len(q) >= self.trials_per_subject]
            if len(available) < self.subjects_per_batch:
                return
            # Prefer subjects with more trials left so the epoch ends balanced.
            available.sort(key=lambda s: (-len(queues[s]), float(torch.rand(()))))
            chosen = available[: self.subjects_per_batch]
            batch = []
            for subject in chosen:
                batch.extend(int(queues[subject].pop()) for _ in range(self.trials_per_subject))
            yield batch


class BaseDataModule(pl.LightningDataModule):
    dataset = None
    train_dataset = None
    test_dataset = None
    target_dataset = None
    all_target_dataset = None

    def __init__(self, preprocessing_dict: Dict, subject_id: int):
        super(BaseDataModule, self).__init__()
        self.preprocessing_dict = preprocessing_dict
        self.subject_id = subject_id

    def prepare_data(self) -> None:
        raise NotImplementedError

    def setup(self, stage: Optional[str] = None) -> None:
        raise NotImplementedError

    def _source_train_dataloader(self) -> DataLoader:
        episodic = self.preprocessing_dict.get("episodic_subjects_per_batch")
        if episodic:
            # Amortized time-scale adaptation: subject-pure episodes.
            batch_size = self.preprocessing_dict["batch_size"]
            if batch_size % int(episodic) != 0:
                raise ValueError("batch_size must be divisible by episodic_subjects_per_batch.")
            return DataLoader(
                self.train_dataset,
                batch_sampler=SubjectBatchSampler(
                    self.train_subject_ids, int(episodic), batch_size // int(episodic)
                ),
                num_workers=self.preprocessing_dict.get("num_workers", os.cpu_count() // 2),
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=4,
                collate_fn=SubjectInterAugCollate(self.preprocessing_dict),
            )
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
        session_2_loader = self.test_dataloader()
        if self.all_target_dataset is None:
            return session_2_loader
        return [session_2_loader, self.all_target_dataloader()]

    def all_target_dataloader(self) -> DataLoader:
        """Evaluate the target subject on session 1 and session 2 together."""
        test_num_workers = self.preprocessing_dict.get("test_num_workers", 0)
        return DataLoader(
            self.all_target_dataset,
            batch_size=self.preprocessing_dict["batch_size"],
            num_workers=test_num_workers,
            pin_memory=True,
            persistent_workers=test_num_workers > 0,
            **({"prefetch_factor": 2} if test_num_workers > 0 else {}),
        )

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
    def _make_subject_dataset(X, y, subject_ids):
        """(x, y, subject id) triples for the episodic source split."""
        return TensorDataset(
            torch.Tensor(X), torch.Tensor(y).type(torch.LongTensor),
            torch.as_tensor(np.asarray(subject_ids), dtype=torch.long),
        )

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
