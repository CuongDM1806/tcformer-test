from typing import Optional

import numpy as np
from torch.utils.data import DataLoader

from .base import BaseDataModule
from utils.load_zhou2016 import load_zhou2016


def _ordered_session_items(split_by_session):
    def sort_key(item):
        key, _ = item
        digits = "".join(character for character in str(key) if character.isdigit())
        return (int(digits) if digits else 999, str(key))

    return sorted(split_by_session.items(), key=sort_key)


def _get_three_sessions(subject_dataset):
    split_by_session = subject_dataset.split("session")
    sessions = [dataset for _, dataset in _ordered_session_items(split_by_session)]
    if len(sessions) != 3:
        raise RuntimeError(
            "Zhou2016 must contain exactly three sessions per subject; "
            f"found keys {list(split_by_session)}."
        )
    return sessions


class Zhou2016LOSO(BaseDataModule):
    """Chronological subject-independent Zhou2016 split.

    Sessions 1--2 of every source subject are labeled training data and
    session 3 is labeled source-only validation data. For the held-out target,
    sessions 1--2 form the optional unlabeled adaptation partition and are
    consumed only when domain adaptation is enabled. Session 3 is reserved
    for optional label-free test-time adaptation and primary scoring. A separate
    all-three-session loader is retained only as an explicitly auxiliary view.
    """

    all_subject_ids = list(range(1, 5))
    class_names = ["hand(L)", "hand(R)", "feet"]
    channels = 14
    classes = 3
    primary_test_label = "SESSION 3"
    auxiliary_test_label = "ALL 3 SESSIONS"
    all_sessions_test_label = auxiliary_test_label

    def __init__(self, preprocessing_dict: dict, subject_id: int):
        super().__init__(preprocessing_dict, subject_id)
        if subject_id not in self.all_subject_ids:
            raise ValueError(
                f"Zhou2016 target subject must be in 1..4, got {subject_id}."
            )
        self._setup_complete = False

    def prepare_data(self) -> None:
        if self._setup_complete or self.dataset is not None:
            return
        self.dataset = load_zhou2016(
            self.all_subject_ids, self.preprocessing_dict
        )

    @staticmethod
    def _subject_dataset(split_by_subject, subject_id):
        dataset = split_by_subject.get(str(subject_id))
        if dataset is None:
            dataset = split_by_subject.get(subject_id)
        if dataset is None:
            raise KeyError(
                f"Zhou2016 subject {subject_id} is absent; available keys: "
                f"{list(split_by_subject)}"
            )
        return dataset

    def setup(self, stage: Optional[str] = None) -> None:
        if self._setup_complete:
            return
        if self.dataset is None:
            self.prepare_data()

        split_by_subject = self.dataset.split("subject")
        source_ids = [
            subject_id
            for subject_id in self.all_subject_ids
            if subject_id != self.subject_id
        ]

        source_train_arrays = []
        source_val_arrays = []
        for source_id in source_ids:
            sessions = _get_three_sessions(
                self._subject_dataset(split_by_subject, source_id)
            )
            session_1 = BaseDataModule._dataset_to_arrays(sessions[0])
            session_2 = BaseDataModule._dataset_to_arrays(sessions[1])
            session_3 = BaseDataModule._dataset_to_arrays(sessions[2])
            X_source_train = np.concatenate(
                [session_1[0], session_2[0]], axis=0
            )
            y_source_train = np.concatenate(
                [session_1[1], session_2[1]], axis=0
            )
            X_source_val, y_source_val = session_3

            if self.preprocessing_dict.get("riemannian_alignment", False):
                X_source_train, X_source_val = (
                    BaseDataModule._riemannian_align_many(
                        X_source_train, X_source_val
                    )
                )
            source_train_arrays.append((X_source_train, y_source_train))
            source_val_arrays.append((X_source_val, y_source_val))

        target_sessions = _get_three_sessions(
            self._subject_dataset(split_by_subject, self.subject_id)
        )
        target_session_arrays = [
            BaseDataModule._dataset_to_arrays(session) for session in target_sessions
        ]
        X_target = np.concatenate(
            [target_session_arrays[0][0], target_session_arrays[1][0]], axis=0
        )
        y_target_aux = np.concatenate(
            [target_session_arrays[0][1], target_session_arrays[1][1]], axis=0
        )
        X_test, y_test = target_session_arrays[2]
        if self.preprocessing_dict.get("riemannian_alignment", False):
            # Fit only on the earlier target-adaptation sessions, then apply
            # the frozen transform to the held-out evaluation session.
            X_target, X_test = BaseDataModule._riemannian_align_many(
                X_target, X_test
            )

        X_train = np.concatenate(
            [item[0] for item in source_train_arrays], axis=0
        )
        y_train = np.concatenate(
            [item[1] for item in source_train_arrays], axis=0
        )
        X_val = np.concatenate(
            [item[0] for item in source_val_arrays], axis=0
        )
        y_val = np.concatenate(
            [item[1] for item in source_val_arrays], axis=0
        )

        expected_timepoints = int(
            round(
                self.preprocessing_dict["sfreq"]
                * self.preprocessing_dict.get("trial_duration", 5.0)
            )
        )
        for split_name, array in (
            ("source train", X_train),
            ("source validation", X_val),
            ("target adaptation", X_target),
            ("target test", X_test),
            ("auxiliary target sessions", np.concatenate([X_target, X_test])),
        ):
            if array.ndim != 3 or array.shape[1:] != (
                self.channels,
                expected_timepoints,
            ):
                raise RuntimeError(
                    f"Zhou2016 {split_name} has shape {tuple(array.shape)}; "
                    f"expected [trials, {self.channels}, {expected_timepoints}]."
                )

        if self.preprocessing_dict.get("z_scale", False):
            X_train, X_val, X_target, X_test = BaseDataModule._z_scale_many(
                X_train, X_val, X_target, X_test
            )
        X_all_target = np.concatenate([X_target, X_test], axis=0)
        y_all_target = np.concatenate([y_target_aux, y_test], axis=0)

        self.train_dataset = BaseDataModule._make_tensor_dataset(X_train, y_train)
        self.val_dataset = BaseDataModule._make_tensor_dataset(X_val, y_val)
        self.target_dataset = BaseDataModule._make_unlabeled_dataset(X_target)
        self.test_dataset = BaseDataModule._make_tensor_dataset(X_test, y_test)
        self.all_target_dataset = BaseDataModule._make_tensor_dataset(
            X_all_target, y_all_target
        )
        self._setup_complete = True
        self.dataset = None

        print(
            f"Zhou2016 LOSO target S{self.subject_id:02d} | "
            f"5.0 s ({expected_timepoints} samples) | "
            f"source_train={len(y_train)} | source_val={len(y_val)} | "
            f"target_adaptation={len(X_target)} | "
            f"target_session_3_test={len(y_test)} | "
            f"target_all_sessions_aux={len(y_all_target)}",
            flush=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Validate on session 3 from source subjects only."""
        num_workers = self.preprocessing_dict.get("test_num_workers", 0)
        return DataLoader(
            self.val_dataset,
            batch_size=self.preprocessing_dict["batch_size"],
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            **({"prefetch_factor": 2} if num_workers > 0 else {}),
        )
