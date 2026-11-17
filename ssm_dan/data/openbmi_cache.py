"""OpenBMI (Lee2019_MI) loading with a per-subject on-disk cache.

MOABB's ``Lee2019_MI`` exposes the labeled offline MI run of each of the two
recording sessions (100 trials each, left vs. right hand); the online feedback
runs are excluded. The raw MAT files total about 61 GB for 54 subjects, so each
subject is downloaded, preprocessed and written to a small ``.npz`` file, after
which its raw files can be deleted. Training reads only these files, from
data/openbmi (CAR) or data/openbmi_nocar (see ssm_dan.data.files); they carry
the preprocessing they were built with and are rebuilt when it differs.
"""
import json
import os
import time
from pathlib import Path
from typing import Dict

import numpy as np

from . import files

OPENBMI_MI_MAPPING = {
    "left_hand": 0,
    "right_hand": 1,
}
OPENBMI_CHANNELS = 62
# MOABB 0.5 downloads Lee2019 from ftp://parrot.genomics.cn, which Kaggle
# cannot reach (connection timeout). Newer MOABB releases use GigaDB's HTTPS
# mirror, which serves the same files; OPENBMI_URL overrides it.
LEE2019_HTTPS_URL = ("https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/"
                     "pub/10.5524/100001_101000/100542/")
# Preprocessing entries that change the cached arrays.
CACHE_KEYS = ("sfreq", "low_cut", "high_cut", "common_average_reference",
              "start", "trial_duration")


def _scale(data, factor):
    return data * factor


def cache_signature(preprocessing_dict: Dict) -> str:
    return json.dumps({key: preprocessing_dict.get(key) for key in CACHE_KEYS},
                      sort_keys=True)


def _cache_name(subject_id: int) -> str:
    return f"openbmi_S{subject_id:02d}.npz"


def _cache_dirs(preprocessing_dict: Dict):
    """data/openbmi or data/openbmi_nocar, as a (write, [read]) pair."""
    write_dir = files.openbmi_dir(no_car=not preprocessing_dict.get("common_average_reference", True))
    return write_dir, [write_dir]


def _find_cached(subject_id: int, signature: str, read_dirs):
    name = _cache_name(subject_id)
    for directory in read_dirs:
        if not directory.is_dir():
            continue
        candidates = [directory / name] if (directory / name).is_file() else \
            sorted(directory.rglob(name))
        for path in candidates:
            with np.load(path, allow_pickle=False) as data:
                if str(data["signature"]) == signature:
                    return path
            print(f"  ignoring {path}: built with different preprocessing", flush=True)
    return None


def _use_https_mirror() -> None:
    import moabb.datasets.Lee2019 as lee2019
    lee2019.Lee2019_URL = os.environ.get("OPENBMI_URL", LEE2019_HTTPS_URL)


def _download_windows(subject_id: int, preprocessing_dict: Dict, verbose: str):
    # Imported here: reading an existing cache needs neither MOABB nor braindecode.
    from braindecode.datasets import MOABBDataset
    from braindecode.preprocessing import (
        Preprocessor,
        create_windows_from_events,
        preprocess,
    )
    from requests.exceptions import RequestException

    _use_https_mirror()
    download_retries = int(preprocessing_dict.get("download_retries", 8))
    retry_delay = float(preprocessing_dict.get("download_retry_delay_seconds", 5.0))
    if download_retries < 0 or retry_delay < 0:
        raise ValueError("OpenBMI download retry settings must be non-negative.")
    for attempt in range(download_retries + 1):
        try:
            dataset = MOABBDataset(dataset_name="Lee2019_MI", subject_ids=[subject_id])
            break
        except (RequestException, OSError) as exc:      # HTTP errors, timeouts, resets
            if attempt == download_retries:
                raise
            wait_seconds = min(retry_delay * (2 ** attempt), 60.0)
            print(f"OpenBMI download failed ({type(exc).__name__}: {exc}). "
                  f"Retry {attempt + 1}/{download_retries} in {wait_seconds:g} s; "
                  "cached MAT files will be reused.", flush=True)
            time.sleep(wait_seconds)

    preprocessors = [
        Preprocessor("pick_types", eeg=True, meg=False, stim=False, verbose=verbose),
        Preprocessor(_scale, factor=1e6, apply_on_array=True),
    ]
    low_cut = preprocessing_dict.get("low_cut")
    high_cut = preprocessing_dict.get("high_cut")
    if low_cut is not None or high_cut is not None:
        preprocessors.append(
            Preprocessor("filter", l_freq=low_cut, h_freq=high_cut, verbose=verbose))
    if preprocessing_dict.get("common_average_reference", True):
        preprocessors.append(Preprocessor(
            "set_eeg_reference", ref_channels="average", ch_type="eeg",
            projection=False, verbose=verbose))
    preprocessors.append(
        Preprocessor("resample", sfreq=preprocessing_dict["sfreq"], verbose=verbose))
    preprocess(dataset, preprocessors)

    sfreq = dataset.datasets[0].raw.info["sfreq"]
    start = float(preprocessing_dict.get("start", 0.0))
    duration = float(preprocessing_dict.get("trial_duration", 4.0))
    return create_windows_from_events(
        dataset,
        trial_start_offset_samples=int(round(start * sfreq)),
        trial_stop_offset_samples=0,
        window_size_samples=int(round(duration * sfreq)),
        window_stride_samples=int(round(duration * sfreq)),
        drop_last_window=True,
        mapping=OPENBMI_MI_MAPPING,
        preload=False,
        on_missing="ignore",
    )


def _ordered_sessions(windows_dataset):
    sessions = windows_dataset.split("session")

    def sort_key(item):
        digits = "".join(c for c in str(item[0]) if c.isdigit())
        return (int(digits) if digits else 999, str(item[0]))

    ordered = [dataset for _, dataset in sorted(sessions.items(), key=sort_key)]
    if len(ordered) != 2:
        raise RuntimeError(f"OpenBMI MI requires exactly two sessions, got {list(sessions)}.")
    return ordered


def _delete_raw_files(subject_id: int) -> None:
    """Remove the subject's downloaded MAT files once its cache is written."""
    try:
        from moabb.datasets import Lee2019_MI
        _use_https_mirror()                          # same URLs -> same local paths
        paths = Lee2019_MI().data_path(subject_id)
    except Exception as exc:                       # cache is already safe on disk
        print(f"  could not locate raw OpenBMI files of S{subject_id:02d}: {exc}", flush=True)
        return
    freed = 0
    for path in paths:
        path = Path(path)
        if path.is_file():
            freed += path.stat().st_size
            path.unlink()
    print(f"  deleted raw files of S{subject_id:02d} ({freed / 1e9:.2f} GB)", flush=True)


def validate_session(subject_id, session_id, X, y, timepoints):
    classes, counts = np.unique(y, return_counts=True)
    if X.ndim != 3 or X.shape[1:] != (OPENBMI_CHANNELS, timepoints):
        raise RuntimeError(f"OpenBMI S{subject_id:02d} session {session_id} has shape "
                           f"{tuple(X.shape)}; expected [trials, {OPENBMI_CHANNELS}, {timepoints}].")
    if not np.array_equal(classes, np.arange(len(OPENBMI_MI_MAPPING))):
        raise RuntimeError(f"OpenBMI S{subject_id:02d} session {session_id} has classes "
                           f"{classes.tolist()}, expected [0, 1].")
    return counts


def load_openbmi_subject(subject_id: int, preprocessing_dict: Dict,
                         verbose: str = "WARNING"):
    """((X_session_1, y_session_1), (X_session_2, y_session_2)) of one subject.

    X is float32 [trials, 62, sfreq * trial_duration] in microvolts, y int64.
    """
    from .base import BaseDataModule

    files.configure()
    signature = cache_signature(preprocessing_dict)
    timepoints = int(round(preprocessing_dict["sfreq"]
                           * float(preprocessing_dict.get("trial_duration", 4.0))))
    write_dir, read_dirs = _cache_dirs(preprocessing_dict)
    cached = _find_cached(subject_id, signature, read_dirs)
    if cached is None:
        print(f"Downloading and preprocessing OpenBMI MI S{subject_id:02d}", flush=True)
        windows = _download_windows(subject_id, preprocessing_dict, verbose)
        sessions = [BaseDataModule.to_arrays(s) for s in _ordered_sessions(windows)]
        del windows
        arrays = {}
        for index, (X, y) in enumerate(sessions, start=1):
            arrays[f"X{index}"] = X.astype(np.float32, copy=False)
            arrays[f"y{index}"] = y.astype(np.int64, copy=False)
        write_dir.mkdir(parents=True, exist_ok=True)
        cached = write_dir / _cache_name(subject_id)
        temporary = cached.with_name(cached.name + ".tmp.npz")
        np.savez(temporary, signature=np.array(signature), **arrays)
        temporary.replace(cached)
        if preprocessing_dict.get("delete_raw_after_cache", True):
            _delete_raw_files(subject_id)

    with np.load(cached, allow_pickle=False) as data:
        sessions = [(data["X1"], data["y1"]), (data["X2"], data["y2"])]
    for session_id, (X, y) in enumerate(sessions, start=1):
        counts = validate_session(subject_id, session_id, X, y, timepoints)
        print(f"  S{subject_id:02d} session={session_id} | trials={len(y)} | "
              f"class_counts={counts.tolist()} | shape={tuple(X.shape)} | {cached}", flush=True)
    return sessions[0], sessions[1]


# --------------------------------------------------------------------------- download
def download(subjects, no_car=False):
    """Download, preprocess and save the given subjects into data/openbmi[_nocar] (skips existing ones).

    The preprocessing is the ``openbmi`` block of configs/datasets.yaml (with the
    ``no_car`` variant when ``no_car``).
    """
    import shutil
    from .. import config as cfg

    settings = cfg.resolve("ssm_dan", "openbmi", ["no_car"] if no_car else [])["preprocessing"]
    out, _ = _cache_dirs(settings)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    for index, subject in enumerate(subjects, start=1):
        load_openbmi_subject(subject, settings)
        print(f"[{index}/{len(subjects)}] S{subject:02d} cached | {(time.time() - started) / 60:.1f} min | "
              f"free disk {shutil.disk_usage(out).free / 1e9:.1f} GB", flush=True)


def main(argv=None):
    """python -m ssm_dan.data.openbmi_cache [--no-car] [--subjects 1-54]"""
    import argparse
    from ..config import parse_subjects

    parser = argparse.ArgumentParser(prog="python -m ssm_dan.data.openbmi_cache",
                                     description="Download and cache OpenBMI (Lee2019_MI).")
    parser.add_argument("--subjects", default="1-54")
    parser.add_argument("--no-car", action="store_true", help="skip the common average reference")
    args = parser.parse_args(argv)
    download(parse_subjects(args.subjects, list(range(1, 55))), args.no_car)


if __name__ == "__main__":
    main()
