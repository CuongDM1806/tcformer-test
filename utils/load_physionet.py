import time
from typing import Dict

from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import (
    Preprocessor,
    create_windows_from_events,
    preprocess,
)
from requests.exceptions import RequestException


PHYSIONET_IMAGERY_MAPPING = {
    "left_hand": 0,
    "right_hand": 1,
    "hands": 2,
    "feet": 3,
}


def load_physionet(
    subject_ids: list,
    preprocessing_dict: Dict,
    verbose: str = "WARNING",
):
    """Load four-class motor-imagery trials from PhysioNet EEGMMIDB.

    Trials are kept in their native voltage unit with no frequency filtering.
    MOABB's ``PhysionetMI`` defaults to imagined (not executed) runs. The
    explicit mapping removes the inter-trial ``rest`` event and gives stable
    class indices across runs that contain different pairs of motor tasks.
    """
    download_retries = int(preprocessing_dict.get("download_retries", 8))
    retry_delay = float(
        preprocessing_dict.get("download_retry_delay_seconds", 5.0)
    )
    if download_retries < 0 or retry_delay < 0:
        raise ValueError("PhysioNet download retry settings must be non-negative.")

    for attempt in range(download_retries + 1):
        try:
            dataset = MOABBDataset(
                dataset_name="PhysionetMI",
                subject_ids=subject_ids,
            )
            break
        except RequestException as exc:
            if attempt == download_retries:
                raise
            wait_seconds = min(retry_delay * (2**attempt), 60.0)
            print(
                f"PhysioNet download failed ({type(exc).__name__}: {exc}). "
                f"Retry {attempt + 1}/{download_retries} in "
                f"{wait_seconds:g} s; cached EDF files will be reused.",
                flush=True,
            )
            time.sleep(wait_seconds)

    preprocessors = [
        Preprocessor(
            "pick_types", eeg=True, meg=False, stim=False, verbose=verbose
        ),
        Preprocessor(
            "resample", sfreq=preprocessing_dict["sfreq"], verbose=verbose
        ),
    ]

    preprocess(dataset, preprocessors)

    sfreq = dataset.datasets[0].raw.info["sfreq"]
    start_offset = int(round(preprocessing_dict.get("start", 0.0) * sfreq))
    trial_duration = float(preprocessing_dict.get("trial_duration", 4.0))
    window_size = int(round(trial_duration * sfreq))
    if start_offset != 0 or trial_duration != 4.0:
        raise ValueError(
            "Full-Mamba PhysioNet trials must cover 0.0 to 4.0 s from cue onset."
        )

    # Most annotations are about 4.1 s, but the final trial of some EDF runs is
    # only 4.0 s. The common 640-sample interval retains those trials instead
    # of dropping roughly 1/15 of the dataset via accepted_bads_ratio.
    # A few recordings in the 109-subject cohort end mid-trial (e.g. 476
    # samples); those truncated trials cannot form a 4.0 s epoch and are
    # removed explicitly so every remaining trial is complete.
    dropped = []
    for ds in dataset.datasets:
        raw = ds.raw
        raw_sfreq = raw.info["sfreq"]
        short = [
            index
            for index, (duration, description) in enumerate(
                zip(raw.annotations.duration, raw.annotations.description)
            )
            if description in PHYSIONET_IMAGERY_MAPPING
            # Same truncation braindecode uses to compute trial stops.
            and int(duration * raw_sfreq) < window_size
        ]
        if short:
            durations = [
                int(raw.annotations.duration[index] * raw_sfreq)
                for index in short
            ]
            raw.annotations.delete(short)
            dropped.append(
                f"S{int(ds.description['subject']):03d} "
                f"run {ds.description.get('run', '?')}: "
                f"{len(short)} trial(s) of {durations} samples"
            )
    if dropped:
        print(
            f"PhysioNet: dropped truncated trials (< {window_size} samples) "
            f"in {len(dropped)} run(s) -> "
            + "; ".join(dropped),
            flush=True,
        )

    return create_windows_from_events(
        dataset,
        trial_start_offset_samples=start_offset,
        trial_stop_offset_samples=0,
        window_size_samples=window_size,
        window_stride_samples=window_size,
        drop_last_window=True,
        mapping=PHYSIONET_IMAGERY_MAPPING,
        preload=False,
        on_missing="ignore",
    )
