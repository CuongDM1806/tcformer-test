import time
from typing import Dict

from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import (
    Preprocessor,
    create_windows_from_events,
    preprocess,
)
from requests.exceptions import RequestException


OPENBMI_MI_MAPPING = {
    "left_hand": 0,
    "right_hand": 1,
}


def _scale(data, factor):
    return data * factor


def load_openbmi(
    subject_ids: list,
    preprocessing_dict: Dict,
    verbose: str = "WARNING",
):
    """Load labeled offline OpenBMI/Lee2019 motor-imagery runs.

    MOABB's ``Lee2019_MI`` exposes the two labeled offline runs (one in each
    recording session) by default and excludes the online feedback runs whose
    MI labels are unavailable through MOABB. Epoch zero is the visual MI cue.
    """
    download_retries = int(preprocessing_dict.get("download_retries", 8))
    retry_delay = float(
        preprocessing_dict.get("download_retry_delay_seconds", 5.0)
    )
    if download_retries < 0 or retry_delay < 0:
        raise ValueError("OpenBMI download retry settings must be non-negative.")

    for attempt in range(download_retries + 1):
        try:
            dataset = MOABBDataset(
                dataset_name="Lee2019_MI",
                subject_ids=subject_ids,
            )
            break
        except RequestException as exc:
            if attempt == download_retries:
                raise
            wait_seconds = min(retry_delay * (2**attempt), 60.0)
            print(
                f"OpenBMI download failed ({type(exc).__name__}: {exc}). "
                f"Retry {attempt + 1}/{download_retries} in {wait_seconds:g} s; "
                "cached MAT files will be reused.",
                flush=True,
            )
            time.sleep(wait_seconds)

    preprocessors = [
        Preprocessor(
            "pick_types", eeg=True, meg=False, stim=False, verbose=verbose
        ),
        Preprocessor(_scale, factor=1e6, apply_on_array=True),
    ]
    low_cut = preprocessing_dict.get("low_cut")
    high_cut = preprocessing_dict.get("high_cut")
    if low_cut is not None or high_cut is not None:
        preprocessors.append(
            Preprocessor(
                "filter", l_freq=low_cut, h_freq=high_cut, verbose=verbose
            )
        )
    if preprocessing_dict.get("common_average_reference", True):
        preprocessors.append(
            Preprocessor(
                "set_eeg_reference",
                ref_channels="average",
                ch_type="eeg",
                projection=False,
                verbose=verbose,
            )
        )
    preprocessors.append(
        Preprocessor(
            "resample", sfreq=preprocessing_dict["sfreq"], verbose=verbose
        )
    )
    preprocess(dataset, preprocessors)

    sfreq = dataset.datasets[0].raw.info["sfreq"]
    start = float(preprocessing_dict.get("start", 0.0))
    duration = float(preprocessing_dict.get("trial_duration", 4.0))
    start_offset = int(round(start * sfreq))
    window_size = int(round(duration * sfreq))
    if start != 0.0 or duration != 4.0:
        raise ValueError(
            "The OpenBMI LOSO protocol must use the full 0.0-4.0 s MI cue interval."
        )

    return create_windows_from_events(
        dataset,
        trial_start_offset_samples=start_offset,
        trial_stop_offset_samples=0,
        window_size_samples=window_size,
        window_stride_samples=window_size,
        drop_last_window=True,
        mapping=OPENBMI_MI_MAPPING,
        preload=False,
        on_missing="ignore",
    )
