from typing import Dict

from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import (
    Preprocessor,
    create_windows_from_events,
    preprocess,
)


def scale(data, factor):
    return data * factor


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

    MOABB's ``PhysionetMI`` defaults to imagined (not executed) runs.  The
    explicit mapping removes the inter-trial ``rest`` event and gives stable
    class indices across runs that contain different pairs of motor tasks.
    """
    dataset = MOABBDataset(
        dataset_name="PhysionetMI",
        subject_ids=subject_ids,
    )

    preprocessors = [
        Preprocessor(
            "pick_types", eeg=True, meg=False, stim=False, verbose=verbose
        ),
        Preprocessor(scale, factor=1e6, apply_on_array=True),
        Preprocessor(
            "resample", sfreq=preprocessing_dict["sfreq"], verbose=verbose
        ),
    ]

    low_cut = preprocessing_dict.get("low_cut")
    high_cut = preprocessing_dict.get("high_cut")
    if low_cut is not None or high_cut is not None:
        preprocessors.append(
            Preprocessor(
                "filter", l_freq=low_cut, h_freq=high_cut, verbose=verbose
            )
        )

    preprocess(dataset, preprocessors)

    sfreq = dataset.datasets[0].raw.info["sfreq"]
    start_offset = int(round(preprocessing_dict.get("start", 0.0) * sfreq))
    trial_duration = float(preprocessing_dict.get("trial_duration", 3.0))
    window_size = int(round(trial_duration * sfreq))
    if start_offset != 0 or trial_duration != 3.0:
        raise ValueError(
            "The PhysioNet MOABB benchmark requires trials from 0.0 to 3.0 s."
        )

    # EDF task annotations last about 4.1 s, whereas MOABB defines the
    # benchmark interval as [0, 3] s.  An explicit window size plus
    # drop_last_window=True produces exactly one onset-aligned 3 s epoch and
    # prevents Braindecode from adding a second, end-aligned window.
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
