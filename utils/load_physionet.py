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
    start_offset = int(preprocessing_dict.get("start", 0.0) * sfreq)
    stop_offset = int(preprocessing_dict.get("stop", 0.0) * sfreq)
    return create_windows_from_events(
        dataset,
        trial_start_offset_samples=start_offset,
        trial_stop_offset_samples=stop_offset,
        mapping=PHYSIONET_IMAGERY_MAPPING,
        preload=False,
        on_missing="ignore",
    )
