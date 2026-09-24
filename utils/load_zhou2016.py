from typing import Dict

from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import (
    Preprocessor,
    create_windows_from_events,
    preprocess,
)


def scale(data, factor):
    return data * factor


def load_zhou2016(
    subject_ids: list,
    preprocessing_dict: Dict,
    verbose: str = "WARNING",
):
    """Load Zhou2016 and create one exact five-second window per MI cue."""
    dataset = MOABBDataset(dataset_name="Zhou2016", subject_ids=subject_ids)

    preprocessors = [
        Preprocessor(
            "pick_types", eeg=True, meg=False, stim=False, verbose=verbose
        ),
        # MNE stores EEG in volts; the other raw-EEG loaders in this project
        # expose microvolts to the network.
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

    sfreq = float(dataset.datasets[0].raw.info["sfreq"])
    start = float(preprocessing_dict.get("start", 0.0))
    duration = float(preprocessing_dict.get("trial_duration", 5.0))
    if start < 0:
        raise ValueError("Zhou2016 trial start must be non-negative.")
    if duration <= 0:
        raise ValueError("Zhou2016 trial_duration must be positive.")

    start_offset_samples = int(round(start * sfreq))
    window_size_samples = int(round(duration * sfreq))
    # Zhou2016 CNT cue annotations are point events: their stored duration is
    # zero. Therefore Braindecode cannot infer a trial extent on its own. Make
    # the stop offset explicit so each cue spans exactly `duration` seconds.
    # Offsets are both relative to cue onset, hence stop = start + duration.
    stop_offset_samples = start_offset_samples + window_size_samples
    return create_windows_from_events(
        dataset,
        trial_start_offset_samples=start_offset_samples,
        trial_stop_offset_samples=stop_offset_samples,
        window_size_samples=window_size_samples,
        window_stride_samples=window_size_samples,
        drop_last_window=False,
        # Braindecode otherwise derives IDs from alphabetically sorted event
        # names (feet, left_hand, right_hand). Keep the project-wide class
        # order explicit and stable for metrics/confusion matrices.
        mapping={"left_hand": 0, "right_hand": 1, "feet": 2},
        preload=False,
    )
