"""MOABB/Braindecode loaders: BCI IV-2a/2b and Zhou2016 trial windows in microvolts."""
from typing import Dict

from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import Preprocessor, create_windows_from_events, preprocess

from . import files

files.configure()   # MOABB reads (and, if missing, downloads) the recordings in data/


def scale(data, factor):
    return data * factor


def _base_preprocessors(preprocessing: Dict, verbose: str):
    return [
        Preprocessor("pick_types", eeg=True, meg=False, stim=False, verbose=verbose),
        # MNE stores EEG in volts; the networks receive microvolts.
        Preprocessor(scale, factor=1e6, apply_on_array=True),
        Preprocessor("resample", sfreq=preprocessing["sfreq"], verbose=verbose),
    ]


def _band_pass(preprocessing: Dict, verbose: str):
    low_cut, high_cut = preprocessing.get("low_cut"), preprocessing.get("high_cut")
    if low_cut is None and high_cut is None:
        return []
    return [Preprocessor("filter", l_freq=low_cut, h_freq=high_cut, verbose=verbose)]


def load_bcic4(subject_ids: list, dataset: str, preprocessing: Dict, verbose: str = "WARNING"):
    """BCI IV-2a (``dataset="2a"``) or 2b (``"2b"``) windows over the MOABB event interval."""
    from .bciciv import prepare

    # Built from the official competition files; MOABB then reads them without any download.
    prepare("bcic2a" if dataset == "2a" else "bcic2b")
    names = ("BNCI2014_001", "BNCI2014001") if dataset == "2a" else ("BNCI2014_004", "BNCI2014004")
    last_error = None
    for name in names:               # MOABB renamed the datasets across versions
        try:
            windows = MOABBDataset(name, subject_ids=subject_ids)
            break
        except ValueError as exc:
            last_error = exc
    else:
        raise last_error

    preprocess(windows, _base_preprocessors(preprocessing, verbose)
               + _band_pass(preprocessing, verbose))
    sfreq = windows.datasets[0].raw.info["sfreq"]
    return create_windows_from_events(
        windows,
        trial_start_offset_samples=int(preprocessing["start"] * sfreq),
        trial_stop_offset_samples=int(preprocessing["stop"] * sfreq),
        preload=False,
    )


def load_zhou2016(subject_ids: list, preprocessing: Dict, verbose: str = "WARNING"):
    """Zhou2016 windows of ``trial_duration`` seconds starting ``start`` s after each cue."""
    windows = MOABBDataset(dataset_name="Zhou2016", subject_ids=subject_ids)
    preprocess(windows, _base_preprocessors(preprocessing, verbose)
               + _band_pass(preprocessing, verbose))

    sfreq = float(windows.datasets[0].raw.info["sfreq"])
    start = float(preprocessing.get("start", 0.0))
    duration = float(preprocessing.get("trial_duration", 5.0))
    if start < 0 or duration <= 0:
        raise ValueError("Zhou2016 needs start >= 0 and trial_duration > 0.")
    start_samples = int(round(start * sfreq))
    window_samples = int(round(duration * sfreq))
    # Zhou2016 cue annotations have zero duration, so the stop offset is given
    # explicitly: both offsets are relative to cue onset.
    return create_windows_from_events(
        windows,
        trial_start_offset_samples=start_samples,
        trial_stop_offset_samples=start_samples + window_samples,
        window_size_samples=window_samples,
        window_stride_samples=window_samples,
        drop_last_window=False,
        # Fixed class order (Braindecode would otherwise sort the event names).
        mapping={"left_hand": 0, "right_hand": 1, "feet": 2},
        preload=False,
    )
