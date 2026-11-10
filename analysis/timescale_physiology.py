"""Relate the learned target time scales to each subject's sensorimotor rhythms (BCI IV-2a).

For every target subject, the SSM time-scale ratio tau_target / tau_source (written by
train_pipeline.py into results/*/subjects/subject_<id>.json) is compared with rhythm
features measured directly on that subject's session-T EEG (the unlabeled Stage-1
session; only cue timing is used, never the class labels):
  mu_peak_hz    peak of the 7-14 Hz power spectrum during imagery (C3, Cz, C4)
  beta_peak_hz  peak of the 14-30 Hz power spectrum during imagery
  mu_erd_pct    mean mu-power change 0.5-4 s after the cue vs. the -1.5 to -0.25 s baseline
  mu_erd_min_s  time of the deepest mu desynchronization within 0-4 s after the cue
Spearman correlations across subjects are printed; with nine subjects they are exploratory.

Usage: python analysis/timescale_physiology.py --results <dir or zip> [...] --out table.csv
"""
import argparse
import csv
import json
import zipfile
from pathlib import Path

import mne
import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt, welch
from scipy.stats import spearmanr

CHANNELS = ["C3", "Cz", "C4"]
BASELINE = (-1.5, -0.25)
IMAGERY = (0.5, 4.0)
FEATURES = ["mu_peak_hz", "beta_peak_hz", "mu_erd_pct", "mu_erd_min_s"]


def load_subject(subject):
    from braindecode.datasets import MOABBDataset
    for name in ("BNCI2014_001", "BNCI2014001"):
        try:
            return MOABBDataset(name, subject_ids=[subject])
        except ValueError:
            continue
    raise RuntimeError("BCI IV-2a is not available through MOABB.")


def training_session(names):
    """The calibration session T: 'session_T' (MOABB 0.5) or '0train' (later)."""
    for name in sorted(names):
        if name.endswith("T") or "train" in name:
            return name
    return sorted(names)[0]


def subject_rhythms(subject):
    dataset = load_subject(subject)
    session = training_session({d.description["session"] for d in dataset.datasets})
    trials, sfreq = [], None
    for item in dataset.datasets:
        if item.description["session"] != session:
            continue
        raw = item.raw.copy().pick(CHANNELS)
        sfreq = raw.info["sfreq"]
        data = raw.get_data()
        events, _ = mne.events_from_annotations(raw, verbose=False)
        start = int(round(BASELINE[0] * sfreq))
        stop = int(round(IMAGERY[1] * sfreq))
        for onset in events[:, 0]:
            if onset + start >= 0 and onset + stop <= data.shape[1]:
                trials.append((data, onset))
    if not trials:
        raise RuntimeError(f"No cue events found for subject {subject}.")

    times = np.arange(int(round(BASELINE[0] * sfreq)), int(round(IMAGERY[1] * sfreq))) / sfreq
    mu_sos = butter(4, [8.0, 13.0], btype="bandpass", fs=sfreq, output="sos")
    envelopes, spectra = [], []
    filtered_runs = {}
    for data, onset in trials:
        key = id(data)
        if key not in filtered_runs:
            filtered_runs[key] = np.abs(hilbert(sosfiltfilt(mu_sos, data, axis=-1), axis=-1)) ** 2
        index = onset + (times * sfreq).round().astype(int)
        envelopes.append(filtered_runs[key][:, index].mean(axis=0))
        imagery = data[:, onset + int(IMAGERY[0] * sfreq): onset + int(IMAGERY[1] * sfreq)]
        frequencies, power = welch(imagery, fs=sfreq, nperseg=int(2 * sfreq), axis=-1)
        spectra.append(power.mean(axis=0))

    spectrum = np.mean(spectra, axis=0)
    mu_band = (frequencies >= 7) & (frequencies <= 14)
    beta_band = (frequencies > 14) & (frequencies <= 30)
    envelope = np.mean(envelopes, axis=0)
    baseline = envelope[(times >= BASELINE[0]) & (times < BASELINE[1])].mean()
    change = (envelope / baseline - 1.0) * 100.0
    smooth = np.convolve(change, np.ones(int(0.25 * sfreq)) / int(0.25 * sfreq), mode="same")
    window = (times >= 0.0) & (times <= IMAGERY[1])
    return {
        "mu_peak_hz": float(frequencies[mu_band][np.argmax(spectrum[mu_band])]),
        "beta_peak_hz": float(frequencies[beta_band][np.argmax(spectrum[beta_band])]),
        "mu_erd_pct": float(change[(times >= IMAGERY[0]) & (times <= IMAGERY[1])].mean()),
        "mu_erd_min_s": float(times[window][np.argmin(smooth[window])]),
        "trials": len(trials),
    }


def subject_records(source, run=None):
    """Yield (run, subject record) from result folders or zip archives.

    A folder is one run, including the per-fold zips inside it (Kaggle output).
    """
    source = Path(source)
    if source.is_file() and source.suffix == ".zip":
        with zipfile.ZipFile(source) as archive:
            for name in archive.namelist():
                if "/subjects/subject_" in name and name.endswith(".json"):
                    yield run or source.stem, json.loads(archive.read(name))
        return
    for path in sorted(source.rglob("*")):
        if path.suffix == ".zip":
            yield from subject_records(path, run or source.name)
        elif path.suffix == ".json" and path.parent.name == "subjects":
            yield run or source.name, json.loads(path.read_text())


def time_scales(record):
    """tau ratio per scored variant: the primary TTA and every extra variant."""
    scales = {}
    if record.get("timescale"):
        scales[f"tta_{record.get('im_tta_parameters', 'primary')}"] = record["timescale"]["tau_ratio"]
    for mode, result in record.get("tta_variants", {}).items():
        if "timescale" in result:
            scales[f"tta_{mode}"] = result["timescale"]["tau_ratio"]
    return scales


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for source in args.results:
        for run, record in subject_records(source):
            for variant, tau in time_scales(record).items():
                rows.append({"run": run, "subject_id": int(record["subject_id"]),
                             "variant": variant, "tau_ratio": tau})
    if not rows:
        raise SystemExit("No subject records with time-scale summaries were found.")

    rhythms = {s: subject_rhythms(s) for s in sorted({r["subject_id"] for r in rows})}
    for row in rows:
        row.update({k: rhythms[row["subject_id"]][k] for k in FEATURES})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["run"], r["variant"], r["subject_id"])))

    print("Subject rhythms (session T):")
    for subject, values in rhythms.items():
        print(f"  S{subject}: " + ", ".join(f"{k}={values[k]:.2f}" for k in FEATURES))
    print("Spearman rho (p) between tau_target/tau_source and rhythm features:")
    for key in sorted({(r["run"], r["variant"]) for r in rows}):
        group = [r for r in rows if (r["run"], r["variant"]) == key]
        taus = [r["tau_ratio"] for r in group]
        if len(group) < 4 or np.ptp(taus) == 0:
            continue
        cells = []
        for feature in FEATURES:
            rho, p = spearmanr(taus, [r[feature] for r in group])
            cells.append(f"{feature} {rho:+.2f} ({p:.2f})")
        print(f"  {key[0]} | {key[1]} | n={len(group)} | " + " | ".join(cells))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
