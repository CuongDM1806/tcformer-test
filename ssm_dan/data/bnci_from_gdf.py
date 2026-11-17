"""Rebuild the BNCI Horizon 2020 MAT files of BCI IV-2a/2b from the competition GDF files.

The BNCI server (data sets 001-2014 and 004-2014) is sometimes unreachable. The same
recordings are distributed by the BCI Competition IV as GDF files
(https://www.bbci.de/competition/iv/download/: BCICIV_2a_gdf.zip, BCICIV_2b_gdf.zip),
with the labels of the evaluation sessions published separately
(https://www.bbci.de/competition/iv/results/: ds2a/true_labels.zip, ds2b/true_labels.zip).
This module writes, from those two archives, MAT files in the BNCI layout that MOABB reads:

    data/MNE-bnci-data/database/data-sets/001-2014/A{01..09}{T,E}.mat     (2a)
    data/MNE-bnci-data/database/data-sets/004-2014/B{01..09}{T,E}.mat     (2b)

Each file holds ``data``, a cell array of runs with fields X (samples x channels, uV),
trial (1-based onsets of the "start of trial" event 768), y (labels, 1..n_classes),
fs, classes, artifacts (event 1023), gender and age (unknown here: '' and NaN; MOABB
does not read them).

How the BNCI files were derived from the GDF files (checked on all 18 BCI IV-2a files,
for which the original MAT files are available: every field MOABB reads is identical,
X bit for bit):
* X = digital value x calibration, in uV (the Graz calibration is a power-of-two
  fraction, so this is exact); saturated samples are kept as they are;
* a run starts at each "new run" event (32766); every run but the first of a file
  drops its first 100 samples;
* 2a: one run per segment (3 EOG runs without trials, then 6 runs of 48 trials);
* 2b: MOABB treats every run of a file as one session, so each session file is one
  run (its segments joined, with the same 100-sample cut); T = sessions 1-3 and
  E = sessions 4-5 of a subject.

    python -m ssm_dan.data.bnci_from_gdf --dataset bcic2b --gdf BCICIV_2b_gdf.zip \\
        --labels true_labels_2b.zip
    python -m ssm_dan.data.bnci_from_gdf --dataset bcic2a --gdf BCICIV_2a_gdf.zip \\
        --labels true_labels_2a.zip --check-against data/   # compare with existing MAT files

``--gdf`` and ``--labels`` accept a zip archive or a directory.
"""
import argparse
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from . import files

CLASSES = {"bcic2a": ["left hand", "right hand", "feet", "tongue"], "bcic2b": ["left hand", "right hand"]}
FOLDER = {"bcic2a": "001-2014", "bcic2b": "004-2014"}
NEW_RUN, TRIAL_START, REJECTED = "32766", "768", "1023"
CUE_CODES = {"769": 1, "770": 2, "771": 3, "772": 4}
RUN_CUT = 100   # samples dropped at the start of every run but the first


def _sessions(dataset, subject):
    """(output MAT name, [GDF names]) for one subject."""
    if dataset == "bcic2a":
        return [(f"A{subject:02d}{s}", [f"A{subject:02d}{s}"]) for s in "TE"]
    return [(f"B{subject:02d}T", [f"B{subject:02d}0{k}T" for k in (1, 2, 3)]),
            (f"B{subject:02d}E", [f"B{subject:02d}0{k}E" for k in (4, 5)])]


def _directory(source, workdir, name):
    source = Path(source)
    if source.is_dir():
        return source
    target = Path(workdir) / name
    with zipfile.ZipFile(source) as archive:
        archive.extractall(target)
    return target


def _find(directory, filename):
    hits = sorted(Path(directory).rglob(filename))
    if not hits:
        raise FileNotFoundError(f"{filename} not found under {directory}")
    return hits[0]


def read_gdf(path):
    """Signal in uV exactly as BioSig computes it, sampling rate, and event positions by code."""
    import mne

    raw = mne.io.read_raw_gdf(path, preload=True, verbose="ERROR")
    extras = raw._raw_extras[0]
    n = len(raw.ch_names)
    cal, offset, units = (np.asarray(extras[key], float)[:n] for key in ("cal", "offsets", "units"))
    digital = np.round((raw.get_data().T / units - offset) / cal)
    X = digital * cal + offset
    sfreq = raw.info["sfreq"]
    events = {}
    for annotation in raw.annotations:
        events.setdefault(annotation["description"], []).append(int(round(annotation["onset"] * sfreq)))
    return X, int(sfreq), {code: np.array(positions) for code, positions in events.items()}


def segments(n_samples, events):
    starts = list(events.get(NEW_RUN, [0])) + [n_samples]
    if starts[0] != 0:
        starts = [0] + starts
    return [(start + (RUN_CUT if i else 0), stop) for i, (start, stop) in enumerate(zip(starts[:-1], starts[1:]))]


def _run(X, sfreq, events, parts, labels, classes):
    """One BNCI run from the given (start, stop) parts of a GDF signal; consumes labels in order."""
    trial_starts = events.get(TRIAL_START, np.array([], int))
    rejected = set(events.get(REJECTED, []))
    chunks, trial, artifacts, offset = [], [], [], 0
    for start, stop in parts:
        chunks.append(X[start:stop])
        for position in trial_starts[(trial_starts >= start) & (trial_starts < stop)]:
            trial.append(position - start + offset + 1)
            artifacts.append(int(position in rejected))
        offset += stop - start
    y = labels[:len(trial)]
    del labels[:len(trial)]
    return {"X": np.concatenate(chunks), "trial": np.array(trial, np.int32), "y": np.array(y, np.uint8),
            "fs": sfreq, "classes": np.array(classes, dtype=object), "artifacts": np.array(artifacts, np.uint8),
            "gender": "", "age": np.nan}


def convert_session(dataset, gdf_paths, label_paths):
    """BNCI runs of one output MAT file."""
    import scipy.io

    runs = []
    for gdf_path, label_path in zip(gdf_paths, label_paths):
        X, sfreq, events = read_gdf(gdf_path)
        labels = [int(v) for v in np.ravel(scipy.io.loadmat(label_path)["classlabel"])]
        n_trials = len(events.get(TRIAL_START, []))
        if len(labels) != n_trials:
            raise ValueError(f"{gdf_path.name}: {n_trials} trials but {len(labels)} labels")
        cues = sorted((p, CUE_CODES[c]) for c in CUE_CODES for p in events.get(c, []))
        if cues and [label for _, label in cues] != labels:   # training sessions: labels also in the GDF
            raise ValueError(f"{gdf_path.name}: cue events disagree with the true labels")
        parts = segments(len(X), events)
        if dataset == "bcic2a":
            runs += [_run(X, sfreq, events, [part], labels, CLASSES[dataset]) for part in parts]
        else:
            runs.append(_run(X, sfreq, events, parts, labels, CLASSES[dataset]))
        assert not labels
    return runs


def write_mat(path, runs):
    import scipy.io

    cell = np.empty((1, len(runs)), dtype=object)
    for i, run in enumerate(runs):
        cell[0, i] = run
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.mat")
    scipy.io.savemat(temporary, {"data": cell}, do_compression=True)
    temporary.replace(path)


def compare(path_a, path_b):
    """Differences between two BNCI MAT files in the fields MOABB reads (empty list if identical)."""
    import scipy.io

    load = lambda p: np.atleast_1d(scipy.io.loadmat(p, struct_as_record=False, squeeze_me=True)["data"])
    a, b = load(path_a), load(path_b)
    if len(a) != len(b):
        return [f"{len(a)} vs {len(b)} runs"]
    problems = []
    for i, (ra, rb) in enumerate(zip(a, b)):
        if not np.array_equal(ra.X, rb.X):
            problems.append(f"run {i}: X differs")
        for field in ("trial", "y", "artifacts"):
            if not np.array_equal(np.atleast_1d(getattr(ra, field)), np.atleast_1d(getattr(rb, field))):
                problems.append(f"run {i}: {field} differs")
        if int(ra.fs) != int(rb.fs) or list(np.atleast_1d(ra.classes)) != list(np.atleast_1d(rb.classes)):
            problems.append(f"run {i}: fs or classes differ")
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m ssm_dan.data.bnci_from_gdf", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=sorted(CLASSES))
    parser.add_argument("--gdf", required=True, help="BCICIV_2a_gdf.zip / BCICIV_2b_gdf.zip or a directory")
    parser.add_argument("--labels", required=True, help="true_labels.zip of the data set or a directory")
    parser.add_argument("--out", default=None, help="data directory (default: data/)")
    parser.add_argument("--check-against", default=None,
                        help="data directory with original BNCI MAT files: compare instead of writing")
    args = parser.parse_args(argv)

    out_root = Path(args.out).resolve() if args.out else files.data_dir()
    folder = files.BNCI / FOLDER[args.dataset]
    failures = 0
    with tempfile.TemporaryDirectory() as workdir:
        gdf_dir = _directory(args.gdf, workdir, "gdf")
        label_dir = _directory(args.labels, workdir, "labels")
        for subject in range(1, 10):
            for mat_name, gdf_names in _sessions(args.dataset, subject):
                runs = convert_session(args.dataset, [_find(gdf_dir, f"{g}.gdf") for g in gdf_names],
                                       [_find(label_dir, f"{g}.mat") for g in gdf_names])
                trials = [len(r["trial"]) for r in runs if len(r["trial"])]
                if args.check_against:
                    candidate = Path(workdir) / f"{mat_name}.mat"
                    write_mat(candidate, runs)
                    problems = compare(candidate, Path(args.check_against) / folder / f"{mat_name}.mat")
                    failures += bool(problems)
                    print(f"{mat_name}: {'identical to the original' if not problems else problems}", flush=True)
                else:
                    write_mat(out_root / folder / f"{mat_name}.mat", runs)
                    print(f"{mat_name}.mat: {len(runs)} runs, trials per run with trials {trials}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
