"""BCI Competition IV-2a/2b from the official competition files.

Source (BCI Competition IV, https://www.bbci.de/competition/iv/), nothing else:

    BCICIV_2a_gdf.zip   https://www.bbci.de/competition/download/competition_iv/BCICIV_2a_gdf.zip
    BCICIV_2b_gdf.zip   https://www.bbci.de/competition/download/competition_iv/BCICIV_2b_gdf.zip
    true labels 2a      https://www.bbci.de/competition/iv/results/ds2a/true_labels.zip
    true labels 2b      https://www.bbci.de/competition/iv/results/ds2b/true_labels.zip

The GDF files hold the recordings; the labels of the evaluation sessions are only in the
true-label files (published after the competition; they also cover the training sessions).
Each data set has one folder in data/ (see data/README.md):

    data/BCICIV_2a/gdf/           A01T.gdf ... A09E.gdf        (BCICIV_2a_gdf.zip, unzipped)
    data/BCICIV_2a/true_labels/   A01T.mat ... A09E.mat        (ds2a/true_labels.zip, unzipped)
    data/BCICIV_2a/mat/           A01T.mat ... A09E.mat        (built here; read by training)
    data/BCICIV_2b/gdf/           B0101T.gdf ... B0905E.gdf
    data/BCICIV_2b/true_labels/   B0101T.mat ... B0905E.mat
    data/BCICIV_2b/mat/           B01T.mat ... B09E.mat        (T = sessions 1-3, E = 4-5)

(the zips themselves, unextracted, are also recognised, there or in an input directory).
When the files in mat/ are missing, ``prepare`` builds them from gdf/ and true_labels/, or
from these files found in an input directory (e.g. a Kaggle Dataset attached with Add Input).
If they are not found, both zips are downloaded from the addresses above and extracted there.
mat/ is never stored: it is always built from gdf/ and true_labels/.

Training reads mat/ with MOABB, whose BNCI loader expects this MAT format (one cell array
of runs with X in uV, trial onsets, labels, sampling rate, class names); ``use_local_files``
points that loader at mat/ instead of its download location.

The conversion is the one used for the BNCI Horizon 2020 files. For BCI IV-2a, it reproduces
the 18 BNCI files: every field MOABB reads is identical, the signal bit for bit.
* X = digital value x calibration, in uV (the Graz calibration is a power-of-two fraction,
  so this is exact); saturated samples are kept as they are;
* a run starts at each "new run" event (32766); every run but the first of a file drops its
  first 100 samples; trial = 1-based onset of event 768 ("start of trial");
* 2a: one run per segment (3 EOG runs without trials, then 6 runs of 48 trials);
* 2b: MOABB treats every run of a file as one session, so each session file is one run (its
  segments joined); T = sessions 1-3, E = sessions 4-5 of a subject.
Labels come from the true-label files; in the training sessions they are checked against the
cue events of the GDF files.

    python -m ssm_dan.data.bciciv --dataset bcic2a            # find or download, then convert
    python -m ssm_dan.data.bciciv --dataset bcic2b --no-download --search /kaggle/input
"""
import argparse
import os
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np

from . import files

GDF_URL = {"bcic2a": "https://www.bbci.de/competition/download/competition_iv/BCICIV_2a_gdf.zip",
           "bcic2b": "https://www.bbci.de/competition/download/competition_iv/BCICIV_2b_gdf.zip"}
LABELS_URL = {"bcic2a": "https://www.bbci.de/competition/iv/results/ds2a/true_labels.zip",
              "bcic2b": "https://www.bbci.de/competition/iv/results/ds2b/true_labels.zip"}
CLASSES = {"bcic2a": ["left hand", "right hand", "feet", "tongue"], "bcic2b": ["left hand", "right hand"]}
BNCI_CODE = {"bcic2a": "001-2014", "bcic2b": "004-2014"}   # data-set codes in MOABB's download URLs
NEW_RUN, TRIAL_START, REJECTED = "32766", "768", "1023"
CUE_CODES = {"769": 1, "770": 2, "771": 3, "772": 4}
RUN_CUT = 100   # samples dropped at the start of every run but the first


# --------------------------------------------------------------------------- which files
def recordings(dataset):
    """Stems of the competition files (one GDF file and one true-label file each)."""
    if dataset == "bcic2a":
        return [f"A{s:02d}{r}" for s in range(1, 10) for r in "TE"]
    return [f"B{s:02d}0{k}{'T' if k <= 3 else 'E'}" for s in range(1, 10) for k in range(1, 6)]


def _sessions(dataset, subject):
    """(output MAT name, [recording stems]) for one subject."""
    if dataset == "bcic2a":
        return [(f"A{subject:02d}{s}", [f"A{subject:02d}{s}"]) for s in "TE"]
    return [(f"B{subject:02d}T", [f"B{subject:02d}0{k}T" for k in (1, 2, 3)]),
            (f"B{subject:02d}E", [f"B{subject:02d}0{k}E" for k in (4, 5)])]


def folder(dataset, part):
    """data/BCICIV_2a/<part> (part: gdf, true_labels or mat)."""
    return files.data_dir() / files.BCI_IV[dataset] / part


def _is_label_file(path):
    # The 2a label files share the names of the built MAT files (A01T.mat, ...);
    # they are a few kB and never inside a mat/ folder.
    return path.stat().st_size < 100_000 and path.parent.name != "mat"


def find(names, directories, label_files=False):
    """A zip archive or a directory that holds all ``names``, or None."""
    names = set(names)
    for directory in directories:
        directory = Path(directory)
        if not directory.is_dir():
            continue
        paths = list(files.walk(directory))
        for archive in sorted(p for p in paths if p.suffix == ".zip"):
            try:
                with zipfile.ZipFile(archive) as z:
                    if names <= {Path(n).name for n in z.namelist()}:
                        return archive
            except (zipfile.BadZipFile, OSError):
                continue
        by_folder = {}
        for path in paths:
            if path.name in names and path.is_file() and (not label_files or _is_label_file(path)):
                by_folder.setdefault(path.parent, set()).add(path.name)
        for folder, found in by_folder.items():
            if found == names:
                return folder
    return None


def _download_into(url, folder, retries=5):
    """Download a zip and extract its files (without sub-folders) into ``folder``."""
    import urllib.request

    folder.mkdir(parents=True, exist_ok=True)
    temporary = folder / f".download.{os.getpid()}.zip"
    for attempt in range(1, retries + 1):
        try:
            print(f"Downloading {url}", flush=True)
            urllib.request.urlretrieve(url, temporary)
            break
        except OSError as error:
            if attempt == retries:
                raise
            print(f"  {type(error).__name__}: {error}; retry {attempt}/{retries - 1} in {30 * attempt} s", flush=True)
            time.sleep(30 * attempt)
    with zipfile.ZipFile(temporary) as z:
        for member in z.namelist():
            if not member.endswith("/"):
                (folder / Path(member).name).write_bytes(z.read(member))
    temporary.unlink()
    return folder


# --------------------------------------------------------------------------- conversion
def _member(source, name, workdir):
    """Path of ``name`` inside a directory or a zip archive (extracted into workdir)."""
    source = Path(source)
    if source.is_dir():
        return next(p for p in source.iterdir() if p.name == name)
    with zipfile.ZipFile(source) as z:
        member = next(n for n in z.namelist() if Path(n).name == name)
        target = Path(workdir) / source.stem / name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(member))
        return target


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


def _segments(n_samples, events):
    starts = list(events.get(NEW_RUN, [0])) + [n_samples]
    if starts[0] != 0:
        starts = [0] + starts
    return [(start + (RUN_CUT if i else 0), stop) for i, (start, stop) in enumerate(zip(starts[:-1], starts[1:]))]


def _run(X, sfreq, events, parts, labels, classes):
    """One run from the given (start, stop) parts of a GDF signal; consumes labels in order."""
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
    """Runs of one MAT file from its GDF files and true-label files."""
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
        parts = _segments(len(X), events)
        if dataset == "bcic2a":
            runs += [_run(X, sfreq, events, [part], labels, CLASSES[dataset]) for part in parts]
        else:
            runs.append(_run(X, sfreq, events, parts, labels, CLASSES[dataset]))
    return runs


def write_mat(path, runs):
    import scipy.io

    cell = np.empty((1, len(runs)), dtype=object)
    for i, run in enumerate(runs):
        cell[0, i] = run
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.mat")
    scipy.io.savemat(temporary, {"data": cell}, do_compression=True)
    temporary.replace(path)


# --------------------------------------------------------------------------- entry point
def prepare(dataset, directories=(), download=True):
    """Build the missing files of data/BCICIV_2x/mat from the official competition files.

    The GDF and true-label files are looked for (unzipped or as zips) in data/ and
    ``directories``; if not found and ``download``, both zips are downloaded from bbci.de and
    extracted into data/BCICIV_2x/gdf and data/BCICIV_2x/true_labels.
    """
    if not files.missing(dataset):
        return
    search = [files.data_dir(), *directories]
    gdf = find([f"{stem}.gdf" for stem in recordings(dataset)], search)
    labels = find([f"{stem}.mat" for stem in recordings(dataset)], search, label_files=True)
    for kind, found, url, part in (("GDF", gdf, GDF_URL[dataset], "gdf"),
                                   ("true-label", labels, LABELS_URL[dataset], "true_labels")):
        if found is None and not download:
            raise FileNotFoundError(
                f"{dataset}: the competition {kind} files were not found in {search}. Unzip {url} "
                f"into {folder(dataset, part)} (see data/README.md).")
    gdf = gdf or _download_into(GDF_URL[dataset], folder(dataset, "gdf"))
    labels = labels or _download_into(LABELS_URL[dataset], folder(dataset, "true_labels"))
    print(f"{dataset}: building {folder(dataset, 'mat')} from {gdf} and {labels}", flush=True)
    with tempfile.TemporaryDirectory() as workdir:
        for subject in range(1, 10):
            for mat_name, stems in _sessions(dataset, subject):
                target = folder(dataset, "mat") / f"{mat_name}.mat"
                if target.is_file():
                    continue
                runs = convert_session(dataset, [_member(gdf, f"{s}.gdf", workdir) for s in stems],
                                       [_member(labels, f"{s}.mat", workdir) for s in stems])
                write_mat(target, runs)
                print(f"  {mat_name}.mat: trials per run {[len(r['trial']) for r in runs if len(r['trial'])]}",
                      flush=True)


def use_local_files():
    """Make MOABB's BNCI loader read data/BCICIV_2x/mat instead of downloading.

    MOABB 0.5 resolves every BNCI file through ``moabb.datasets.bnci.data_path(url, ...)``
    (e.g. url = .../001-2014/A01T.mat); this replaces that function so that the URL is
    mapped to the built file. The files and the way MOABB reads them are unchanged, and
    MOABB can no longer download from the BNCI server.
    """
    import moabb.datasets.bnci as bnci

    by_code = {code: dataset for dataset, code in BNCI_CODE.items()}

    def data_path(url, path=None, force_update=False, update_path=None, verbose=None):
        code, name = url.rstrip("/").split("/")[-2:]
        if code not in by_code:
            raise ValueError(f"{url}: only BCI IV-2a/2b are available locally")
        local = folder(by_code[code], "mat") / name
        if not local.is_file():
            raise FileNotFoundError(f"{local} is missing; run python -m ssm_dan.data.bciciv "
                                    f"--dataset {by_code[code]}")
        return [str(local)]

    bnci.data_path = data_path


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m ssm_dan.data.bciciv", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=sorted(GDF_URL))
    parser.add_argument("--search", default="", help=f"more directories ({os.pathsep}-separated) to look in")
    parser.add_argument("--no-download", action="store_true", help="fail instead of downloading")
    args = parser.parse_args(argv)
    prepare(args.dataset, [d for d in args.search.split(os.pathsep) if d], download=not args.no_download)
    files.report(args.dataset)
    return 0 if not files.missing(args.dataset) else 1


if __name__ == "__main__":
    sys.exit(main())
