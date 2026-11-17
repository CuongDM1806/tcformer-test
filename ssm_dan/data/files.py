"""Where the EEG data live, which files each dataset needs, and how missing files are obtained.

All datasets are read from one fixed directory, ``data/`` at the repository root
(``$SSM_DAN_DATA`` moves it). A file that is already there is used as it is; only
missing files are obtained. The expected layout is MOABB's own, so MOABB reads the
files without contacting any server (see data/README.md):

    data/BCICIV_2a/{gdf,true_labels,mat}/                                 BCI IV-2a
    data/BCICIV_2b/{gdf,true_labels,mat}/                                 BCI IV-2b
    data/MNE-zhou-2016/subject_1/1A.cnt ... subject_4/3B.cnt               Zhou2016
    data/openbmi/openbmi_S01.npz ... openbmi_S54.npz                       OpenBMI (CAR)
    data/openbmi_nocar/openbmi_S01.npz ...                                 OpenBMI without CAR

BCI IV-2a/2b are built from the official competition files (GDF recordings + true labels)
by ssm_dan.data.bciciv; the BNCI Horizon 2020 server is never used. For the other datasets,
missing files are taken, in this order, from
1. other directories given to ``link_from`` (e.g. attached Kaggle inputs): files with the
   expected names are linked into data/;
2. the repository's data branch ``<dataset>-data`` (same layout as data/; OpenBMI: CAR only);
3. the original servers (MOABB download of Zhou2016, retried; OpenBMI is preprocessed from raw).

Command line (``--check`` lists what is missing; ``--download`` fetches it):

    python -m ssm_dan.data.files --dataset bcic2a --check
    python -m ssm_dan.data.files --dataset bcic2a --download
    python -m ssm_dan.data.files --dataset all --check

Standard library only at import time (the launcher uses it in a bare notebook kernel).
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DATASETS = ("bcic2a", "bcic2b", "zhou2016", "openbmi")
BCI_IV = {"bcic2a": "BCICIV_2a", "bcic2b": "BCICIV_2b"}   # gdf/, true_labels/ and the built mat/
ZHOU = Path("MNE-zhou-2016")
MOABB_NAMES = {"zhou2016": ("Zhou2016",)}
SUBJECTS = {"bcic2a": range(1, 10), "bcic2b": range(1, 10), "zhou2016": range(1, 5), "openbmi": range(1, 55)}


def data_dir() -> Path:
    return Path(os.environ.get("SSM_DAN_DATA", ROOT / "data")).resolve()


def openbmi_dir(no_car=False) -> Path:
    return data_dir() / ("openbmi_nocar" if no_car else "openbmi")


def configure(env=None):
    """Point MOABB/MNE at data/ (in ``env``, default os.environ); returns the mapping."""
    env = os.environ if env is None else env
    root = str(data_dir())
    env["SSM_DAN_DATA"] = root
    env["MNE_DATA"] = root
    env["MNE_DATASETS_ZHOU_PATH"] = root
    # Raw OpenBMI MAT files are only needed while building data/openbmi*; they are deleted afterwards.
    env.setdefault("MNE_DATASETS_LEE2019_MI_PATH", str(data_dir() / "openbmi_raw"))
    return env


def expected(dataset, no_car=False):
    """Paths (relative to data/) of every file the dataset needs."""
    if dataset == "bcic2a":
        return [Path(BCI_IV[dataset], "mat", f"A{s:02d}{r}.mat") for s in SUBJECTS[dataset] for r in "TE"]
    if dataset == "bcic2b":
        return [Path(BCI_IV[dataset], "mat", f"B{s:02d}{r}.mat") for s in SUBJECTS[dataset] for r in "TE"]
    if dataset == "zhou2016":
        return [ZHOU / f"subject_{s}" / f"{session}{run}.cnt"
                for s in SUBJECTS[dataset] for session in "123" for run in "AB"]
    if dataset == "openbmi":
        folder = Path(openbmi_dir(no_car).name)
        return [folder / f"openbmi_S{s:02d}.npz" for s in SUBJECTS[dataset]]
    raise ValueError(f"Unknown dataset {dataset!r}; choose from {DATASETS}.")


def _openbmi_car(path):
    """common_average_reference recorded in an OpenBMI file (None if unreadable)."""
    try:
        import numpy as np
        with np.load(path, allow_pickle=False) as data:
            return json.loads(str(data["signature"])).get("common_average_reference")
    except Exception:
        return None


def _usable(dataset, path, no_car):
    if not path.is_file():
        return False
    if dataset == "openbmi":
        return _openbmi_car(path) is (not no_car)
    return path.stat().st_size > 0


def missing(dataset, no_car=False):
    root = data_dir()
    return [rel for rel in expected(dataset, no_car) if not _usable(dataset, root / rel, no_car)]


def _place(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    try:
        destination.symlink_to(source.resolve())
    except OSError:
        shutil.copy2(source, destination)


def link_from(dataset, directories, no_car=False):
    """Link files with the expected names found under ``directories`` into data/; returns the count."""
    wanted = missing(dataset, no_car)
    if not wanted:
        return 0
    root, linked = data_dir(), 0
    by_name = {}
    for rel in wanted:
        by_name.setdefault(rel.name, []).append(rel)
    for directory in directories:
        directory = Path(directory)
        if not directory.is_dir() or directory.resolve() == root or root in directory.resolve().parents:
            continue
        for path in directory.rglob("*"):
            if path.name not in by_name or root in path.parents:
                continue
            for rel in by_name[path.name]:
                destination = root / rel
                if _usable(dataset, destination, no_car) or not path.is_file():
                    continue
                # Same parent folder name (001-2014, subject_3, ...) unless it is an OpenBMI file.
                if dataset != "openbmi" and path.parent.name != rel.parent.name:
                    continue
                if dataset == "openbmi" and _openbmi_car(path) is not (not no_car):
                    continue
                _place(path, destination)
                linked += 1
    return linked


REPOSITORY_URLS = ["https://github.com/CuongDM1806/ssm-dan.git",
                   "https://github.com/CuongDM1806/tcformer-test.git"]   # former name


def repository_urls():
    """This repository's clone URL (origin first), to fetch its data branches."""
    import subprocess

    origin = subprocess.run(["git", "-C", str(ROOT), "remote", "get-url", "origin"],
                            capture_output=True, text=True).stdout.strip()
    return list(dict.fromkeys(([origin] if origin.startswith("https://") else []) + REPOSITORY_URLS))


def data_branch(dataset):
    """Data-only branch of this repository that holds the dataset's files, in the data/ layout."""
    return f"{dataset}-data"


def fetch_branch(dataset, urls, scratch, no_car=False):
    """Clone the dataset's data branch and move the files that are still missing into data/.

    Returns the number of files moved (0 if the branch does not exist or cannot be reached).
    """
    import subprocess

    if dataset == "openbmi" and no_car:
        return 0                      # the branch holds the paper's preprocessing (CAR) only
    branch = data_branch(dataset)
    clone = Path(scratch) / branch
    if not (clone / ".git").exists():
        for url in urls:
            print(f"{dataset}: fetching branch {branch} of {url}", flush=True)
            done = subprocess.run(["git", "clone", "--depth", "1", "--single-branch", "--branch", branch,
                                   url, str(clone)], env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
            if done.returncode == 0:
                break
    if not clone.is_dir():
        return 0
    moved = 0
    for rel in missing(dataset, no_car):
        source, destination = clone / rel, data_dir() / rel
        if _usable(dataset, source, no_car):
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() or destination.is_symlink():
                destination.unlink()
            shutil.move(str(source), destination)
            moved += 1
    shutil.rmtree(clone, ignore_errors=True)
    return moved


def download(dataset, retries=10):
    """Download the missing BCI IV / Zhou2016 files with MOABB (needs the training environment)."""
    configure()
    import moabb.datasets as moabb_datasets

    if dataset == "zhou2016" and missing(dataset):
        # MOABB fetches Zhou2016 as one zip and lays out all four subjects at once.
        shutil.rmtree(data_dir() / ZHOU, ignore_errors=True)
    name = next(n for n in MOABB_NAMES[dataset] if hasattr(moabb_datasets, n))
    moabb_dataset = getattr(moabb_datasets, name)()
    for subject in moabb_dataset.subject_list:
        for attempt in range(1, retries + 1):
            try:
                moabb_dataset.data_path(subject)
                break
            except Exception as error:       # servers drop connections now and then
                if attempt == retries:
                    raise
                print(f"{dataset} subject {subject}: {type(error).__name__}; retry {attempt}/{retries - 1} "
                      f"in {30 * attempt} s", flush=True)
                time.sleep(30 * attempt)
        print(f"{dataset} subject {subject} ready", flush=True)


def report(dataset, no_car=False):
    gone = missing(dataset, no_car)
    label = dataset + (" (no CAR)" if dataset == "openbmi" and no_car else "")
    total = len(expected(dataset, no_car))
    if gone:
        shown = ", ".join(str(p) for p in gone[:4]) + (" ..." if len(gone) > 4 else "")
        print(f"{label}: {total - len(gone)} of {total} files in {data_dir()}; missing {shown}", flush=True)
    else:
        print(f"{label}: all {total} files in {data_dir()}", flush=True)
    return gone


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m ssm_dan.data.files", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="all", help="bcic2a | bcic2b | zhou2016 | openbmi | all")
    parser.add_argument("--no-car", action="store_true", help="OpenBMI files without CAR")
    parser.add_argument("--check", action="store_true", help="only list what is missing (default)")
    parser.add_argument("--download", action="store_true", help="download missing BCI IV / Zhou2016 files")
    parser.add_argument("--link-from", default="", help=f"directories ({os.pathsep}-separated) to take files from")
    args = parser.parse_args(argv)
    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        if args.link_from:
            linked = link_from(dataset, [d for d in args.link_from.split(os.pathsep) if d], args.no_car)
            if linked:
                print(f"{dataset}: linked {linked} files", flush=True)
        if args.download and missing(dataset):
            if dataset in ("bcic2a", "bcic2b"):
                from .bciciv import prepare
                prepare(dataset)
            elif dataset == "zhou2016":
                download(dataset)
        report(dataset, args.no_car)
    return 0


if __name__ == "__main__":
    sys.exit(main())
