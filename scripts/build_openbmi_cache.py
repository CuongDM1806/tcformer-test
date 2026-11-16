"""Build the per-subject OpenBMI cache (openbmi_SXX.npz) used by every OpenBMI run.

Each Lee2019_MI subject is downloaded through MOABB (~1.1 GB of MAT files),
preprocessed with the ``openbmi`` settings of configs/datasets.yaml, written to
``--out`` and its raw files are deleted. Subjects already cached with the same
preprocessing are skipped; cache files found under $OPENBMI_CACHE_INPUT (e.g.
an earlier, interrupted output attached on Kaggle) are copied into ``--out``.
Several subjects can be built in parallel (``--workers``).

    python scripts/build_openbmi_cache.py --out openbmi_cache                 # with CAR (paper)
    python scripts/build_openbmi_cache.py --out openbmi_cache_nocar --no-car  # without CAR
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ssm_dan import config as cfg  # noqa: E402


def preprocessing(no_car):
    variants = ["no_car"] if no_car else []
    return cfg.resolve("ssm_dan", "openbmi", variants)["preprocessing"]


def build(subjects, out, no_car):
    os.environ["OPENBMI_CACHE_DIR"] = str(out)
    from ssm_dan.data.openbmi_cache import (
        _cache_dirs, _cache_name, _find_cached, cache_signature, load_openbmi_subject)

    settings = preprocessing(no_car)
    signature = cache_signature(settings)
    _, read_dirs = _cache_dirs(settings)
    started = time.time()
    for index, subject in enumerate(subjects, start=1):
        found = _find_cached(subject, signature, read_dirs)
        if found is not None and found.parent != out:
            shutil.copy2(found, out / _cache_name(subject))
            print(f"  copied S{subject:02d} from {found}", flush=True)
        load_openbmi_subject(subject, settings)
        print(f"[{index}/{len(subjects)}] S{subject:02d} cached | {(time.time() - started) / 60:.1f} min | "
              f"free disk {shutil.disk_usage(out).free / 1e9:.1f} GB", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subjects", default="1-54")
    parser.add_argument("--out", default="openbmi_cache")
    parser.add_argument("--no-car", action="store_true", help="skip the common average reference")
    parser.add_argument("--workers", type=int, default=1, help="subjects built in parallel")
    args = parser.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    subjects = cfg.parse_subjects(args.subjects, list(range(1, 55)))
    if args.workers <= 1:
        build(subjects, out, args.no_car)
    else:
        processes = []
        for worker in range(args.workers):
            part = subjects[worker::args.workers]
            if not part:
                continue
            command = [sys.executable, "-u", __file__, "--out", str(out), "--workers", "1",
                       "--subjects", ",".join(map(str, part))] + (["--no-car"] if args.no_car else [])
            processes.append(subprocess.Popen(command))
            time.sleep(20)   # stagger the MOABB set-up of the workers
        if any(process.wait() != 0 for process in processes):
            sys.exit("A cache worker failed; rerun to finish the missing subjects.")

    files = sorted(out.glob("openbmi_S*.npz"))
    missing = [s for s in subjects if not (out / f"openbmi_S{s:02d}.npz").exists()]
    print(f"{len(files)} cache files ({sum(f.stat().st_size for f in files) / 1e9:.2f} GB) in {out}; "
          f"missing: {missing or 'none'}", flush=True)


if __name__ == "__main__":
    main()
