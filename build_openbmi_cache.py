"""Build the per-subject OpenBMI cache used by the OpenBMI LOSO runs.

Downloads each Lee2019_MI subject through MOABB, preprocesses it with the
`openbmi` settings of configs/hada_tcformer.yaml, writes openbmi_SXX.npz to the
output directory and deletes the raw MAT files. Subjects already cached with
the same preprocessing are skipped; files found under OPENBMI_CACHE_INPUT
(e.g. the output of an earlier, interrupted run attached on Kaggle) are copied
into the output directory, so one output holds the complete cache.

    python build_openbmi_cache.py --subjects 1-54 --out /kaggle/working/openbmi_cache
"""
import argparse
import os
import shutil
import time
from pathlib import Path

import yaml

CONFIG = Path(__file__).resolve().parent / "configs" / "hada_tcformer.yaml"


def parse_subjects(text):
    subjects = []
    for part in text.split(","):
        if "-" in part:
            first, last = part.split("-")
            subjects.extend(range(int(first), int(last) + 1))
        elif part:
            subjects.append(int(part))
    return subjects


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", default="1-54")
    parser.add_argument("--out", default="openbmi_cache")
    args = parser.parse_args()

    out = Path(args.out).resolve()
    os.environ["OPENBMI_CACHE_DIR"] = str(out)
    from utils.load_openbmi import (
        _cache_dirs, _cache_name, _find_cached, cache_signature, load_openbmi_subject,
    )

    preprocessing = yaml.safe_load(CONFIG.read_text())["preprocessing"]["openbmi"]
    subjects = parse_subjects(args.subjects)
    started = time.time()
    signature = cache_signature(preprocessing)
    _, read_dirs = _cache_dirs(preprocessing)
    out.mkdir(parents=True, exist_ok=True)
    for index, subject_id in enumerate(subjects, start=1):
        found = _find_cached(subject_id, signature, read_dirs)
        if found is not None and found.parent != out:
            shutil.copy2(found, out / _cache_name(subject_id))
            print(f"  copied S{subject_id:02d} from {found}", flush=True)
        load_openbmi_subject(subject_id, preprocessing)
        free_gb = shutil.disk_usage(args.out).free / 1e9
        print(f"[{index}/{len(subjects)}] S{subject_id:02d} cached | "
              f"elapsed {(time.time() - started) / 60:.1f} min | free disk {free_gb:.1f} GB",
              flush=True)
    files = sorted(Path(args.out).glob("openbmi_S*.npz"))
    size = sum(f.stat().st_size for f in files) / 1e9
    print(f"Cache complete: {len(files)} subjects, {size:.2f} GB in {args.out}", flush=True)


if __name__ == "__main__":
    main()
