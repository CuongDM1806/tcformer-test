"""Helpers for the Kaggle notebooks in notebooks/ (standard library + PyYAML only).

The notebook kernel (Python 3.11) only orchestrates; training runs in a
Python 3.10 virtual environment with the pinned dependencies.

Results are written to /kaggle/working/results/<dataset>/<run>/seed<k>/ (the
same layout as local runs). To resume an unfinished run, attach the previous
version's output (Add Input -> Your Work -> Notebook Output) and run again:
finished folds are copied over and skipped.
"""
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_URL = "https://github.com/CuongDM1806/tcformer-test.git"
VENV = Path("/tmp/ssm-dan-venv")
WORKING = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
RESULTS = WORKING / "results"
TORCH = ["torch==2.7.1", "torchvision==0.22.1", "--index-url", "https://download.pytorch.org/whl/cu126"]


def sh(command, cwd=None, env=None):
    command = [str(part) for part in command]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def setup(extra_packages=()):
    """Create the Python 3.10 environment (once per session); return its python."""
    python = VENV / "bin" / "python"
    if not shutil.which("uv"):
        sh([sys.executable, "-m", "pip", "install", "-q", "uv"])
    uv = shutil.which("uv") or "uv"
    if not python.exists():
        sh([uv, "venv", "--python", "3.10", VENV])
        sh([uv, "pip", "install", "--python", python, *TORCH])
        sh([uv, "pip", "install", "--python", python, "-r", repo_root() / "requirements.txt"])
    if extra_packages:
        sh([uv, "pip", "install", "--python", python, *extra_packages])
    subprocess.run(["nvidia-smi", "-L"], check=False)
    return python


def environment():
    env = os.environ.copy()
    env.update({
        "PYTHONUNBUFFERED": "1",
        "MPLBACKEND": "Agg",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "MNE_DATA": "/tmp/mne_data",
        "MNE_DATASETS_LEE2019_MI_PATH": "/tmp/mne_data",
        "OPENBMI_CACHE_DIR": "/tmp/openbmi_cache",     # writable; never part of the output
        "OPENBMI_CACHE_INPUT": str(INPUT),            # attached cache Datasets are searched here
        "SSM_DAN_RESULTS": str(RESULTS),
    })
    Path("/tmp/mne_data").mkdir(parents=True, exist_ok=True)
    return env


def _stream(command, log_path, prefix, env, cwd):
    """Run a command, write its output to log_path and echo it with a prefix."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log:
        process = subprocess.Popen([str(c) for c in command], cwd=cwd, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1)
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(prefix, line, end="", flush=True)
    return process.wait()


def resume(dataset, name, seed):
    """Copy folds finished by an attached earlier output into /kaggle/working/results."""
    target = RESULTS / dataset / name / f"seed{seed}"
    copied = 0
    for previous in INPUT.rglob(f"results/{dataset}/{name}/seed{seed}"):
        for path in previous.rglob("*"):
            destination = target / path.relative_to(previous)
            if path.is_file() and not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
                copied += path.parent.name == "subjects"
    if copied:
        print(f"Resumed {copied} finished folds from an attached output.", flush=True)
    return target


def check_openbmi_cache(no_car=False):
    """Fail early unless all 54 OpenBMI cache files with the right preprocessing are attached."""
    import json
    import numpy as np

    wanted = not no_car
    found = set()
    for path in INPUT.rglob("openbmi_S*.npz"):
        with np.load(path, allow_pickle=False) as data:
            if json.loads(str(data["signature"])).get("common_average_reference") is wanted:
                found.add(int(path.stem[-2:]))
    missing = sorted(set(range(1, 55)) - found)
    if missing:
        name = "openbmi-nocar-cuong" if no_car else "openbmi-dataset-cuong"
        raise FileNotFoundError(
            f"OpenBMI cache {'without' if no_car else 'with'} CAR is missing subjects {missing}. "
            f"Attach the Kaggle Dataset {name} (Add Input), or build it with "
            "notebooks/kaggle_openbmi_cache.ipynb.")
    print(f"OpenBMI cache ({'no CAR' if no_car else 'CAR'}): 54 subjects attached.", flush=True)


def paper_process_mode(dataset, variants):
    """How the paper's run of this experiment was launched.

    TorchScript optimizes the selective scan after its first calls in a
    process, so the first fold of a process can differ in the last bits from
    later folds. The BCI IV and Zhou2016 runs trained all folds in one process
    on one GPU; the OpenBMI and session-specific-RA runs used one process per fold.
    """
    return "per_fold" if dataset == "openbmi" or "per_session_ra" in variants else "one_process"


def _command(python, method, dataset, variant, subjects, seed, gpu, overrides, budget=None):
    command = [python, "-u", "-m", "ssm_dan.run", "--method", method, "--dataset", dataset,
               "--subjects", subjects, "--seed", seed, "--gpu", gpu]
    if variant:
        command += ["--variant", variant]
    for item in overrides:
        command += ["--set", item]
    if budget is not None:
        command += ["--time-budget-h", f"{budget:.2f}"]
    return command


def run(method="ssm_dan", dataset="bcic2a", variant="", subjects="all", seed=0, gpus=(0, 1),
        overrides=(), mode="paper", time_budget_h=11.5, first_fold_estimate_h=1.5):
    """Train every requested fold that is not finished yet; return the run directory.

    mode="paper": launch folds as the paper's run did (see paper_process_mode);
    "one_process": all folds sequentially in one process on gpus[0];
    "per_fold": one process per fold, one fold per GPU at a time.
    """
    sys.path.insert(0, str(repo_root()))
    from ssm_dan import config as cfg, results
    from ssm_dan.data import SUBJECT_IDS

    started = time.monotonic()
    python = setup(["transformers==4.48.3"] if method == "eegencoder" else [])
    variants = [v.strip() for v in variant.split(",") if v.strip()]
    config = cfg.resolve(method, dataset, variants, overrides, seed=seed)
    if dataset == "openbmi":
        check_openbmi_cache(no_car=not config["preprocessing"].get("common_average_reference", True))
    out_dir = resume(dataset, config["name"], seed)
    targets = cfg.parse_subjects(subjects, SUBJECT_IDS[dataset])
    pending = [s for s in targets if not results.is_done(out_dir, s)]
    if mode == "paper":
        mode = paper_process_mode(dataset, variants)
    print(f"{config['name']} | {dataset} | seed {seed} | {len(pending)} of {len(targets)} folds to train | "
          f"mode {mode} | GPUs {list(gpus) if mode == 'per_fold' else [gpus[0]]} | output {out_dir}",
          flush=True)
    env = environment()

    if mode == "one_process":
        if pending:
            budget = time_budget_h - (time.monotonic() - started) / 3600
            command = _command(python, method, dataset, variant, ",".join(map(str, pending)), seed,
                               gpus[0], overrides, budget)
            _stream(command, out_dir / "logs" / f"run_{time.strftime('%Y%m%d_%H%M%S')}.log",
                    f"[GPU{gpus[0]}]", env, repo_root())
    elif mode == "per_fold":
        todo, fold_hours, lock = queue.Queue(), [], threading.Lock()
        for subject in pending:
            todo.put(subject)

        def worker(gpu):
            while True:
                try:
                    subject = todo.get_nowait()
                except queue.Empty:
                    return
                remaining = time_budget_h - (time.monotonic() - started) / 3600
                if (max(fold_hours) if fold_hours else first_fold_estimate_h) > remaining:
                    print(f"[GPU{gpu}] S{subject:02d} deferred: {remaining:.1f} h left", flush=True)
                    continue
                fold_started = time.monotonic()
                code = _stream(_command(python, method, dataset, variant, subject, seed, gpu, overrides),
                               out_dir / "logs" / f"S{subject:02d}.log", f"[GPU{gpu} S{subject:02d}]",
                               env, repo_root())
                with lock:
                    if code == 0:
                        fold_hours.append((time.monotonic() - fold_started) / 3600)
                    else:
                        print(f"[GPU{gpu}] S{subject:02d} FAILED (exit {code}); see logs/S{subject:02d}.log",
                              flush=True)

        threads = []
        for gpu in gpus:
            threads.append(threading.Thread(target=worker, args=(gpu,)))
            threads[-1].start()
            time.sleep(60 if len(gpus) > 1 else 0)   # stagger data loading of parallel folds
        for thread in threads:
            thread.join()
    else:
        raise ValueError("mode must be 'paper', 'one_process' or 'per_fold'.")

    print(results.summarize(out_dir) or "No finished folds.", flush=True)
    left = [s for s in targets if not results.is_done(out_dir, s)]
    if left:
        print(f"NOT FINISHED: subjects {left}. Attach this version's output "
              "(Add Input -> Your Work -> Notebook Output) and run again.", flush=True)
    else:
        print(f"All {len(targets)} folds of {config['name']} on {dataset} are finished.", flush=True)
    return out_dir


def build_openbmi_cache(no_car=False, subjects="1-54", workers=3):
    """Download and preprocess OpenBMI into /kaggle/working/openbmi_cache[_nocar]."""
    python = setup()
    out = WORKING / ("openbmi_cache_nocar" if no_car else "openbmi_cache")
    command = [python, "-u", repo_root() / "scripts" / "build_openbmi_cache.py", "--out", out,
               "--subjects", subjects, "--workers", workers] + (["--no-car"] if no_car else [])
    code = _stream(command, WORKING / "openbmi_cache_build.log", "", environment(), repo_root())
    if code != 0:
        print("Some subjects are missing: attach this output and run again.", flush=True)
    return out


# --------------------------------------------------------------------------- parity
# Development commit (or branch) that produced each run reported in the paper.
LEGACY_REFS = {
    "full": "b7574e2d",
    "no_ra": "feature/hada-autodial-zhou-mi-preprocessing-no-ra",
    "no_tta": "feature/hada-autodial-zhou-mi-preprocessing-no-im-tta",
    "no_da": "feature/hada-full-mamba-no-da",
    "source_only": "feature/full-mamba-source-only",
    "per_session_ra": "feature/ssm-dan-per-session-ra",
    "eegconformer": "feature/sota-baselines-bcic2a",
    "atcnet": "feature/sota-baselines-bcic2a",
    "eegencoder": "feature/sota-baselines-bcic2a",
    "satransnet": "feature/sota-baselines-bcic2a",
    "tcformer": "feature/tcformer-original",
}
OPENBMI_REFS = {"full": "feature/hada-autodial-openbmi54", "no_ra": "feature/hada-autodial-openbmi54",
                "no_da": "feature/hada-autodial-openbmi54", "source_only": "feature/hada-autodial-openbmi54",
                "per_session_ra": "feature/ssm-dan-per-session-ra"}


def legacy_ref(label):
    dataset, run = label.split("/")
    return (OPENBMI_REFS if dataset == "openbmi" else LEGACY_REFS)[run]


def parity(labels=None, steps=6):
    """Compare data, initial weights, trained weights and adapted test logits with the old code."""
    import json

    python = setup(["transformers==4.48.3"])
    root = repo_root()
    fixture = root / "tests" / "fixtures" / "paper_configs.json"
    all_labels = sorted(json.loads(fixture.read_text())["runs"])
    labels = labels or [label for label in all_labels if not label.startswith("openbmi")]
    if any(label.startswith("openbmi") for label in labels):
        check_openbmi_cache()
    mirror = Path("/tmp/legacy-repo")
    if not mirror.exists():
        sh(["git", "clone", "--no-checkout", REPO_URL, mirror])
    env = environment()
    out = WORKING / "parity"
    rows = []
    for label in labels:
        ref = legacy_ref(label)
        checkout = Path("/tmp/legacy") / ref.replace("/", "_")
        if not checkout.exists():
            remote_ref = ref if len(ref) == 8 and "/" not in ref else f"origin/{ref}"
            sh(["git", "-C", mirror, "worktree", "add", "--detach", checkout, remote_ref])
        reports = {}
        for side, cwd in (("legacy", checkout), ("new", root)):
            report = out / f"{label.replace('/', '_')}_{side}.json"
            command = [python, root / "tests" / "parity" / "probe.py", "--side", side, "--label", label,
                       "--fixture", fixture, "--steps", steps, "--out", report]
            code = _stream(command, report.with_suffix(".log"), f"[{label} {side}]", env, cwd)
            reports[side] = json.loads(report.read_text()) if code == 0 else None
        old, new = reports["legacy"], reports["new"]
        if old is None or new is None:
            rows.append((label, "ERROR", "", "", "", ""))
            continue
        checks = [old["data"] == new["data"], old["init"] == new["init"],
                  old["trained"] == new["trained"], old["test_logits"] == new["test_logits"]]
        rows.append((label, "SAME" if all(checks) else "DIFFERENT", *["ok" if c else "DIFF" for c in checks]))

    print(f"\n{'run':28s} {'result':10s} {'data':6s} {'init':6s} {'trained':8s} {'adapted test logits'}")
    for row in rows:
        print(f"{row[0]:28s} {row[1]:10s} {row[2]:6s} {row[3]:6s} {row[4]:8s} {row[5]}")
    (out / "parity_summary.json").write_text(json.dumps(rows, indent=1))
    return rows
