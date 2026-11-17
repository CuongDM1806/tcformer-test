"""Launch experiments on any machine: a workstation, a GPU server, Kaggle or Colab.

``python -m ssm_dan.run`` trains folds in the current Python environment. This
launcher adds what long runs need on shared or hosted machines:

* a Python 3.10 environment with the pinned dependencies when the current one
  cannot train (e.g. Kaggle and Colab run Python 3.11+, MOABB 0.5 needs 3.10);
* several GPUs: one fold per GPU at a time (``--mode per_fold``);
* resuming: folds finished in an earlier session are copied from the input
  directories (e.g. an attached Kaggle output, a mounted Drive) and skipped;
* a time budget (hosted sessions end after a fixed time);
* the data: every dataset is read from ``data/`` in the repository (see
  data/README.md). Files already there are used as they are; missing ones are
  linked from the input directories (e.g. an attached Kaggle Dataset), then
  fetched from the data branch ``<dataset>-data`` of this repository, or downloaded.

Paths follow the platform (detected automatically, or ``--platform``):

    platform  outputs (results/)            searched for earlier results and data files
    kaggle    /kaggle/working                /kaggle/input
    colab     /content                       /content/drive/MyDrive (if mounted)
    local     current directory              $SSM_DAN_INPUT (os.pathsep-separated), if set

Command line (same options as ssm_dan.run, plus --gpus, --mode, --time-budget-h):

    python -m ssm_dan.launch --method ssm_dan --dataset openbmi --gpus 0,1
    python -m ssm_dan.launch --method ssm_dan --dataset bcic2a --variant no_ra --mode one_process
    python -m ssm_dan.launch --data --dataset bcic2a         # only make the data available

From a notebook:  from ssm_dan import launch; launch.run("ssm_dan", "bcic2a", gpus=[0, 1])
Standard library + PyYAML only, so it also runs in a notebook kernel without PyTorch.
"""
import argparse
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TORCH = ["torch==2.7.1", "torchvision==0.22.1", "--index-url", "https://download.pytorch.org/whl/cu126"]


@dataclass
class Platform:
    name: str
    work: Path                                     # results/ and caches are written here
    inputs: list = field(default_factory=list)     # searched for earlier results and caches
    scratch: Path = Path("/tmp")                   # downloads and the Python environment

    @property
    def results(self) -> Path:
        return Path(os.environ.get("SSM_DAN_RESULTS", self.work / "results"))


def detect(name=None) -> Platform:
    if name is None:
        if Path("/kaggle/working").is_dir():
            name = "kaggle"
        elif os.environ.get("COLAB_RELEASE_TAG") or "google.colab" in sys.modules:
            name = "colab"
        else:
            name = "local"
    if name == "kaggle":
        return Platform("kaggle", Path("/kaggle/working"), [Path("/kaggle/input")])
    if name == "colab":
        drive = Path("/content/drive/MyDrive")
        return Platform("colab", Path("/content"), [drive] if drive.is_dir() else [])
    if name == "local":
        inputs = [Path(p) for p in os.environ.get("SSM_DAN_INPUT", "").split(os.pathsep) if p]
        return Platform("local", Path.cwd(), inputs, Path(os.environ.get("TMPDIR", "/tmp")))
    raise ValueError(f"Unknown platform {name!r}: kaggle | colab | local")


def sh(command, cwd=None, env=None):
    command = [str(part) for part in command]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _can_train(python) -> bool:
    probe = "import sys, torch, moabb, braindecode, pytorch_lightning; assert sys.version_info[:2] == (3, 10)"
    return subprocess.run([str(python), "-c", probe], capture_output=True).returncode == 0


def python_for_training(platform: Platform, extra_packages=()):
    """The current interpreter if it can train; otherwise a Python 3.10 environment (created once)."""
    if _can_train(sys.executable):
        python = Path(sys.executable)
    else:
        venv = platform.scratch / "ssm-dan-venv"
        python = venv / "bin" / "python"
        if not shutil.which("uv"):
            sh([sys.executable, "-m", "pip", "install", "-q", "uv"])
        uv = shutil.which("uv") or "uv"
        if not python.exists():
            sh([uv, "venv", "--python", "3.10", venv])
            sh([uv, "pip", "install", "--python", python, *TORCH])
            sh([uv, "pip", "install", "--python", python, "-r", ROOT / "requirements.txt"])
    if extra_packages:
        sh([python, "-m", "pip", "install", "-q", *extra_packages] if python == Path(sys.executable)
           else [shutil.which("uv") or "uv", "pip", "install", "--python", python, *extra_packages])
    subprocess.run(["nvidia-smi", "-L"], check=False)
    return python


def environment(platform: Platform):
    """Environment of the training processes (data in data/, see ssm_dan.data.files)."""
    from ssm_dan.data import files

    env = os.environ.copy()
    env.update({
        "PYTHONUNBUFFERED": "1",
        "MPLBACKEND": "Agg",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "SSM_DAN_RESULTS": str(platform.results),
        # Raw OpenBMI recordings (only while building data/openbmi*) go to scratch space.
        "MNE_DATASETS_LEE2019_MI_PATH": str(platform.scratch / "openbmi_raw"),
    })
    return files.configure(env)


def _stream(command, log_path, prefix, env):
    """Run a command, write its output to log_path and echo it with a prefix."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log:
        process = subprocess.Popen([str(c) for c in command], cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1)
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(prefix, line, end="", flush=True)
    return process.wait()


def resume(platform: Platform, dataset, name, seed):
    """Copy folds finished in an earlier session (found in the input directories) into results/."""
    target = platform.results / dataset / name / f"seed{seed}"
    copied = 0
    for root in platform.inputs:
        for previous in root.rglob(f"results/{dataset}/{name}/seed{seed}"):
            if previous.resolve() == target.resolve():
                continue
            for path in previous.rglob("*"):
                destination = target / path.relative_to(previous)
                if path.is_file() and not destination.exists():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, destination)
                    copied += path.parent.name == "subjects"
    if copied:
        print(f"Resumed {copied} finished folds from an earlier session.", flush=True)
    return target


def search_directories(platform: Platform):
    """Where existing data files are looked for before anything is downloaded."""
    return [*platform.inputs, platform.work, Path.home() / "mne_data"]


def ensure_data(dataset, no_car=False, workers=3, platform=None, env=None, python=None):
    """Make every file of ``dataset`` available in data/; files already there are used as they are.

    BCI IV-2a/2b: data/BCICIV_2x/mat is built from the official competition files (GDF + true
    labels) in data/BCICIV_2x/{gdf,true_labels} or the input directories (e.g. a Kaggle Dataset
    added with Add Input), else downloaded from bbci.de (ssm_dan.data.bciciv). Zhou2016, OpenBMI:
    1. files in data/ are used;
    2. missing files with the expected names are linked from the input directories
       (attached Kaggle Datasets or outputs, a mounted Drive, $SSM_DAN_INPUT, ~/mne_data);
    3. OpenBMI with CAR: the data branch ``openbmi-data`` of this repository;
    4. what is still missing is downloaded from the original source (Zhou2016: figshare;
       OpenBMI: raw recordings, preprocessed, ``workers`` subjects at a time).
    """
    from ssm_dan.data import files

    platform = platform if isinstance(platform, Platform) else detect(platform)
    env = env or environment(platform)
    label = dataset + (" without CAR" if dataset == "openbmi" and no_car else "")
    if not files.missing(dataset, no_car):
        files.report(dataset, no_car)
        return
    if dataset in ("bcic2a", "bcic2b"):
        python = python or python_for_training(platform)
        _stream([python, "-u", "-m", "ssm_dan.data.bciciv", "--dataset", dataset,
                 "--search", os.pathsep.join(str(d) for d in search_directories(platform))],
                platform.scratch / f"{dataset}_data.log", f"[{dataset}]", env)
        if not files.report(dataset):
            return
        raise FileNotFoundError(f"{dataset}: the MAT files could not be built; see the messages above "
                                "and data/README.md.")
    linked = files.link_from(dataset, search_directories(platform), no_car)
    if linked:
        print(f"{label}: linked {linked} files found in the inputs into {files.data_dir()}", flush=True)
    if dataset == "openbmi" and files.missing(dataset, no_car):
        moved = files.fetch_branch(dataset, files.repository_urls(), platform.scratch, no_car)
        if moved:
            print(f"{label}: {moved} files from branch {files.data_branch(dataset)}", flush=True)
    if files.missing(dataset, no_car):
        print(f"{label}: downloading the missing files", flush=True)
        if dataset == "openbmi":
            subjects = [int(rel.stem[-2:]) for rel in files.missing(dataset, no_car)]
            build_openbmi_cache(no_car, ",".join(map(str, subjects)), workers, platform)
        else:
            python = python or python_for_training(platform)
            _stream([python, "-u", "-m", "ssm_dan.data.files", "--dataset", dataset, "--download"],
                    platform.scratch / f"{dataset}_download.log", f"[{dataset}]", env)
    gone = files.report(dataset, no_car)
    if gone:
        raise FileNotFoundError(
            f"{label}: {len(gone)} files are still missing (first: {gone[0]}). Put them in "
            f"{files.data_dir()} as described in data/README.md, or attach a directory that contains them.")


def paper_process_mode(dataset, variants):
    """How the reported run of this experiment was launched.

    TorchScript optimizes the selective scan after its first calls in a
    process, so the first fold of a process can differ in the last bits from
    later folds. The BCI IV and Zhou2016 runs trained all folds in one process
    on one GPU; the OpenBMI and session-specific-RA runs used one process per fold.
    """
    return "per_fold" if dataset == "openbmi" or "per_session_ra" in variants else "one_process"


def available_gpus():
    """Number of CUDA devices reported by nvidia-smi (None if it is not available)."""
    try:
        listing = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return sum(line.startswith("GPU ") for line in listing.splitlines())


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


def run(method="ssm_dan", dataset="bcic2a", variant="", subjects="all", seed=0, gpus=(0,),
        overrides=(), mode="paper", time_budget_h=None, first_fold_estimate_h=1.5, platform=None):
    """Train every requested fold that is not finished yet; return the run directory.

    mode="paper": launch folds as the reported run did (see paper_process_mode);
    "one_process": all folds sequentially in one process on gpus[0];
    "per_fold": one process per fold, one fold per GPU at a time.
    time_budget_h: do not start a fold that would not end in time (default: 11.5 h
    on Kaggle and Colab, no limit elsewhere).
    """
    sys.path.insert(0, str(ROOT))
    from ssm_dan import config as cfg, results
    from ssm_dan.data import SUBJECT_IDS

    started = time.monotonic()
    platform = platform if isinstance(platform, Platform) else detect(platform)
    if time_budget_h is None and platform.name in ("kaggle", "colab"):
        time_budget_h = 11.5
    python = python_for_training(platform, ["transformers==4.48.3"] if method == "eegencoder" else [])
    env = environment(platform)
    variants = [v.strip() for v in variant.split(",") if v.strip()]
    config = cfg.resolve(method, dataset, variants, overrides, seed=seed)
    ensure_data(dataset, not config["preprocessing"].get("common_average_reference", True),
                platform=platform, env=env, python=python)
    out_dir = resume(platform, dataset, config["name"], seed)
    targets = cfg.parse_subjects(subjects, SUBJECT_IDS[dataset])
    pending = [s for s in targets if not results.is_done(out_dir, s)]
    if mode == "paper":
        mode = paper_process_mode(dataset, variants)
    gpus = list(gpus)
    count = available_gpus()
    if count is not None and count < len(gpus):
        print(f"Only {count} GPU(s) available; using {gpus[:max(count, 1)]}.", flush=True)
        gpus = gpus[:max(count, 1)]
    print(f"[{platform.name}] {config['name']} | {dataset} | seed {seed} | {len(pending)} of {len(targets)} "
          f"folds to train | mode {mode} | GPUs {gpus if mode == 'per_fold' else gpus[:1]} | output {out_dir}",
          flush=True)

    def remaining():
        return None if time_budget_h is None else time_budget_h - (time.monotonic() - started) / 3600

    if mode == "one_process":
        if pending:
            command = _command(python, method, dataset, variant, ",".join(map(str, pending)), seed,
                               gpus[0], overrides, remaining())
            _stream(command, out_dir / "logs" / f"run_{time.strftime('%Y%m%d_%H%M%S')}.log",
                    f"[GPU{gpus[0]}]", env)
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
                left = remaining()
                if left is not None and (max(fold_hours) if fold_hours else first_fold_estimate_h) > left:
                    print(f"[GPU{gpu}] S{subject:02d} deferred: {left:.1f} h left", flush=True)
                    continue
                fold_started = time.monotonic()
                code = _stream(_command(python, method, dataset, variant, subject, seed, gpu, overrides),
                               out_dir / "logs" / f"S{subject:02d}.log", f"[GPU{gpu} S{subject:02d}]", env)
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
        print(f"NOT FINISHED: subjects {left}. Run again (keep the earlier results/ visible) to continue.",
              flush=True)
    else:
        print(f"All {len(targets)} folds of {config['name']} on {dataset} are finished.", flush=True)
    return out_dir


def build_openbmi_cache(no_car=False, subjects="1-54", workers=3, platform=None):
    """Download and preprocess OpenBMI into data/openbmi[_nocar], ``workers`` subjects at a time.

    On Kaggle and Colab the new files are also copied to the saved output
    (<work>/data/openbmi[_nocar]), so a later session can attach them instead of rebuilding.
    """
    sys.path.insert(0, str(ROOT))
    from ssm_dan.config import parse_subjects
    from ssm_dan.data import files

    platform = platform if isinstance(platform, Platform) else detect(platform)
    python = python_for_training(platform)
    env = environment(platform)
    out = files.openbmi_dir(no_car)
    ids = parse_subjects(subjects, list(range(1, 55)))
    threads = []
    for worker in range(max(1, workers)):
        part = ids[worker::max(1, workers)]
        if not part:
            continue
        command = [python, "-u", "-m", "ssm_dan.data.openbmi_cache",
                   "--subjects", ",".join(map(str, part))] + (["--no-car"] if no_car else [])
        threads.append(threading.Thread(target=_stream, args=(
            command, platform.scratch / f"{out.name}_build_w{worker}.log", f"[w{worker}]", env)))
        threads[-1].start()
        time.sleep(20)   # stagger the MOABB set-up of the workers
    for thread in threads:
        thread.join()
    if platform.name in ("kaggle", "colab"):
        saved = platform.work / "data" / out.name
        saved.mkdir(parents=True, exist_ok=True)
        for subject in ids:
            path = out / f"openbmi_S{subject:02d}.npz"
            if path.is_file() and not (saved / path.name).exists():
                shutil.copy2(path, saved / path.name)
        print(f"OpenBMI files also saved to {saved} (attach this output next time)", flush=True)
    missing = [s for s in ids if not (out / f"openbmi_S{s:02d}.npz").exists()]
    print(f"OpenBMI files in {out}; missing subjects: {missing or 'none'}", flush=True)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m ssm_dan.launch", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", default="ssm_dan")
    parser.add_argument("--dataset", default="bcic2a")
    parser.add_argument("--variant", default="")
    parser.add_argument("--subjects", default="all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpus", default="0", help="comma-separated CUDA devices, e.g. 0,1")
    parser.add_argument("--mode", default="paper", choices=["paper", "one_process", "per_fold"])
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--time-budget-h", type=float, default=None)
    parser.add_argument("--platform", default=None, choices=["kaggle", "colab", "local"])
    parser.add_argument("--data", action="store_true",
                        help="only make the data of --dataset available (use, link, fetch or download)")
    parser.add_argument("--no-car", action="store_true", help="OpenBMI files without CAR")
    parser.add_argument("--workers", type=int, default=3, help="parallel subjects when building OpenBMI files")
    args = parser.parse_args(argv)
    if args.data:
        ensure_data(args.dataset, args.no_car, args.workers, args.platform)
        return 0
    run(args.method, args.dataset, args.variant, args.subjects, args.seed,
        [int(g) for g in args.gpus.split(",")], args.overrides, args.mode, args.time_budget_h,
        platform=args.platform)
    return 0


if __name__ == "__main__":
    sys.exit(main())
