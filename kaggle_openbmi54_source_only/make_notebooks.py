"""Generate the Kaggle notebooks for the source-only OpenBMI (Lee2019_MI) LOSO baseline.

Same backbone, data protocol, epochs and batch size as kaggle_openbmi54_full, with every
target-side component switched off: no Riemannian alignment, no domain adaptation
(no target batches, AutoDIAL, InfoMax, adversarial or MMD losses) and no IM-TTA.
The target subject is only seen at test time (session 2), with source BatchNorm statistics.
Three jobs cover the 54 targets (S1-S18, S19-S37, S38-S54), two folds at a time on T4 x2.
Run:  python kaggle_openbmi54_full/make_notebooks.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
JOBS = [(1, 18), (19, 37), (38, 54)]

TEMPLATE = r'''# OpenBMI (Lee2019_MI) SSM-DAN LOSO, source-only (no RA, no DA, no IM-TTA) | Kaggle job __JOB__/__TOTAL__ | targets __TARGETS__
# Settings: Accelerator = GPU T4 x2, Internet = On. Run with "Save Version -> Save & Run All".
# Two folds train at the same time, one per GPU; each fold has its own log file.
# Every fold trains on labeled source data of the other 53 subjects only; the target is used only for testing (summary.csv).
# Data: Kaggle Dataset openbmi-dataset-cuong (54 preprocessed openbmi_S*.npz files). Attach it via "Add Input"
# (private: the owner account, or an account it is shared with); if it is public it is downloaded instead.
# Resume: if folds are left over, add this notebook's previous Output via
# "Add Input -> Your Work -> Notebook Output" and run again; finished folds are skipped.
import codecs, os, queue, select, shutil, subprocess, sys, threading, time, zipfile
from pathlib import Path

NOTEBOOK_START = time.monotonic()
BRANCH = 'feature/hada-autodial-openbmi54'
REPO_URL = 'https://github.com/CuongDM1806/tcformer-test.git'
RUN_TAG = '__RUN_TAG__'
DATASET = 'openbmi'
TARGET_SUBJECTS = __TARGET_LIST__
SEEDS = [0]
GPUS = [0, 1]              # one fold per GPU at a time
EPOCHS = 125               # max_epochs_loso_openbmi in the config; batch size 48 as configured
# Source-only: every target-side component of the full model is switched off.
DOMAIN_ADAPTATION = False  # no target batches during training
PREPROCESSING = {'riemannian_alignment': False}
MODEL_OVERRIDES = {'feature_alignment': 'none', 'target_im_weight': 0.0, 'adversarial_weight': 0.0,
                   'mmd_weight': 0.0, 'temporal_mmd_weight': 0.0,
                   'im_tta_steps': 0, 'recalibrate_target_statistics': False}
TIME_BUDGET_H = 11.5       # stop starting new folds so the version ends before Kaggle's 12 h limit
FIRST_FOLD_ESTIMATE_H = 1.5  # used until a fold has finished in this run

REPO_PATH = Path('/tmp/repo')  # keep the clone and venv out of the saved output
MNE_DATA = Path('/tmp/mne_data')
OPENBMI_CACHE = Path('/tmp/openbmi_cache')  # writable cache dir; the attached Dataset is read from /kaggle/input
DATASET_REF = 'nguyncuongv186/openbmi-dataset-cuong'
DOWNLOADED = Path('/tmp/openbmi_dataset')  # used only when the Dataset is not attached
OUTPUT = Path(f'/kaggle/working/{RUN_TAG}')
OUTPUT.mkdir(parents=True, exist_ok=True)


def run(command, cwd=None, env=None):
    command = list(map(str, command))
    print('+', ' '.join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


# ---- Setup ----
run([sys.executable, '-m', 'pip', 'install', '-q', 'uv'])
if (REPO_PATH / '.git').is_dir():
    run(['git', 'fetch', 'origin', BRANCH], cwd=REPO_PATH)
    run(['git', 'reset', '--hard', f'origin/{BRANCH}'], cwd=REPO_PATH)
else:
    run(['git', 'clone', '--branch', BRANCH, '--single-branch', REPO_URL, REPO_PATH])
run(['git', 'log', '-1', '--oneline'], cwd=REPO_PATH)
UV = shutil.which('uv') or 'uv'
run([UV, 'venv', '--clear', '--python', '3.10', '.venv'], cwd=REPO_PATH)
PYTHON = REPO_PATH / '.venv/bin/python'
run([UV, 'pip', 'install', '--python', PYTHON, 'torch==2.7.1', 'torchvision==0.22.1',
     '--index-url', 'https://download.pytorch.org/whl/cu126'])
run([UV, 'pip', 'install', '--python', PYTHON, '-r', 'requirements.txt'], cwd=REPO_PATH)
run(['nvidia-smi'])

environment = os.environ.copy()
environment.update({'PYTHONUNBUFFERED': '1', 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
                    'MPLBACKEND': 'Agg', 'MNE_DATA': str(MNE_DATA),
                    'MNE_DATASETS_LEE2019_MI_PATH': str(MNE_DATA),
                    'OPENBMI_CACHE_DIR': str(OPENBMI_CACHE), 'OPENBMI_CACHE_INPUT': f'/kaggle/input{os.pathsep}{DOWNLOADED}'})
MNE_DATA.mkdir(parents=True, exist_ok=True)

# Read the preprocessed OpenBMI cache; the raw 61 GB is never downloaded here.
def find_cache():
    cached = [path for root in (Path('/kaggle/input'), DOWNLOADED) if root.is_dir()
              for path in sorted(root.rglob('openbmi_S*.npz'))]
    return cached, sorted(set(range(1, 55)) - {int(path.stem[-2:]) for path in cached})


cached, missing = find_cache()
if missing:
    # Not attached: the public download API works only if the Dataset is public.
    archive = Path('/tmp/openbmi_dataset.zip')
    fetched = subprocess.run(['curl', '-fL', '--retry', '3', '-o', str(archive),
                              f'https://www.kaggle.com/api/v1/datasets/download/{DATASET_REF}'])
    if fetched.returncode == 0:
        run(['unzip', '-o', '-q', archive, '-d', DOWNLOADED])
        archive.unlink()
        cached, missing = find_cache()
    else:
        print(f'{DATASET_REF} is not attached and could not be downloaded (private?).', flush=True)
if missing:
    print('Contents of /kaggle/input (depth 3):', flush=True)
    for path in sorted(Path('/kaggle/input').glob('*/*/*'))[:40] or sorted(Path('/kaggle/input').glob('**/*'))[:40]:
        print('  ', path, flush=True)
    raise FileNotFoundError(f'OpenBMI cache incomplete, missing subjects {missing}: '
                            'attach the Dataset openbmi-dataset-cuong via "Add Input" (share it with this account if it is private).')
print(f'Using {len(cached)} OpenBMI cache files from {cached[0].parent}', flush=True)

# ---- Two folds in parallel, one per GPU; each fold is archived as soon as it ends ----
print_lock = threading.Lock()
failures, deferred, fold_hours = [], [], []


def stream(command, log_path, prefix):
    command = list(map(str, command))
    with print_lock:
        print(prefix, '+', ' '.join(command), flush=True)
    process = subprocess.Popen(command, cwd=REPO_PATH, env=environment, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, bufsize=0)
    decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
    started, pending_text = time.monotonic(), ''
    with open(log_path, 'w', encoding='utf-8') as log:
        while True:
            ready, _, _ = select.select([process.stdout], [], [], 60)
            if ready:
                chunk = os.read(process.stdout.fileno(), 4096)
                if not chunk:
                    break
                text = decoder.decode(chunk)
                log.write(text); log.flush()
                *lines, pending_text = (pending_text + text).split('\n')
                with print_lock:
                    for line in lines:
                        print(prefix, line, flush=True)
            elif process.poll() is None:
                beat = f'[heartbeat] elapsed={(time.monotonic() - started) / 60:.1f}m'
                log.write(beat + '\n'); log.flush()
                with print_lock:
                    print(prefix, beat, flush=True)
            else:
                break
    if pending_text:
        with print_lock:
            print(prefix, pending_text, flush=True)
    return process.wait()


def worker(gpu, jobs):
    config_name = f'hada_tcformer_gpu{gpu}'
    while True:
        try:
            seed, subject = jobs.get_nowait()
        except queue.Empty:
            return
        fold = f'seed{seed}_S{subject:03d}'
        prefix = f'[GPU{gpu} S{subject:03d}]'
        estimate = max(fold_hours) if fold_hours else FIRST_FOLD_ESTIMATE_H
        remaining = TIME_BUDGET_H - (time.monotonic() - NOTEBOOK_START) / 3600
        if estimate > remaining:
            deferred.append(fold)
            with print_lock:
                print(f'{prefix} deferred: {remaining:.1f} h left, a fold takes ~{estimate:.1f} h', flush=True)
            continue
        fold_start = time.monotonic()
        override = (
            "import yaml; from pathlib import Path; "
            "c=yaml.safe_load(Path('configs/hada_tcformer.yaml').read_text()); "
            f"c['subject_ids']=[{subject}]; c['max_epochs_loso_openbmi']={EPOCHS}; "
            f"c['domain_adaptation']={DOMAIN_ADAPTATION}; c['evaluate_without_im_tta']=False; "
            f"c['model_kwargs'].update({MODEL_OVERRIDES!r}); "
            f"c['preprocessing'][{DATASET!r}].update({PREPROCESSING!r}); "
            "c['log_every_n_epochs']=5; c['enable_progress_bar']=False; "
            "c['model_kwargs']['log_every_n_batches']=0; "
            f"Path('configs/{config_name}.yaml').write_text(yaml.safe_dump(c, sort_keys=False))"
        )
        subprocess.run([str(PYTHON), '-c', override], cwd=REPO_PATH, check=True)
        # Result folders carry the GPU id, so the two workers never touch each other's files.
        for old in (REPO_PATH / 'results').glob(f'*_GPU{gpu}_*'):
            shutil.rmtree(old, ignore_errors=True)
        with print_lock:
            print(f'===== {prefix} {RUN_TAG} | seed {seed} | epochs {EPOCHS} | '
                  'source-only: no RA, no DA, no IM-TTA =====', flush=True)
        code = stream([PYTHON, '-u', 'train_pipeline.py', '--model', config_name,
                       '--dataset', DATASET, '--loso', '--gpu_id', gpu, '--seed', seed],
                      OUTPUT / f'{fold}.log', prefix)
        if code != 0:
            failures.append(fold)
            with print_lock:
                print(f'!!!!! {prefix} failed (exit {code}); see {OUTPUT / f"{fold}.log"}', flush=True)
            continue
        archive = OUTPUT / f'{fold}_results.zip'
        with zipfile.ZipFile(f'{archive}.part', 'w', zipfile.ZIP_DEFLATED) as zf:
            for result_dir in (REPO_PATH / 'results').glob(f'*_GPU{gpu}_*'):
                for path in result_dir.rglob('*'):
                    if path.is_file():
                        zf.write(path, path.relative_to(REPO_PATH))
        os.replace(f'{archive}.part', archive)
        fold_hours.append((time.monotonic() - fold_start) / 3600)
        with print_lock:
            print(f'Saved {archive} ({fold_hours[-1]:.2f} h)', flush=True)


# Folds finished by an earlier version of this notebook (attached as input) are carried over.
for previous in Path('/kaggle/input').rglob('seed*_S*_results.zip'):
    if previous.parent.name == RUN_TAG and not (OUTPUT / previous.name).exists():
        shutil.copy2(previous, OUTPUT / previous.name)
        log = previous.with_name(previous.name.replace('_results.zip', '.log'))
        if log.exists():
            shutil.copy2(log, OUTPUT / log.name)
        print(f'Resumed from earlier run: {previous.name}', flush=True)
for previous in Path('/kaggle/input').rglob('seed*_S*_results'):
    # Kaggle may show an attached zip as an extracted folder; zip it again.
    if previous.is_dir() and previous.parent.name == RUN_TAG and not (OUTPUT / f'{previous.name}.zip').exists():
        shutil.make_archive(str(OUTPUT / previous.name), 'zip', root_dir=previous)
        print(f'Resumed from earlier run (extracted): {previous.name}', flush=True)

jobs = queue.Queue()
for seed in SEEDS:
    for subject in TARGET_SUBJECTS:
        if (OUTPUT / f'seed{seed}_S{subject:03d}_results.zip').exists():
            print(f'Skip seed{seed}_S{subject:03d}: already finished.', flush=True)
        else:
            jobs.put((seed, subject))

threads = [threading.Thread(target=worker, args=(gpu, jobs)) for gpu in GPUS]
for thread in threads:
    thread.start()
    time.sleep(90)   # stagger the starts so the two folds do not load the cache at once
for thread in threads:
    thread.join()

# ---- Summary of this job (source-only) ----
import json
rows = []
for archive in sorted(OUTPUT.glob('seed*_S*_results.zip')):
    seed = int(archive.name.split('_')[0][4:])
    with zipfile.ZipFile(archive) as zf:
        for name in zf.namelist():
            if '/subjects/subject_' in name and name.endswith('.json'):
                rows.append({**json.loads(zf.read(name)), 'seed': seed})


with open(OUTPUT / 'summary.csv', 'w') as f:
    f.write('run,seed,subject_id,test_acc,test_kappa,best_val_acc,train_time_min\n')
    for r in sorted(rows, key=lambda r: (r['seed'], r['subject_id'])):
        f.write(f"{RUN_TAG},{r['seed']},{r['subject_id']},{r['test_acc']:.4f},{r['test_kappa']:.4f},"
                f"{r['best_val_acc']:.4f},{r['train_time_min']:.1f}\n")
print(open(OUTPUT / 'summary.csv').read())
left = sorted(deferred + failures)
if left:
    # No exception here, so this version still ends normally and its Output is saved.
    print(f'NOT FINISHED: {left} (failed: {sorted(failures)}).')
    print("Re-run: Add Input -> Your Work -> Notebook Output -> this version, then Save & Run All.")
else:
    print(f'All folds of {RUN_TAG} finished.')
'''


def notebook(code):
    return {
        "cells": [{"cell_type": "code", "execution_count": None, "metadata": {},
                   "outputs": [], "source": code.splitlines(keepends=True)}],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    for old in HERE.glob('job*.ipynb'):
        old.unlink()
    for index, (first, last) in enumerate(JOBS, start=1):
        code = (TEMPLATE
                .replace('__JOB__', str(index))
                .replace('__TOTAL__', str(len(JOBS)))
                .replace('__TARGETS__', f'S{first}-S{last}')
                .replace('__RUN_TAG__', f'openbmi_source_only_s{first:03d}-s{last:03d}')
                .replace('__TARGET_LIST__', f'list(range({first}, {last} + 1))'))
        path = HERE / f'job{index:02d}_s{first:03d}_s{last:03d}.ipynb'
        path.write_text(json.dumps(notebook(code), indent=1, ensure_ascii=False) + '\n',
                        encoding='utf-8')
        print(f'wrote {path.name}')


if __name__ == '__main__':
    main()
