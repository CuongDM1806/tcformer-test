"""Generate the per-shard Kaggle notebooks for the full SSM-DAN PhysioNet run.

Each notebook trains FOLDS_PER_JOB target folds, two at a time on Kaggle T4 x2,
and records the full model (RA + AutoDIAL + IM-TTA) and the same weights scored
without IM-TTA. Run:  python kaggle_physionet106_full/make_notebooks.py
"""
import json
from pathlib import Path

EXCLUDED = [88, 92, 100]
FOLDS_PER_JOB = 4
SUBJECTS = [s for s in range(1, 110) if s not in EXCLUDED]
HERE = Path(__file__).resolve().parent

TEMPLATE = r'''# PhysioNet MI full SSM-DAN LOSO (RA + AutoDIAL + IM-TTA) | Kaggle job __JOB__/__TOTAL__ | targets __TARGETS__
# Settings: Accelerator = GPU T4 x2, Internet = On. Run with "Save Version -> Save & Run All".
# Two folds train at the same time, one per GPU; each fold has its own log file.
# Every fold reports the full model and the same weights without IM-TTA (summary.csv).
# Data: dataset dmc2003vn/physionet-mi-106subj (attach via Add Input, or it is downloaded from the public API).
# Resume: if folds are left over, add this notebook's previous Output via
# "Add Input -> Your Work -> Notebook Output" and run again; finished folds are skipped.
import codecs, os, queue, select, shutil, subprocess, sys, threading, time, zipfile
from pathlib import Path

NOTEBOOK_START = time.monotonic()
BRANCH = 'feature/hada-autodial-physionet106'
REPO_URL = 'https://github.com/CuongDM1806/tcformer-test.git'
RUN_TAG = '__RUN_TAG__'
DATASET = 'physionet'
TARGET_SUBJECTS = __TARGET_LIST__
EXCLUDED = {88, 92, 100}   # must match exclude_subject_ids in the config
SEEDS = [0]
GPUS = [0, 1]              # one fold per GPU at a time
EPOCHS = 65                # same as the source-only PhysioNet runs
BATCH_SIZE = 54            # source + target share a forward pass; 2 accumulated steps = 108
ACCUMULATE = 2
TIME_BUDGET_H = 11.5       # stop starting new folds so the version ends before Kaggle's 12 h limit
FIRST_FOLD_ESTIMATE_H = 4.5  # used until a fold has finished in this run
DATASET_REF = 'dmc2003vn/physionet-mi-106subj'   # public Kaggle Dataset with the decoded epochs
CACHE_FILE = 'physionet_106subj_140d39fd97.npz'  # name must match the config hash

REPO_PATH = Path('/tmp/repo')  # keep the clone and venv out of the saved output
MNE_DATA = Path('/tmp/mne_data')
ARRAY_CACHE = Path('/tmp/physionet_arrays')  # decoded epochs shared by every fold
OUTPUT = Path(f'/kaggle/working/{RUN_TAG}')
OUTPUT.mkdir(parents=True, exist_ok=True)
PREPROCESSING = {'riemannian_alignment': True, 'batch_size': BATCH_SIZE,
                 'accumulate_grad_batches': ACCUMULATE, 'array_cache_dir': str(ARRAY_CACHE)}


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
                    'MNE_DATASETS_BNCI_PATH': str(MNE_DATA),
                    'MNE_DATASETS_EEGBCI_PATH': str(MNE_DATA)})
MNE_DATA.mkdir(parents=True, exist_ok=True)

# Reuse the decoded epochs from the public Kaggle Dataset (made by job 0):
# attached via "Add Input" if present, otherwise downloaded from the public API.
ARRAY_CACHE.mkdir(parents=True, exist_ok=True)
cache_files = list(Path('/kaggle/input').rglob(CACHE_FILE))
if not cache_files:
    archive = Path('/tmp/physionet_dataset.zip')
    run(['curl', '-fL', '--retry', '5', '-o', archive,
         f'https://www.kaggle.com/api/v1/datasets/download/{DATASET_REF}'])
    run(['unzip', '-o', '-q', archive, '-d', '/tmp/physionet_dataset'])
    archive.unlink()
    cache_files = list(Path('/tmp/physionet_dataset').rglob(CACHE_FILE))
if not cache_files:
    raise FileNotFoundError(f'{CACHE_FILE} not found in {DATASET_REF}')
link = ARRAY_CACHE / CACHE_FILE
if not link.exists():
    link.symlink_to(cache_files[0])
print('Using PhysioNet cache:', cache_files[0], flush=True)

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
            f"c['subject_ids']=[{subject}]; c['max_epochs_loso']={EPOCHS}; "
            "c['domain_adaptation']=True; c['evaluate_without_im_tta']=True; "
            "c['log_every_n_epochs']=5; c['enable_progress_bar']=False; "
            "c['model_kwargs']['log_every_n_batches']=0; "
            f"c['preprocessing'][{DATASET!r}].update({PREPROCESSING!r}); "
            f"Path('configs/{config_name}.yaml').write_text(yaml.safe_dump(c, sort_keys=False))"
        )
        subprocess.run([str(PYTHON), '-c', override], cwd=REPO_PATH, check=True)
        # Result folders carry the GPU id, so the two workers never touch each other's files.
        for old in (REPO_PATH / 'results').glob(f'*_GPU{gpu}_*'):
            shutil.rmtree(old, ignore_errors=True)
        with print_lock:
            print(f'===== {prefix} {RUN_TAG} | seed {seed} | epochs {EPOCHS} | '
                  f'BS={BATCH_SIZE}x{ACCUMULATE} | RA + AutoDIAL + IM-TTA =====', flush=True)
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
                    # Checkpoints are large and not needed for the results.
                    if path.is_file() and path.suffix != '.ckpt':
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
        if subject in EXCLUDED:
            print(f'Skip S{subject:03d}: excluded recording.', flush=True)
        elif (OUTPUT / f'seed{seed}_S{subject:03d}_results.zip').exists():
            print(f'Skip seed{seed}_S{subject:03d}: already finished.', flush=True)
        else:
            jobs.put((seed, subject))

threads = [threading.Thread(target=worker, args=(gpu, jobs)) for gpu in GPUS]
for thread in threads:
    thread.start()
    time.sleep(90)   # stagger the starts so the two folds do not load the cache at once
for thread in threads:
    thread.join()

# ---- Summary of this job: full model and the same weights without IM-TTA ----
import json
rows = []
for archive in sorted(OUTPUT.glob('seed*_S*_results.zip')):
    seed = int(archive.name.split('_')[0][4:])
    with zipfile.ZipFile(archive) as zf:
        for name in zf.namelist():
            if '/subjects/subject_' in name and name.endswith('.json'):
                rows.append({**json.loads(zf.read(name)), 'seed': seed})


def fmt(value):
    return '' if value is None else f'{value:.4f}'


with open(OUTPUT / 'summary.csv', 'w') as f:
    f.write('run,seed,subject_id,test_acc,test_kappa,no_tta_acc,no_tta_kappa,best_val_acc,train_time_min\n')
    for r in sorted(rows, key=lambda r: (r['seed'], r['subject_id'])):
        f.write(f"{RUN_TAG},{r['seed']},{r['subject_id']},{r['test_acc']:.4f},{r['test_kappa']:.4f},"
                f"{fmt(r.get('no_tta_acc'))},{fmt(r.get('no_tta_kappa'))},"
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
    shards = [SUBJECTS[i:i + FOLDS_PER_JOB] for i in range(0, len(SUBJECTS), FOLDS_PER_JOB)]
    for old in HERE.glob('job*.ipynb'):
        old.unlink()
    for index, shard in enumerate(shards, start=1):
        first, last = shard[0], shard[-1]
        code = (TEMPLATE
                .replace('__JOB__', str(index))
                .replace('__TOTAL__', str(len(shards)))
                .replace('__TARGETS__', ', '.join(f'S{s}' for s in shard))
                .replace('__RUN_TAG__', f'physionet_4cls_full_s{first:03d}-s{last:03d}')
                .replace('__TARGET_LIST__', repr(shard)))
        path = HERE / f'job{index:02d}_s{first:03d}_s{last:03d}.ipynb'
        path.write_text(json.dumps(notebook(code), indent=1, ensure_ascii=False) + '\n',
                        encoding='utf-8')
    print(f'{len(shards)} notebooks, {len(SUBJECTS)} folds, {FOLDS_PER_JOB} per job')


if __name__ == '__main__':
    main()
