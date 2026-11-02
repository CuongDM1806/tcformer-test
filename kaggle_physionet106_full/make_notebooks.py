"""Generate the per-shard Kaggle notebooks for the SSM-DAN PhysioNet runs.

Three variants share one code branch and differ only in config overrides:
  full   RA + Stage-1 DA (AutoDIAL, target InfoMax, adversarial, MMD) + IM-TTA
  no_ra  the full model without Riemannian alignment
  no_da  RA + IM-TTA, every Stage-1 DA component removed (the paper's w/o DA)
Each notebook trains the target folds of one 11-subject range, two at a time on Kaggle T4 x2,
and also scores the same weights without IM-TTA.
Run:  python kaggle_physionet106_full/make_notebooks.py
"""
import json
from pathlib import Path

EXCLUDED = [88, 92, 100]
# Same subject ranges as the source-only jobs (S1-S11, S12-S22, ..., S100-S109);
# excluded recordings are skipped inside each job. A job that cannot finish in
# one Kaggle version is resumed from its previous output.
SUBJECTS_PER_JOB = 11
RANGES = [list(range(start, min(start + SUBJECTS_PER_JOB, 110)))
          for start in range(1, 110, SUBJECTS_PER_JOB)]
SUBJECTS = [s for s in range(1, 110) if s not in EXCLUDED]
HERE = Path(__file__).resolve().parent
REPO = HERE.parent

# Every variant uses 54 trials per step and 2 accumulated steps (effective
# batch 108), so the optimizer steps are identical. IM-TTA and testing use the
# whole target subject as one batch (see PhysioNetMILOSO.test_dataloader).
NO_DA_MODEL = {'feature_alignment': 'none', 'target_im_weight': 0.0, 'adversarial_weight': 0.0,
               'mmd_weight': 0.0, 'temporal_mmd_weight': 0.0}
VARIANTS = {
    'full': dict(folder='kaggle_physionet106_full', label='RA + AutoDIAL + IM-TTA',
                 ra=True, da=True, model={}, estimate_h=4.5),
    'no_ra': dict(folder='kaggle_physionet106_no_ra', label='no RA: AutoDIAL + IM-TTA',
                  ra=False, da=True, model={}, estimate_h=4.5),
    'no_da': dict(folder='kaggle_physionet106_no_da', label='no DA: RA + IM-TTA',
                  ra=True, da=False, model=NO_DA_MODEL, estimate_h=3.0),
}

CHECK = dict(folder='kaggle_physionet106_full_batch_check', subjects=[90, 95, 96, 97],
             run_tag='physionet_4cls_full_batch_check')

TEMPLATE = r'''# PhysioNet MI SSM-DAN LOSO, variant __VARIANT__ (__LABEL__) | Kaggle job __JOB__/__TOTAL__ | targets __TARGETS__
# Settings: Accelerator = GPU T4 x2, Internet = On. Run with "Save Version -> Save & Run All".
# Two folds train at the same time, one per GPU; each fold has its own log file.
# Every fold reports this variant and the same weights without IM-TTA (summary.csv).
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
BATCH_SIZE = 54            # 2 accumulated steps = effective batch 108 in every variant
ACCUMULATE = 2
TIME_BUDGET_H = 11.5       # stop starting new folds so the version ends before Kaggle's 12 h limit
FIRST_FOLD_ESTIMATE_H = __ESTIMATE__  # used until a fold has finished in this run
RIEMANNIAN_ALIGNMENT = __RA__
DOMAIN_ADAPTATION = __DA__
MODEL_OVERRIDES = __MODEL__
DATASET_REF = 'dmc2003vn/physionet-mi-106subj'   # public Kaggle Dataset with the decoded epochs
CACHE_FILE = 'physionet_106subj_140d39fd97.npz'  # name must match the config hash

REPO_PATH = Path('/tmp/repo')  # keep the clone and venv out of the saved output
MNE_DATA = Path('/tmp/mne_data')
ARRAY_CACHE = Path('/tmp/physionet_arrays')  # decoded epochs shared by every fold
OUTPUT = Path(f'/kaggle/working/{RUN_TAG}')
OUTPUT.mkdir(parents=True, exist_ok=True)
PREPROCESSING = {'riemannian_alignment': RIEMANNIAN_ALIGNMENT, 'batch_size': BATCH_SIZE,
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
            f"c['domain_adaptation']={DOMAIN_ADAPTATION}; c['evaluate_without_im_tta']=True; "
            f"c['model_kwargs'].update({MODEL_OVERRIDES!r}); "
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
                  f'BS={BATCH_SIZE}x{ACCUMULATE} | __LABEL__ =====', flush=True)
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
    folds = sum(s not in EXCLUDED for shard in RANGES for s in shard)
    for variant, spec in VARIANTS.items():
        folder = REPO / spec['folder']
        folder.mkdir(exist_ok=True)
        for old in folder.glob('job*.ipynb'):
            old.unlink()
        for index, shard in enumerate(RANGES, start=1):
            first, last = shard[0], shard[-1]
            code = (TEMPLATE
                    .replace('__VARIANT__', variant)
                    .replace('__LABEL__', spec['label'])
                    .replace('__ESTIMATE__', str(spec['estimate_h']))
                    .replace('__RA__', str(spec['ra']))
                    .replace('__DA__', str(spec['da']))
                    .replace('__MODEL__', repr(spec['model']))
                    .replace('__JOB__', str(index))
                    .replace('__TOTAL__', str(len(RANGES)))
                    .replace('__TARGETS__', f'S{first}-S{last}')
                    .replace('__RUN_TAG__', f'physionet_4cls_{variant}_s{first:03d}-s{last:03d}')
                    .replace('__TARGET_LIST__', f'list(range({first}, {last} + 1))'))
            path = folder / f'job{index:02d}_s{first:03d}_s{last:03d}.ipynb'
            path.write_text(json.dumps(notebook(code), indent=1, ensure_ascii=False) + '\n',
                            encoding='utf-8')
        print(f'{variant}: {len(RANGES)} notebooks, {folds} folds, {SUBJECTS_PER_JOB} subjects per job')

    # Short full-model run to check IM-TTA with one whole-target test batch.
    folder = REPO / CHECK['folder']
    folder.mkdir(exist_ok=True)
    spec = VARIANTS['full']
    code = (TEMPLATE
            .replace('__VARIANT__', 'full')
            .replace('__LABEL__', spec['label'])
            .replace('__ESTIMATE__', str(spec['estimate_h']))
            .replace('__RA__', str(spec['ra']))
            .replace('__DA__', str(spec['da']))
            .replace('__MODEL__', repr(spec['model']))
            .replace('__JOB__', '1')
            .replace('__TOTAL__', '1')
            .replace('__TARGETS__', ', '.join(f'S{s}' for s in CHECK['subjects']))
            .replace('__RUN_TAG__', CHECK['run_tag'])
            .replace('__TARGET_LIST__', repr(CHECK['subjects'])))
    (folder / f"{CHECK['run_tag']}.ipynb").write_text(
        json.dumps(notebook(code), indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f"check: {CHECK['run_tag']}, targets {CHECK['subjects']}")


if __name__ == '__main__':
    main()
