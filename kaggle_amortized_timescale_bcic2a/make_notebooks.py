"""Generate the Kaggle notebook for amortized time-scale adaptation on BCI IV-2a.

A small context encoder predicts every subject's step-size offsets of the selective-SSM
blocks from a set of that subject's unlabeled trials, in one forward pass
(amortized_timescale in configs/hada_tcformer.yaml). Training is episodic: each source
batch holds 4 subjects x 12 trials (S&R augmentation within a subject), and each subject's
offsets come from 8-48 other unlabeled trials of the same subject; the target's offsets
come from its unlabeled Stage-1 batch. Everything else is the paper's SSM-DAN run.
Every fold reports
  - four test-time variants of the same weights: none / batchnorm (SSM-DAN IM-TTA) /
    timescale / both (the gradient offsets on top of the predicted ones), and
  - accuracy with only K unlabeled evaluation trials as context (no gradient, no IM-TTA),
    K = 5, 10, 20, 50, 100 (10 random draws each), all trials, and session T as context.
Run:  python kaggle_amortized_timescale_bcic2a/make_notebooks.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

VARIANTS = {
    'amortized': dict(stage1=False, label='context encoder predicts the time scale of each subject'),
}

TEMPLATE = r'''# SSM-DAN amortized time-scale adaptation | BCI IV-2a LOSO, 9 targets | __LABEL__
# Settings: Accelerator = GPU T4 x2, Internet = On. Run with "Save Version -> Save & Run All". About 5-6 h.
# Two folds train at the same time, one per GPU; each fold has its own log file.
# Every fold scores the same weights with TTA = none / batchnorm (SSM-DAN) / timescale / both, and with only
# K unlabeled evaluation trials as context (summary.csv); time scales vs. mu/beta rhythms: timescale_physiology.csv.
# Resume: if folds are left over, add this notebook's previous Output via
# "Add Input -> Your Work -> Notebook Output" and run again; finished folds are skipped.
import codecs, json, os, queue, select, shutil, subprocess, sys, threading, time, zipfile
from pathlib import Path

NOTEBOOK_START = time.monotonic()
BRANCH = 'feature/ssm-dan-amortized-timescale'
REPO_URL = 'https://github.com/CuongDM1806/tcformer-test.git'
RUN_TAG = 'bcic2a_amortized_timescale'
DATASET = 'bcic2a'
TARGET_SUBJECTS = list(range(1, 10))
SEEDS = [0]
GPUS = [0, 1]              # one fold per GPU at a time
MODEL_OVERRIDES = {'amortized_timescale': True, 'target_timescale': True,
                   'target_timescale_stage1': __STAGE1__, 'im_tta_parameters': 'batchnorm'}
PREPROCESSING = {'episodic_subjects_per_batch': 4}   # 48 source trials = 4 subjects x 12
EXTRA_TTA_MODES = ['none', 'timescale', 'both']
CONTEXT_SWEEP = {'sizes': [5, 10, 20, 50, 100], 'draws': 10}
TIME_BUDGET_H = 11.5       # stop starting new folds so the version ends before Kaggle's 12 h limit
FIRST_FOLD_ESTIMATE_H = 1.4  # used until a fold has finished in this run

REPO_PATH = Path('/tmp/repo')  # keep the clone and venv out of the saved output
MNE_DATA = Path('/tmp/mne_data')
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
assert 'class TimescaleContextEncoder' in (REPO_PATH / 'models/hada_tcformer.py').read_text()
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
                    'MNE_DATASETS_BNCI_PATH': str(MNE_DATA)})
MNE_DATA.mkdir(parents=True, exist_ok=True)

# Download BCI IV-2a once, before the two workers start, so they never download at the same time.
download = (
    "from braindecode.datasets import MOABBDataset\n"
    "for name in ('BNCI2014_001', 'BNCI2014001'):\n"
    "    try:\n"
    f"        MOABBDataset(name, subject_ids={TARGET_SUBJECTS!r}); break\n"
    "    except ValueError:\n"
    "        continue\n"
    "print('BCI IV-2a downloaded')\n"
)
run([PYTHON, '-c', download], cwd=REPO_PATH, env=environment)

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
            f"c['subject_ids']=[{subject}]; c['extra_tta_modes']={EXTRA_TTA_MODES!r}; "
            f"c['context_sweep']={CONTEXT_SWEEP!r}; "
            f"c['model_kwargs'].update({MODEL_OVERRIDES!r}); "
            f"c['preprocessing'][{DATASET!r}].update({PREPROCESSING!r}); "
            "c['log_every_n_epochs']=10; c['enable_progress_bar']=False; "
            "c['model_kwargs']['log_every_n_batches']=0; "
            f"Path('configs/{config_name}.yaml').write_text(yaml.safe_dump(c, sort_keys=False))"
        )
        subprocess.run([str(PYTHON), '-c', override], cwd=REPO_PATH, check=True)
        # Result folders carry the GPU id, so the two workers never touch each other's files.
        for old in (REPO_PATH / 'results').glob(f'*_GPU{gpu}_*'):
            shutil.rmtree(old, ignore_errors=True)
        with print_lock:
            print(f'===== {prefix} {RUN_TAG} | seed {seed} | __LABEL__ =====', flush=True)
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
    time.sleep(90)   # stagger the starts so the two folds do not load data at once
for thread in threads:
    thread.join()

# ---- Summary: the four TTA variants of the same weights, and the learned time scales ----
rows = []
for archive in sorted(OUTPUT.glob('seed*_S*_results.zip')):
    with zipfile.ZipFile(archive) as zf:
        for name in zf.namelist():
            if '/subjects/subject_' in name and name.endswith('.json'):
                rows.append(json.loads(zf.read(name)))
MODES = ['none', 'batchnorm', 'timescale', 'both']
CONTEXTS = [f"test_k{k}" for k in CONTEXT_SWEEP['sizes']] + ['test_all', 'session_T']


def scores(record):
    out = {'batchnorm': (record['test_acc'], record['test_kappa'], record.get('timescale'))}
    for mode, result in record.get('tta_variants', {}).items():
        out[mode] = (result['acc'], result['kappa'], result.get('timescale'))
    return out


with open(OUTPUT / 'summary.csv', 'w') as f:
    f.write('run,subject_id,' + ','.join(f'{m}_acc,{m}_kappa,{m}_tau_ratio' for m in MODES)
            + ',' + ','.join(f'ctx_{c}_acc' for c in CONTEXTS) + ',best_val_acc,train_time_min\n')
    for record in sorted(rows, key=lambda r: r['subject_id']):
        s = scores(record)
        cells = []
        for mode in MODES:
            acc, kappa, ts = s.get(mode, (None, None, None))
            tau = ts['tau_ratio'] if ts else None
            cells += ['' if acc is None else f'{acc:.4f}', '' if kappa is None else f'{kappa:.4f}',
                      '' if tau is None else f'{tau:.4f}']
        context = record.get('amortized_context', {})
        cells += [f"{context[c]['acc']:.4f}" if c in context else '' for c in CONTEXTS]
        f.write(f"{RUN_TAG},{record['subject_id']}," + ','.join(cells)
                + f",{record['best_val_acc']:.4f},{record['train_time_min']:.1f}\n")
print(open(OUTPUT / 'summary.csv').read())
if rows:
    print(f'Mean over {len(rows)} targets (accuracy %; SSM-DAN = batchnorm):')
    for mode in MODES:
        accs = [scores(r)[mode][0] for r in rows if mode in scores(r)]
        if accs:
            print(f'  TTA={mode:9s} {100 * sum(accs) / len(accs):6.2f}  (n={len(accs)})')
    print('Only K unlabeled evaluation trials as context (no gradient, no IM-TTA):')
    for c in CONTEXTS:
        accs = [r['amortized_context'][c]['acc'] for r in rows if c in r.get('amortized_context', {})]
        if accs:
            print(f'  {c:10s} {100 * sum(accs) / len(accs):6.2f}  (n={len(accs)})')

# Learned time scales vs. each subject's mu/beta rhythms (exploratory; no labels used).
if rows:
    result = subprocess.run([str(PYTHON), 'analysis/timescale_physiology.py', '--results', str(OUTPUT),
                             '--out', str(OUTPUT / 'timescale_physiology.csv')],
                            cwd=REPO_PATH, env=environment)
    if result.returncode != 0:
        print('Time-scale/physiology analysis failed; the accuracy results above are unaffected.')

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
    for old in HERE.glob('*.ipynb'):
        old.unlink()
    for variant, spec in VARIANTS.items():
        code = (TEMPLATE
                .replace('__VARIANT__', variant)
                .replace('__LABEL__', spec['label'])
                .replace('__STAGE1__', str(spec['stage1'])))
        path = HERE / 'bcic2a_amortized_timescale.ipynb'
        path.write_text(json.dumps(notebook(code), indent=1, ensure_ascii=False) + '\n',
                        encoding='utf-8')
        print(f'wrote {path.name}')


if __name__ == '__main__':
    main()
