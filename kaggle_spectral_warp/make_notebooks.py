"""Generate the Kaggle notebooks of spectral warp alignment (GRL replacement).

The target subject gets one learned frequency-scale factor s in [0.8, 1.25] on
the first temporal convolutions (models/tcformer.py, warp_temporal_kernel).
In mode "adversarial" the domain discriminator still trains on source vs
target features, but its counter-signal moves only s: no reversed gradient
reaches the network. Everything else is the paper's SSM-DAN configuration
(frozen RA, AutoDIAL, MMD, temporal MMD, IM, IM-TTA). Each fold also logs the
learned s and the measured mu peak of the source pool and of the unlabeled
target trials, to test whether s tracks the individual mu frequency.

  bcic2a_spectral_warp_adversarial.ipynb       BCI IV-2a, 9 LOSO targets, T4 x2
  openbmi_spectral_warp_adversarial_s001_s012.ipynb   OpenBMI S1-S12, T4 x2

Run:  python kaggle_spectral_warp/make_notebooks.py
"""
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
_spec = importlib.util.spec_from_file_location(
    'openbmi54_make_notebooks', REPO / 'kaggle_openbmi54_full' / 'make_notebooks.py')
openbmi54 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(openbmi54)

BRANCH = 'feature/ssm-dan-spectral-warp'

VARIANTS = {
    # The GRL is replaced: the discriminator moves only the target warp factor.
    'adversarial': dict(label='spectral warp replaces GRL (s in [0.8, 1.25])',
                        overrides={'spectral_warp_max': 0.25,
                                   'spectral_warp_mode': 'adversarial'},
                        preprocessing={}),
}

TEMPLATE = r'''# SSM-DAN + spectral warp alignment | BCI IV-2a LOSO, 9 targets | variant __VARIANT__: __LABEL__
# Settings: Accelerator = GPU T4 x2, Internet = On. Run with "Save Version -> Save & Run All". About 7 h.
# Two folds train at the same time, one per GPU; each fold has its own log file.
# Everything is the paper's SSM-DAN run (frozen session-T RA, AutoDIAL, MMD, IM, IM-TTA) except the GRL:
# the target subject gets one learned frequency-scale factor s on the first temporal convolutions, and the
# domain discriminator moves only s (no reversed gradient into the network). Each fold is scored with
# IM-TTA (test_acc) and without it (no_tta_acc), and logs s and the measured mu peaks of source/target.
# References: paper 76.12% / 72.15% (no IM-TTA).
# Resume: if folds are left over, add this notebook's previous Output via
# "Add Input -> Your Work -> Notebook Output" and run again; finished folds are skipped.
import codecs, json, os, queue, select, shutil, subprocess, sys, threading, time, zipfile
from pathlib import Path

NOTEBOOK_START = time.monotonic()
BRANCH = 'feature/ssm-dan-spectral-warp'
REPO_URL = 'https://github.com/CuongDM1806/tcformer-test.git'
RUN_TAG = 'bcic2a_spectral_warp___VARIANT__'
DATASET = 'bcic2a'
TARGET_SUBJECTS = list(range(1, 10))
SEEDS = [0]
GPUS = [0, 1]              # one fold per GPU at a time
MODEL_OVERRIDES = __OVERRIDES__
PREPROCESSING = __PREPROCESSING__
TIME_BUDGET_H = 11.5       # stop starting new folds so the version ends before Kaggle's 12 h limit
FIRST_FOLD_ESTIMATE_H = 1.5  # used until a fold has finished in this run

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
assert 'def warp_temporal_kernel' in (REPO_PATH / 'models/tcformer.py').read_text(), 'branch lacks spectral warp'
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
            f"c['subject_ids']=[{subject}]; c['evaluate_without_im_tta']=True; c['measure_mu_peak']=True; "
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

# ---- Summary: with IM-TTA (test_*) and the same weights without it (no_tta_*) ----
rows = []
for archive in sorted(OUTPUT.glob('seed*_S*_results.zip')):
    with zipfile.ZipFile(archive) as zf:
        for name in zf.namelist():
            if '/subjects/subject_' in name and name.endswith('.json'):
                rows.append(json.loads(zf.read(name)))


def fmt(value):
    return '' if value is None else f'{value:.4f}'


with open(OUTPUT / 'summary.csv', 'w') as f:
    f.write('run,seed,subject_id,test_acc,test_kappa,no_tta_acc,no_tta_kappa,best_val_acc,train_time_min,'
            'warp_s,source_mu_hz,target_mu_hz,mu_ratio\n')
    for r in sorted(rows, key=lambda r: r['subject_id']):
        f.write(f"{RUN_TAG},0,{r['subject_id']},{r['test_acc']:.4f},{r['test_kappa']:.4f},"
                f"{fmt(r.get('no_tta_acc'))},{fmt(r.get('no_tta_kappa'))},"
                f"{r['best_val_acc']:.4f},{r['train_time_min']:.1f},"
                f"{fmt(r.get('spectral_warp_scale'))},{fmt(r.get('source_mu_peak_hz'))},"
                f"{fmt(r.get('target_mu_peak_hz'))},{fmt(r.get('mu_peak_ratio'))}\n")
print(open(OUTPUT / 'summary.csv').read())
if rows:
    acc = [r['test_acc'] for r in rows]
    no_tta = [r['no_tta_acc'] for r in rows if r.get('no_tta_acc') is not None]
    print(f'Mean over {len(rows)} targets: IM-TTA {100 * sum(acc) / len(acc):.2f}%'
          + (f' | no IM-TTA {100 * sum(no_tta) / len(no_tta):.2f}%' if no_tta else '')
          + ' (paper SSM-DAN 76.12% / 72.15%)')

left = sorted(deferred + failures)
if left:
    # No exception here, so this version still ends normally and its Output is saved.
    print(f'NOT FINISHED: {left} (failed: {sorted(failures)}).')
    print("Re-run: Add Input -> Your Work -> Notebook Output -> this version, then Save & Run All.")
else:
    print(f'All folds of {RUN_TAG} finished.')
'''


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f'expected exactly one occurrence of {old!r}, found {text.count(old)}')
    return text.replace(old, new)


def write(path, code):
    path.write_text(json.dumps(openbmi54.notebook(code), indent=1, ensure_ascii=False) + '\n',
                    encoding='utf-8')
    print(f'wrote {path.relative_to(REPO)}')


def bcic2a_notebook(variant, spec):
    code = (TEMPLATE
            .replace('__VARIANT__', variant)
            .replace('__LABEL__', spec['label'])
            .replace('__OVERRIDES__', repr(spec['overrides']))
            .replace('__PREPROCESSING__', repr(spec['preprocessing'])))
    write(HERE / f'bcic2a_spectral_warp_{variant}.ipynb', code)


OLD_OPENBMI_BRANCH = "BRANCH = 'feature/hada-autodial-openbmi54'"
OPENBMI_SUMMARY_HEADER = (
    "f.write('run,seed,subject_id,test_acc,test_kappa,no_tta_acc,no_tta_kappa,best_val_acc,"
    "train_time_min\\n')")
OPENBMI_SUMMARY_ROW_END = '''f"{r['best_val_acc']:.4f},{r['train_time_min']:.1f}\\n")'''


def openbmi_notebook(variant, spec, first=1, last=12):
    block = (f"# Spectral warp alignment, variant {variant}: {spec['label']}.\n"
             f"MODEL_OVERRIDES = {spec['overrides']!r}\n")
    override = ("f\"c['model_kwargs'].update({MODEL_OVERRIDES!r}); c['measure_mu_peak']=True; \"\n"
                "            ")
    tag = f's{first:03d}-s{last:03d}'
    code = (openbmi54.TEMPLATE
            .replace('__TITLE__', f'spectral warp alignment ({variant})')
            .replace('__NAME__', f'spectral-warp model ({variant})')
            .replace('__LABEL__', spec['label'])
            .replace('__SETTINGS__', block)
            .replace('__OVERRIDE__', override)
            .replace('__JOB__', '1')
            .replace('__TOTAL__', '1')
            .replace('__TARGETS__', f'S{first}-S{last}')
            .replace('__RUN_TAG__', f'openbmi_spectral_warp_{variant}_{tag}')
            .replace('__TARGET_LIST__', f'list(range({first}, {last} + 1))'))
    code = replace_once(code, OLD_OPENBMI_BRANCH, f"BRANCH = '{BRANCH}'")
    code = replace_once(code, 'FIRST_FOLD_ESTIMATE_H = 1.5', 'FIRST_FOLD_ESTIMATE_H = 2.0')
    code = replace_once(
        code, "run(['git', 'log', '-1', '--oneline'], cwd=REPO_PATH)\n",
        "run(['git', 'log', '-1', '--oneline'], cwd=REPO_PATH)\n"
        "assert 'def warp_temporal_kernel' in (REPO_PATH / 'models/tcformer.py').read_text(), "
        "'branch lacks spectral warp'\n")
    code = replace_once(
        code, OPENBMI_SUMMARY_HEADER,
        "f.write('run,seed,subject_id,test_acc,test_kappa,no_tta_acc,no_tta_kappa,best_val_acc,"
        "train_time_min,warp_s,source_mu_hz,target_mu_hz,mu_ratio\\n')")
    code = replace_once(
        code, OPENBMI_SUMMARY_ROW_END,
        '''f"{r['best_val_acc']:.4f},{r['train_time_min']:.1f},"\n'''
        '''                f"{fmt(r.get('spectral_warp_scale'))},{fmt(r.get('source_mu_peak_hz'))},"\n'''
        '''                f"{fmt(r.get('target_mu_peak_hz'))},{fmt(r.get('mu_peak_ratio'))}\\n")''')
    write(HERE / f"openbmi_spectral_warp_{variant}_{tag.replace('-', '_')}.ipynb", code)


def main():
    for old in HERE.glob('*.ipynb'):
        old.unlink()
    for variant, spec in VARIANTS.items():
        bcic2a_notebook(variant, spec)
        openbmi_notebook(variant, spec)


if __name__ == '__main__':
    main()
