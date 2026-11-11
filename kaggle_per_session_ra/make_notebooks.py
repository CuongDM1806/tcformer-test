"""Generate the Kaggle notebooks of the per-session RA check (full SSM-DAN).

Per-session RA whitens every recording session with its own label-free
reference (``ra_reference: per_session``), instead of fitting the reference on
a subject's earlier sessions and applying it unchanged to the later ones.
Everything else is the configuration of the existing full-model runs, so each
notebook differs from its baseline only in the RA reference:

  ssm_dan_per_session_ra_bcic2a_s01_s09.ipynb
      BCI IV-2a, all 9 LOSO targets, one GPU. Built from
      kaggle_train_autodial_bcic2a_loso.ipynb (the 76.12% run in the paper).
  ssm_dan_per_session_ra_bcic2b_s01_s09.ipynb
      BCI IV-2b, all 9 LOSO targets, one GPU. Built from
      kaggle_train_autodial_bcic2b_loso.ipynb.
  ssm_dan_per_session_ra_zhou2016_s01_s04.ipynb
      Zhou2016, all 4 LOSO targets, one GPU. Built from
      kaggle_train_autodial_zhou2016_loso.ipynb.
  ssm_dan_per_session_ra_openbmi_s001_s012.ipynb, _s013_s026, _s027_s040, _s041_s054
      OpenBMI, all 54 targets in four jobs, two GPUs each. Built from the
      OpenBMI-54 full-model template; baseline = kaggle_openbmi54_full.

Run:  python kaggle_per_session_ra/make_notebooks.py
"""
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
# Same file name as this script, so load the OpenBMI-54 generator by path.
_spec = importlib.util.spec_from_file_location(
    'openbmi54_make_notebooks', REPO / 'kaggle_openbmi54_full' / 'make_notebooks.py')
openbmi54 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(openbmi54)

BRANCH = 'feature/ssm-dan-per-session-ra'
OLD_OPENBMI_BRANCH = "BRANCH = 'feature/hada-autodial-openbmi54'"
OLD_2A_BRANCH = "BRANCH = 'feature/hada-autodial-zhou-mi-preprocessing'"
# Fails fast on Kaggle if the cloned branch predates the ra_reference option.
VERIFY = ("assert 'per_session' in (REPO_PATH / 'datamodules/base.py').read_text(), "
          "'branch lacks ra_reference: per_session'\n"
          "print('Verified: per-session RA available on', BRANCH, flush=True)\n")


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f'expected exactly one occurrence of {old!r}, found {text.count(old)}')
    return text.replace(old, new)


def write(path, code):
    path.write_text(json.dumps(openbmi54.notebook(code), indent=1, ensure_ascii=False) + '\n',
                    encoding='utf-8')
    print(f'wrote {path.relative_to(REPO)}')


def openbmi_notebook(first=1, last=12):
    block = ("# Per-session RA: every session is whitened by its own label-free reference\n"
             "# (target session 2 included). Everything else is the full model.\n"
             "PREPROCESSING = {'ra_reference': 'per_session'}\n")
    override = ("f\"c['preprocessing'][{DATASET!r}].update({PREPROCESSING!r}); \"\n"
                "            ")
    code = (openbmi54.TEMPLATE
            .replace('__TITLE__', 'full model with per-session RA')
            .replace('__NAME__', 'full model (per-session RA)')
            .replace('__LABEL__', 'per-session RA + AutoDIAL + IM-TTA')
            .replace('__SETTINGS__', block)
            .replace('__OVERRIDE__', override)
            .replace('__JOB__', '1')
            .replace('__TOTAL__', '1')
            .replace('__TARGETS__', f'S{first}-S{last}')
            .replace('__RUN_TAG__', f'openbmi_full_per_session_ra_s{first:03d}-s{last:03d}')
            .replace('__TARGET_LIST__', f'list(range({first}, {last} + 1))'))
    code = replace_once(code, OLD_OPENBMI_BRANCH, f"BRANCH = '{BRANCH}'")
    code = replace_once(code, "run(['git', 'log', '-1', '--oneline'], cwd=REPO_PATH)\n",
                        "run(['git', 'log', '-1', '--oneline'], cwd=REPO_PATH)\n" + VERIFY)
    write(HERE / f'ssm_dan_per_session_ra_openbmi_s{first:03d}_s{last:03d}.ipynb', code)


SUMMARY = '''
# ---------------- summary.csv (same columns as the OpenBMI jobs) ----------------
import json
rows = [json.loads(path.read_text()) for path in sorted(results_dir.glob("*/subjects/subject_*.json"))]


def fmt(value):
    return "" if value is None else f"{value:.4f}"


summary = Path(f"/kaggle/working/{RUN_TAG}_summary.csv")
with open(summary, "w") as f:
    f.write("run,seed,subject_id,test_acc,test_kappa,no_tta_acc,no_tta_kappa,best_val_acc,train_time_min\\n")
    for r in sorted(rows, key=lambda r: r["subject_id"]):
        f.write(f"{RUN_TAG},0,{r['subject_id']},{r['test_acc']:.4f},{r['test_kappa']:.4f},"
                f"{fmt(r.get('no_tta_acc'))},{fmt(r.get('no_tta_kappa'))},"
                f"{r['best_val_acc']:.4f},{r['train_time_min']:.1f}\\n")
print(summary.read_text())
accs = [r["test_acc"] for r in rows]
print(f"Mean accuracy over {len(accs)} targets: {100 * sum(accs) / max(len(accs), 1):.2f}% "
      "(per-session RA; paper baseline with calibration-session RA: __BASELINE__)")
'''

# BCI IV-2b only: copy B0*T/E.mat files from an attached Kaggle Dataset into the
# MOABB cache, so the run does not depend on the BNCI server being online.
BNCI_FALLBACK = '''MNE_DATA.mkdir(parents=True, exist_ok=True)
BNCI_CACHE = MNE_DATA / "MNE-bnci-data" / "database" / "data-sets" / "004-2014"
BNCI_CACHE.mkdir(parents=True, exist_ok=True)
for mat in sorted(Path("/kaggle/input").rglob("B0[1-9][TE].mat")):
    if not (BNCI_CACHE / mat.name).exists():
        shutil.copy2(mat, BNCI_CACHE / mat.name)
cached = sorted(f.name for f in BNCI_CACHE.glob("B0*.mat"))
print(f"BCI IV-2b files in local cache: {len(cached)}/18", flush=True)
'''

SINGLE_GPU = {
    'bcic2a': dict(source='kaggle_train_autodial_bcic2a_loso.ipynb', name='BCI IV-2a',
                   out='bcic2a_s01_s09', baseline='76.12%',
                   ra_note='sessions T and E of every subject are whitened by their own label-free RA reference.'),
    'bcic2b': dict(source='kaggle_train_autodial_bcic2b_loso.ipynb', name='BCI IV-2b',
                   out='bcic2b_s01_s09', baseline='see the BCI IV-2b row of the paper',
                   ra_note='each of the five sessions of every subject is whitened by its own label-free RA reference.'),
    'zhou2016': dict(source='kaggle_train_autodial_zhou2016_loso.ipynb', name='Zhou2016',
                     out='zhou2016_s01_s04', baseline='see the Zhou2016 row of the paper',
                     ra_note='each of the three sessions of every subject is whitened by its own label-free RA reference.'),
}


def single_gpu_notebook(dataset):
    spec = SINGLE_GPU[dataset]
    module = {'bcic2a': 'bcic4_2a', 'bcic2b': 'bcic4_2b', 'zhou2016': 'zhou2016'}[dataset]
    module_check = (f"assert '_ra_per_session' in (REPO_PATH / 'datamodules/{module}.py').read_text(), "
                    f"'datamodules/{module}.py lacks per-session RA'\n")
    code = ''.join(json.loads((REPO / spec['source']).read_text(encoding='utf-8'))['cells'][0]['source'])
    first_line = code.split('\n', 1)[0]
    code = replace_once(code, first_line,
                        f"# SSM-DAN (full) with per-session RA — {spec['name']} LOSO (one cell). Enable GPU + Internet.\n"
                        f"# Identical to {spec['source']} except that\n"
                        f"# {spec['ra_note']}")
    code = replace_once(code, OLD_2A_BRANCH, f"BRANCH = '{BRANCH}'")
    code = replace_once(code, 'RUN_TAG = f"{DATASET}_{FEATURE_ALIGNMENT}_{JOB_TAG}"',
                        'RUN_TAG = f"{DATASET}_{FEATURE_ALIGNMENT}_per_session_ra_{JOB_TAG}"')
    code = replace_once(code, 'print("Verified: AutoDIAL model, Zhou2016 support, best source-validation checkpoint.")\n',
                        'print("Verified: AutoDIAL model, Zhou2016 support, best source-validation checkpoint.")\n'
                        + VERIFY + module_check)
    code = replace_once(code, '''    f"c['model_kwargs']['feature_alignment']={FEATURE_ALIGNMENT!r}; "\n''',
                        '''    f"c['model_kwargs']['feature_alignment']={FEATURE_ALIGNMENT!r}; "\n'''
                        '''    f"c['preprocessing'][{DATASET!r}]['ra_reference']='per_session'; "\n'''
                        '''    "c['evaluate_without_im_tta']=True; "\n''')
    code = replace_once(code, '| alignment={FEATURE_ALIGNMENT} "',
                        '| alignment={FEATURE_ALIGNMENT} | RA=per-session "')
    code = replace_once(code, 'primary test.", flush=True)',
                        'primary test. RA: every session uses its own label-free reference.", flush=True)')
    if dataset == 'bcic2b':
        code = replace_once(code, 'MNE_DATA.mkdir(parents=True, exist_ok=True)\n', BNCI_FALLBACK)
    code = replace_once(code, 'archive = shutil.make_archive(',
                        SUMMARY.lstrip('\n').replace('__BASELINE__', spec['baseline'])
                        + '\narchive = shutil.make_archive(')
    write(HERE / f"ssm_dan_per_session_ra_{spec['out']}.ipynb", code)


OPENBMI_JOBS = [(1, 12), (13, 26), (27, 40), (41, 54)]


if __name__ == '__main__':
    for dataset in SINGLE_GPU:
        single_gpu_notebook(dataset)
    for first, last in OPENBMI_JOBS:
        openbmi_notebook(first, last)
