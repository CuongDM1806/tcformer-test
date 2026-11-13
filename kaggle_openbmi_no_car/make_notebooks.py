"""Generate the Kaggle notebooks of the OpenBMI no-CAR check (Claim 1, hypothesis H1).

Question: does RA help OpenBMI once the common average reference (CAR) is
removed, i.e. once the common reference component stays in the signals as in
BCI IV-2a? The only preprocessing change is ``common_average_reference: false``
(4-38 Hz, 100 Hz, 0-4 s kept), set in configs/hada_tcformer.yaml of this branch.

Step 1 (no GPU, Internet on):
  prepare_openbmi_nocar_cache.ipynb
      Downloads Lee2019_MI, preprocesses it without CAR and writes 54
      openbmi_SXX.npz files. Save its Output as the Kaggle Dataset
      ``openbmi-nocar-cuong``.

Step 2 (GPU T4 x2, Internet on, attach openbmi-nocar-cuong via Add Input):
  openbmi_nocar_{full,no_ra,per_session_ra}_s001_s018 / _s019_s037 / _s038_s054
      full            fixed RA (reference = session 1) + AutoDIAL + IM-TTA
      no_ra           the full model without RA
      per_session_ra  every session whitened by its own label-free reference
  Every fold is scored with IM-TTA (test_*) and without it (no_tta_*).

Run:  python kaggle_openbmi_no_car/make_notebooks.py
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

BRANCH = 'feature/ssm-dan-openbmi-no-car'
DATASET_NAME = 'openbmi-nocar-cuong'
DATASET_REF = f'nguyncuongv186/{DATASET_NAME}'
JOBS = [(1, 18), (19, 37), (38, 54)]

# Fails fast on Kaggle if the cloned branch is not the no-CAR one.
VERIFY = (
    "import yaml as _yaml\n"
    "_openbmi = _yaml.safe_load((REPO_PATH / 'configs/hada_tcformer.yaml').read_text())['preprocessing']['openbmi']\n"
    "assert _openbmi['common_average_reference'] is False, 'branch still applies CAR to OpenBMI'\n"
    "assert 'per_session' in (REPO_PATH / 'datamodules/base.py').read_text(), "
    "'branch lacks ra_reference: per_session'\n"
    "print('Verified: OpenBMI without CAR, per-session RA available, on', BRANCH, flush=True)\n"
)


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f'expected exactly one occurrence of {old!r}, found {text.count(old)}')
    return text.replace(old, new)


def write(path, code):
    path.write_text(json.dumps(openbmi54.notebook(code), indent=1, ensure_ascii=False) + '\n',
                    encoding='utf-8')
    print(f'wrote {path.relative_to(REPO)}')


# ---------------- Step 1: cache without CAR ----------------
def prepare_notebook():
    source = REPO / 'kaggle_prepare_openbmi_cache.ipynb'
    code = ''.join(json.loads(source.read_text(encoding='utf-8'))['cells'][0]['source'])
    code = replace_once(
        code,
        "# OpenBMI (Lee2019_MI) cache for the OpenBMI LOSO runs. Internet on; no GPU needed.\n"
        "# Downloads each subject (~1.1 GB of MAT files), preprocesses it (4-38 Hz, CAR,\n"
        "# 100 Hz, 0-4 s), writes openbmi_SXX.npz (~20 MB) and deletes the raw files.\n",
        "# OpenBMI (Lee2019_MI) cache WITHOUT CAR for the no-CAR LOSO runs. Internet on; no GPU needed.\n"
        "# Downloads each subject (~1.1 GB of MAT files), preprocesses it (4-38 Hz, NO CAR,\n"
        "# 100 Hz, 0-4 s), writes openbmi_SXX.npz (~20 MB) and deletes the raw files.\n"
        f"# Save the Output as the Kaggle Dataset '{DATASET_NAME}' (Output -> New Dataset).\n")
    code = replace_once(code,
                        "# Afterwards: Output -> New Dataset, then Add Input in the three training\n"
                        "# notebooks.",
                        "# Afterwards: Output -> New Dataset, then Add Input in the nine\n"
                        "# openbmi_nocar_* training notebooks.")
    code = replace_once(code, "BRANCH = 'feature/hada-autodial-openbmi'", f"BRANCH = '{BRANCH}'")
    code = replace_once(code, 'REPO_PATH = Path("/kaggle/working/tcformer-autodial-openbmi-cache")',
                        'REPO_PATH = Path("/kaggle/working/tcformer-openbmi-no-car-cache")')
    code = replace_once(code, 'CACHE_DIR = Path("/kaggle/working/openbmi_cache")',
                        'CACHE_DIR = Path("/kaggle/working/openbmi_cache_nocar")')
    code = replace_once(code, 'BUILD_LOG = Path("/kaggle/working/openbmi_cache_build.log")',
                        'BUILD_LOG = Path("/kaggle/working/openbmi_cache_nocar_build.log")')
    code = replace_once(code, 'print("Verified: OpenBMI cache builder.")\n',
                        'print("Verified: OpenBMI cache builder.")\n' + VERIFY)
    write(HERE / 'prepare_openbmi_nocar_cache.ipynb', code)


# ---------------- Step 2: training ----------------
# Only cache files built without CAR count; the CAR cache (openbmi-dataset-cuong)
# has the same file names, so attaching it by mistake must not start a 61 GB download.
FIND_CACHE_OLD = '''def find_cache():
    cached = [path for root in (Path('/kaggle/input'), DOWNLOADED) if root.is_dir()
              for path in sorted(root.rglob('openbmi_S*.npz'))]
    return cached, sorted(set(range(1, 55)) - {int(path.stem[-2:]) for path in cached})
'''
FIND_CACHE_NEW = '''def find_cache():
    import json as _json
    import numpy as _np
    cached = []
    for root in (Path('/kaggle/input'), DOWNLOADED):
        if not root.is_dir():
            continue
        for path in sorted(root.rglob('openbmi_S*.npz')):
            with _np.load(path, allow_pickle=False) as data:
                car = _json.loads(str(data['signature'])).get('common_average_reference')
            if car is False:
                cached.append(path)
            else:
                print(f'  ignoring {path}: built with CAR', flush=True)
    return cached, sorted(set(range(1, 55)) - {int(path.stem[-2:]) for path in cached})
'''

VARIANTS = {
    'full': dict(title='full model (fixed RA + AutoDIAL + IM-TTA), NO CAR',
                 name='full model without CAR', label='no CAR | fixed RA + AutoDIAL + IM-TTA',
                 preprocessing={'common_average_reference': False}),
    'no_ra': dict(title='no RA (AutoDIAL + IM-TTA), NO CAR',
                  name='no-RA variant without CAR', label='no CAR | no RA: AutoDIAL + IM-TTA',
                  preprocessing={'common_average_reference': False, 'riemannian_alignment': False}),
    'per_session_ra': dict(title='full model with per-session RA, NO CAR',
                           name='per-session RA model without CAR',
                           label='no CAR | per-session RA + AutoDIAL + IM-TTA',
                           preprocessing={'common_average_reference': False,
                                          'ra_reference': 'per_session'}),
}


def training_notebook(variant, first, last):
    spec = VARIANTS[variant]
    block = (f"# No-CAR check: {spec['label']}.\n"
             f"PREPROCESSING = {spec['preprocessing']!r}\n")
    override = ("f\"c['preprocessing'][{DATASET!r}].update({PREPROCESSING!r}); \"\n"
                "            ")
    tag = f's{first:03d}-s{last:03d}'
    code = (openbmi54.TEMPLATE
            .replace('__TITLE__', spec['title'])
            .replace('__NAME__', spec['name'])
            .replace('__LABEL__', spec['label'])
            .replace('__SETTINGS__', block)
            .replace('__OVERRIDE__', override)
            .replace('__JOB__', str(JOBS.index((first, last)) + 1))
            .replace('__TOTAL__', str(len(JOBS)))
            .replace('__TARGETS__', f'S{first}-S{last}')
            .replace('__RUN_TAG__', f'openbmi_nocar_{variant}_{tag}')
            .replace('__TARGET_LIST__', f'list(range({first}, {last} + 1))'))
    code = replace_once(code, "BRANCH = 'feature/hada-autodial-openbmi54'", f"BRANCH = '{BRANCH}'")
    code = replace_once(code, "run(['git', 'log', '-1', '--oneline'], cwd=REPO_PATH)\n",
                        "run(['git', 'log', '-1', '--oneline'], cwd=REPO_PATH)\n" + VERIFY)
    code = replace_once(code, "DATASET_REF = 'nguyncuongv186/openbmi-dataset-cuong'",
                        f"DATASET_REF = '{DATASET_REF}'")
    code = replace_once(code, '# Data: Kaggle Dataset openbmi-dataset-cuong (54 preprocessed openbmi_S*.npz files).',
                        f'# Data: Kaggle Dataset {DATASET_NAME} (54 openbmi_S*.npz files built WITHOUT CAR by\n'
                        '# prepare_openbmi_nocar_cache.ipynb). Do not use openbmi-dataset-cuong (that cache has CAR).')
    code = replace_once(code, "'attach the Dataset openbmi-dataset-cuong via \"Add Input\"",
                        f"'attach the Dataset {DATASET_NAME} (no-CAR cache) via \"Add Input\"")
    code = replace_once(code, FIND_CACHE_OLD, FIND_CACHE_NEW)
    write(HERE / f"openbmi_nocar_{variant}_{tag.replace('-', '_')}.ipynb", code)


if __name__ == '__main__':
    prepare_notebook()
    for variant in VARIANTS:
        for first, last in JOBS:
            training_notebook(variant, first, last)
