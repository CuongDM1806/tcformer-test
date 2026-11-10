"""Generate the Kaggle notebooks of the per-session RA check (full SSM-DAN).

Per-session RA whitens every recording session with its own label-free
reference (``ra_reference: per_session``), instead of fitting the reference on
a subject's earlier sessions and applying it unchanged to the later ones.
Everything else is the configuration of the existing full-model runs, so each
notebook differs from its baseline only in the RA reference:

  ssm_dan_per_session_ra_bcic2a_s01_s09.ipynb
      BCI IV-2a, all 9 LOSO targets, one GPU. Built from
      kaggle_train_autodial_bcic2a_loso.ipynb (the 76.12% run in the paper).
  ssm_dan_per_session_ra_openbmi_s001_s012.ipynb
      OpenBMI targets S1-S12, two GPUs. Built from the OpenBMI-54 full-model
      template; baseline = kaggle_openbmi54_full job01.

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


SUMMARY_2A = '''
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
      "(per-session RA; paper baseline with session-T RA: 76.12%)")
'''


def bcic2a_notebook():
    source = REPO / 'kaggle_train_autodial_bcic2a_loso.ipynb'
    code = ''.join(json.loads(source.read_text(encoding='utf-8'))['cells'][0]['source'])
    code = replace_once(code, '# AutoDIAL Full-Mamba HADA — BCI IV-2a LOSO (one cell). Enable GPU + Internet.',
                        '# SSM-DAN (full) with per-session RA — BCI IV-2a LOSO (one cell). Enable GPU + Internet.\n'
                        '# Identical to kaggle_train_autodial_bcic2a_loso.ipynb (76.12% in the paper) except that\n'
                        '# sessions T and E of every subject are whitened by their own label-free RA reference.')
    code = replace_once(code, OLD_2A_BRANCH, f"BRANCH = '{BRANCH}'")
    code = replace_once(code, 'RUN_TAG = f"{DATASET}_{FEATURE_ALIGNMENT}_{JOB_TAG}"',
                        'RUN_TAG = f"{DATASET}_{FEATURE_ALIGNMENT}_per_session_ra_{JOB_TAG}"')
    code = replace_once(code, 'print("Verified: AutoDIAL model, Zhou2016 support, best source-validation checkpoint.")\n',
                        'print("Verified: AutoDIAL model, Zhou2016 support, best source-validation checkpoint.")\n'
                        + VERIFY)
    code = replace_once(code, '''    f"c['model_kwargs']['feature_alignment']={FEATURE_ALIGNMENT!r}; "\n''',
                        '''    f"c['model_kwargs']['feature_alignment']={FEATURE_ALIGNMENT!r}; "\n'''
                        '''    f"c['preprocessing'][{DATASET!r}]['ra_reference']='per_session'; "\n'''
                        '''    "c['evaluate_without_im_tta']=True; "\n''')
    code = replace_once(code, '| alignment={FEATURE_ALIGNMENT} "',
                        '| alignment={FEATURE_ALIGNMENT} | RA=per-session "')
    code = replace_once(code, 'Target: session T is unlabeled UDA data; session E is IM-TTA + AutoDIAL recalibration + primary test.',
                        'Target: session T is unlabeled UDA data; session E is IM-TTA + AutoDIAL recalibration + primary test. '
                        'RA: every session uses its own label-free reference.')
    code = replace_once(code, 'archive = shutil.make_archive(', SUMMARY_2A.lstrip('\n') + '\narchive = shutil.make_archive(')
    write(HERE / 'ssm_dan_per_session_ra_bcic2a_s01_s09.ipynb', code)


if __name__ == '__main__':
    bcic2a_notebook()
    openbmi_notebook()
