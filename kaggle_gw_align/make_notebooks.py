"""Generate the Kaggle notebooks of the fused-GW channel alignment runs (full SSM-DAN, BCI IV-2a).

Built from kaggle_train_autodial_bcic2a_loso.ipynb (the 76.12% / 72.15% no-TTA run in the
paper) and differing only in the input alignment (utils/gw_alignment.py):

  ssm_dan_gw_chnorm_bcic2a_s01_s09.ipynb   per-channel scaling + GW, no RA  (replaces RA)
  ssm_dan_gw_ra_bcic2a_s01_s09.ipynb       GW, then RA                      (complements RA)

Both use the calibration-session reference, like the paper (ra_reference unchanged).

Run:  python kaggle_gw_align/make_notebooks.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SOURCE = 'kaggle_train_autodial_bcic2a_loso.ipynb'
BRANCH = 'feature/ssm-dan-gw-align'
OLD_BRANCH = "BRANCH = 'feature/hada-autodial-zhou-mi-preprocessing'"

VARIANTS = {
    'gw_chnorm': dict(ra=False, label='per-channel scaling + fused-GW channel mapping, RA switched off'),
    'gw_ra': dict(ra=True, label='fused-GW channel mapping, then the usual RA'),
}

VERIFY = ("assert 'align_loso' in (REPO_PATH / 'datamodules/bcic4_2a.py').read_text(), "
          "'branch lacks GW channel alignment'\n"
          "print('Verified: GW channel alignment available on', BRANCH, flush=True)\n")

SUMMARY = '''
# ---------------- summary.csv + GW coupling diagnostics from the log ----------------
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
no_tta = [r["no_tta_acc"] for r in rows if r.get("no_tta_acc") is not None]
print(f"Mean over {len(accs)} targets: IM-TTA {100 * sum(accs) / max(len(accs), 1):.2f}%"
      + (f" | no IM-TTA {100 * sum(no_tta) / len(no_tta):.2f}%" if no_tta else ""))
gw_lines = [line.strip() for line in TRAIN_LOG.read_text(errors="replace").splitlines()
            if "GW " in line and "diag_mass" in line]
print(f"GW coupling lines in the log: {len(gw_lines)} (diag_mass 1.000 = identity mapping)")
for line in gw_lines[-9:]:
    print(" ", line)
'''


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f'expected exactly one occurrence of {old!r}, found {text.count(old)}')
    return text.replace(old, new)


def notebook(variant):
    spec = VARIANTS[variant]
    source = json.loads((REPO / SOURCE).read_text(encoding='utf-8'))
    code = ''.join(source['cells'][0]['source'])
    first_line = code.split('\n', 1)[0]
    code = replace_once(code, first_line,
                        f"# SSM-DAN (full) with fused-GW channel alignment [{variant}] — BCI IV-2a LOSO (one cell).\n"
                        f"# Enable GPU + Internet. Identical to {SOURCE} except the input alignment:\n"
                        f"# {spec['label']}.")
    code = replace_once(code, "JOB_TAG = 's01_s09'\n",
                        "JOB_TAG = 's01_s09'\n"
                        f"CHANNEL_ALIGNMENT = {variant!r}\n"
                        "GW_ALPHA = 0.5   # weight of the functional-structure term (1 = no electrode-geometry term)\n")
    code = replace_once(code, OLD_BRANCH, f"BRANCH = '{BRANCH}'")
    code = replace_once(code, 'RUN_TAG = f"{DATASET}_{FEATURE_ALIGNMENT}_{JOB_TAG}"',
                        'RUN_TAG = f"{DATASET}_{FEATURE_ALIGNMENT}_{CHANNEL_ALIGNMENT}_{JOB_TAG}"')
    code = replace_once(code, 'print("Verified: AutoDIAL model, Zhou2016 support, best source-validation checkpoint.")\n',
                        'print("Verified: AutoDIAL model, Zhou2016 support, best source-validation checkpoint.")\n'
                        + VERIFY)
    code = replace_once(code, '''    f"c['model_kwargs']['feature_alignment']={FEATURE_ALIGNMENT!r}; "\n''',
                        '''    f"c['model_kwargs']['feature_alignment']={FEATURE_ALIGNMENT!r}; "\n'''
                        '''    f"c['preprocessing'][{DATASET!r}]['channel_alignment']={CHANNEL_ALIGNMENT!r}; "\n'''
                        '''    f"c['preprocessing'][{DATASET!r}]['gw_alpha']={GW_ALPHA!r}; "\n'''
                        '''    f"c['preprocessing'][{DATASET!r}]['riemannian_alignment']=''' + str(spec['ra']) + '''; "\n'''
                        '''    "c['evaluate_without_im_tta']=True; "\n''')
    code = replace_once(code, '| alignment={FEATURE_ALIGNMENT} "',
                        '| alignment={FEATURE_ALIGNMENT} | input={CHANNEL_ALIGNMENT} (alpha={GW_ALPHA}) "')
    code = replace_once(code, 'archive = shutil.make_archive(',
                        SUMMARY.lstrip('\n') + '\narchive = shutil.make_archive(')
    source['cells'][0]['source'] = code.splitlines(keepends=True)
    for cell in source['cells']:
        if cell.get('cell_type') == 'code':
            cell['outputs'], cell['execution_count'] = [], None
    path = HERE / f'ssm_dan_{variant}_bcic2a_s01_s09.ipynb'
    path.write_text(json.dumps(source, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'wrote {path.relative_to(REPO)}')


if __name__ == '__main__':
    for variant in VARIANTS:
        notebook(variant)
