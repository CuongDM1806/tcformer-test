"""Build gw_alignment_bcic2a.ipynb (Kaggle, CPU only).

Proxy test (MDM, TS-LR; no deep network) of fused Gromov-Wasserstein channel alignment on
BCI IV-2a, LOSO: train on the other 8 subjects' session T, test on the target's session E.

Configurations (all label-free on the target):
  none        no alignment
  chnorm      per-channel scaling only (diagonal of the reference covariance)
  ra          Riemannian alignment (reference)
  gw_chnorm   chnorm, then fused-GW channel mapping onto a source template  -> replaces RA
  gw_ra       fused-GW channel mapping, then RA                               -> complements RA

Run:  python analysis/make_gw_alignment_notebook.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_NOTEBOOK = HERE / "gw_alignment_bcic2a.ipynb"

INTRO = """# Fused Gromov–Wasserstein channel alignment – proxy trên BCI IV-2a (không cần GPU)

**Câu hỏi:** ghép kênh bằng fused GW có **thay được RA** (`gw_chnorm`) hoặc **bổ trợ RA** (`gw_ra`) không?

**Phép ghép:** với mỗi subject, cấu trúc chức năng là `1 − coherence` giữa các cặp kênh (8–30 Hz), tính trên trial không nhãn. Fused GW ghép kênh của subject vào một **template** (trung bình cấu trúc của các subject nguồn trong fold), có thêm chi phí hình học = bình phương khoảng cách 3D giữa điện cực (montage chuẩn 10-05). Như vậy tín hiệu chỉ được dời sang kênh lân cận. Kênh mới j = Σ_i n·T_ij·x_i.

| Cấu hình | Làm gì |
|---|---|
| `none` | không căn chỉnh |
| `chnorm` | chỉ chia mỗi kênh cho độ lệch chuẩn của nó (đối chứng: GW có thêm gì so với chuẩn hoá kênh) |
| `ra` | Riemannian alignment (mốc so sánh) |
| `gw_chnorm` | chnorm rồi GW — **thay hẳn RA** |
| `gw_ra` | GW rồi RA — **bổ trợ RA** |

Mỗi cấu hình chạy với 2 cách lấy thống kê căn chỉnh: `frozen` (fit trên session T, áp cho E, như bài) và `per_session` (mỗi session tự fit). Bộ phân loại proxy: MDM và TS-LR trên tangent space log-Euclid. Tiền xử lý giống pipeline 2a: 0–4 s sau đầu interval MOABB, không lọc, đơn vị µV.

**Đọc kết quả:** `gw_chnorm ≥ ra` → thay được RA. `gw_ra > ra` → bổ trợ. Cả hai không hơn → dừng hướng này. `ALPHAS` ở ô chạy để thử độ nhạy; **không chọn alpha theo điểm test**, alpha chính là 0.5.

**Kaggle:** Accelerator = **None (CPU)**, Internet = **On**. Kết quả ở `/kaggle/working/gw_alignment/` và `gw_alignment.zip`.
"""

SETUP = """import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'moabb>=1.1', 'pot>=0.9', 'pyriemann>=0.6'], check=True)
import mne, moabb, ot, pyriemann
print('mne', mne.__version__, '| moabb', moabb.__version__, '| POT', ot.__version__, '| pyriemann', pyriemann.__version__)
"""

LOAD = r'''import re, shutil, time
from pathlib import Path
import numpy as np
import pandas as pd
import mne
from moabb import datasets as moabb_datasets
from moabb.utils import set_download_dir
from sklearn.covariance import OAS

mne.set_log_level('ERROR')
DATA_DIR = Path('/tmp/mne_data'); DATA_DIR.mkdir(parents=True, exist_ok=True)
set_download_dir(str(DATA_DIR))
OUT = Path('/kaggle/working/gw_alignment'); OUT.mkdir(parents=True, exist_ok=True)

START, DURATION = 0.0, 4.0        # seconds after the MOABB interval start, as the 2a pipeline
COH_BAND = (8.0, 30.0)            # band of the functional structure used by GW


def natural_key(text):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', str(text))]


def load_dataset():
    for name in ('BNCI2014_001', 'BNCI2014001'):
        if hasattr(moabb_datasets, name):
            return getattr(moabb_datasets, name)()
    raise ImportError('BNCI2014_001 not found in this MOABB version')


def epochs_from_raw(raw, dataset):
    event_id = dataset.event_id
    stim = [ch for ch, kind in zip(raw.ch_names, raw.get_channel_types()) if kind == 'stim']
    if stim:
        events = mne.find_events(raw, stim_channel=stim[0], shortest_event=1, verbose=False)
    else:
        events, _ = mne.events_from_annotations(raw, event_id=event_id, verbose=False)
    codes = sorted(event_id.values())
    events = events[np.isin(events[:, 2], codes)]
    eeg = raw.copy().pick('eeg').load_data(verbose=False)
    eeg.apply_function(lambda v: v * 1e6, channel_wise=False)          # V -> uV
    sfreq = eeg.info['sfreq']
    tmin = float(dataset.interval[0]) + START
    present = {k: c for k, c in event_id.items() if c in set(events[:, 2])}
    ep = mne.Epochs(eeg, events, event_id=present, tmin=tmin, tmax=tmin + DURATION - 1.0 / sfreq,
                    baseline=None, preload=True, verbose=False)
    y = np.searchsorted(codes, ep.events[:, 2]).astype(np.int64)
    return ep.get_data().astype(np.float64), y, sfreq, eeg.ch_names


def coherence_distance(x, sfreq, band=COH_BAND):
    """1 - mean magnitude-squared coherence over `band`, cross-spectra averaged over trials."""
    win = np.hanning(x.shape[-1])
    spec = np.fft.rfft(x * win, axis=-1)                               # [trials, C, F]
    freqs = np.fft.rfftfreq(x.shape[-1], 1.0 / sfreq)
    spec = spec[..., (freqs >= band[0]) & (freqs <= band[1])]
    cross = np.einsum('ncf,ndf->fcd', spec, spec.conj()) / len(x)       # [F, C, C]
    power = np.real(np.einsum('fcc->fc', cross))
    coh = np.abs(cross) ** 2 / (power[:, :, None] * power[:, None, :])
    d = 1.0 - coh.mean(axis=0)
    np.fill_diagonal(d, 0.0)
    return d


dataset = load_dataset()
SUBJECTS = {}
for subject in dataset.subject_list:
    t0 = time.time()
    data = dataset.get_data(subjects=[subject])[subject]
    sessions = []
    for session in sorted(data, key=natural_key):
        parts = [epochs_from_raw(data[session][run], dataset) for run in sorted(data[session], key=natural_key)]
        x = np.concatenate([p[0] for p in parts]); y = np.concatenate([p[1] for p in parts])
        sfreq, ch_names = parts[0][2], parts[0][3]
        covs = np.stack([OAS().fit(t.T).covariance_ for t in x])
        sessions.append(dict(covs=covs, y=y, coh_dist=coherence_distance(x, sfreq)))
    if len(sessions) != 2:
        raise RuntimeError(f'S{subject}: {len(sessions)} sessions, expected 2 (T, E)')
    SUBJECTS[subject] = sessions
    print(f'S{subject:02d}: ' + ', '.join(f"{s['covs'].shape[0]} trials" for s in sessions)
          + f' | {len(ch_names)} ch | {time.time() - t0:.0f} s', flush=True)
CH_NAMES = ch_names

montage = mne.channels.make_standard_montage('standard_1005')
pos = montage.get_positions()['ch_pos']
missing = [c for c in CH_NAMES if c not in pos]
if missing:
    raise RuntimeError(f'channels without standard position: {missing}')
XYZ = np.array([pos[c] for c in CH_NAMES]) * 1000.0                    # mm
DIST_MM = np.linalg.norm(XYZ[:, None] - XYZ[None], axis=-1)
'''

METHOD = r'''import ot
from scipy.linalg import eigh
from pyriemann.utils.mean import mean_riemann

EPS = 1e-6


def spd_power(c, p):
    w, v = eigh(c)
    return (v * np.maximum(w, 1e-12) ** p) @ v.T


def congruence(a, covs):
    out = np.einsum('ij,njk,lk->nil', a, covs, a, optimize=True)
    reg = EPS * np.trace(out, axis1=1, axis2=2).mean() / out.shape[1]
    return out + reg * np.eye(out.shape[1])


def tangent(covs):
    """log-Euclidean tangent vectors at identity (upper triangle, off-diagonal x sqrt 2)."""
    w, v = np.linalg.eigh(covs)
    logs = np.einsum('nij,nj,nkj->nik', v, np.log(np.maximum(w, 1e-12)), v)
    iu = np.triu_indices(covs.shape[1])
    scale = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    return logs[:, iu[0], iu[1]] * scale


def fgw_map(struct_subject, struct_template, alpha):
    """Fused GW coupling subject -> template; returns mapping A (new = A @ x) and diagnostics."""
    n = len(struct_subject)
    p = q = np.full(n, 1.0 / n)
    geom = (DIST_MM / DIST_MM.max()) ** 2
    coupling = ot.gromov.fused_gromov_wasserstein(geom, struct_subject, struct_template, p, q,
                                                  loss_fun='square_loss', alpha=alpha)
    a = n * coupling.T                                                   # rows: template channel j
    diag = float(np.trace(coupling))
    disp = float((coupling * DIST_MM).sum())                             # mean displacement, mm
    return a, diag, disp


def reference_sessions(sessions, mode):
    """Which trials estimate the alignment statistics of each session."""
    if mode == 'frozen':
        return [0] * len(sessions)                                       # session T for both
    if mode == 'per_session':
        return list(range(len(sessions)))
    raise ValueError(mode)


def align_subject(sessions, config, mode, template, alpha):
    """Aligned covariances per session + GW diagnostics, for one subject."""
    refs = reference_sessions(sessions, mode)
    out, diags = [], []
    for k, s in enumerate(sessions):
        r = sessions[refs[k]]
        covs = s['covs']
        ref_covs = r['covs']
        if config in ('chnorm', 'gw_chnorm'):
            d = np.diag(ref_covs.mean(axis=0)) ** -0.5
            covs = covs * d[None, :, None] * d[None, None, :]
            ref_covs = ref_covs * d[None, :, None] * d[None, None, :]
        if config in ('gw_chnorm', 'gw_ra'):
            a, diag, disp = fgw_map(r['coh_dist'], template, alpha)
            covs, ref_covs = congruence(a, covs), congruence(a, ref_covs)
            diags.append(dict(session=k, diag_mass=diag, displacement_mm=disp))
        if config in ('ra', 'gw_ra'):
            w = spd_power(mean_riemann(ref_covs), -0.5)
            covs = congruence(w, covs)
        out.append(covs)
    return out, diags


def mdm(x_tr, y_tr, x_te):
    classes = np.unique(y_tr)
    means = np.stack([x_tr[y_tr == c].mean(axis=0) for c in classes])
    return classes[((x_te[:, None] - means[None]) ** 2).sum(-1).argmin(1)]


def tslr(x_tr, y_tr, x_te):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(x_tr, y_tr).predict(x_te)


CLASSIFIERS = {'MDM': mdm, 'TS-LR': tslr}


def run(config, mode, alpha=0.5):
    rows, diag_rows = [], []
    for target in SUBJECTS:
        sources = [s for s in SUBJECTS if s != target]
        # Template from the sources only (session T); the target never contributes to it.
        template = np.mean([SUBJECTS[s][0]['coh_dist'] for s in sources], axis=0)
        x_tr, y_tr = [], []
        for s in sources:
            covs, _ = align_subject(SUBJECTS[s], config, mode, template, alpha)
            x_tr.append(tangent(covs[0])); y_tr.append(SUBJECTS[s][0]['y'])
        covs, diags = align_subject(SUBJECTS[target], config, mode, template, alpha)
        x_tr, y_tr = np.concatenate(x_tr), np.concatenate(y_tr)
        x_te, y_te = tangent(covs[1]), SUBJECTS[target][1]['y']
        for name, clf in CLASSIFIERS.items():
            acc = 100.0 * float(np.mean(clf(x_tr, y_tr, x_te) == y_te))
            rows.append(dict(config=config, mode=mode, alpha=alpha, classifier=name, subject=target, acc=acc))
        for d in diags:
            diag_rows.append(dict(config=config, mode=mode, alpha=alpha, subject=target, **d))
    return rows, diag_rows
'''

RUN = r'''CONFIGS = ['none', 'chnorm', 'ra', 'gw_chnorm', 'gw_ra']
MODES = ['frozen', 'per_session']
ALPHA_MAIN = 0.5
ALPHAS = [0.5]            # sensitivity, e.g. [0.2, 0.5, 0.8]; report all, do not pick by test score

rows, diag_rows = [], []
for mode in MODES:
    for config in CONFIGS:
        alphas = ALPHAS if config.startswith('gw') else [ALPHA_MAIN]
        for alpha in alphas:
            t0 = time.time()
            r, d = run(config, mode, alpha)
            rows += r; diag_rows += d
            accs = pd.DataFrame(r).groupby('classifier')['acc'].mean()
            print(f'{mode:11s} {config:10s} alpha={alpha:.2f} | '
                  + ' | '.join(f'{k} {v:.2f}' for k, v in accs.items()) + f' | {time.time() - t0:.0f} s', flush=True)

results = pd.DataFrame(rows)
diagnostics = pd.DataFrame(diag_rows)
results.to_csv(OUT / 'per_subject.csv', index=False)
diagnostics.to_csv(OUT / 'gw_coupling_diagnostics.csv', index=False)
'''

REPORT = r'''from scipy.stats import wilcoxon

def compare(a, b):
    diff = a - b
    if np.allclose(diff, 0):
        return np.nan
    return wilcoxon(a, b, zero_method='zsplit').pvalue

summary = []
main = results[(results.alpha == ALPHA_MAIN) | ~results.config.str.startswith('gw')]
for (mode, clf), g in main.groupby(['mode', 'classifier']):
    table = g.pivot(index='subject', columns='config', values='acc')
    for config in CONFIGS:
        row = dict(mode=mode, classifier=clf, config=config,
                   mean=table[config].mean(), std=table[config].std(ddof=0))
        for ref in ('none', 'ra'):
            if config != ref:
                d = table[config] - table[ref]
                row[f'vs_{ref}'] = d.mean()
                row[f'wins_{ref}'] = f'{(d > 0).sum()}/{(d == 0).sum()}/{(d < 0).sum()}'
                row[f'p_{ref}'] = compare(table[config].values, table[ref].values)
        summary.append(row)
summary = pd.DataFrame(summary)
summary.to_csv(OUT / 'summary.csv', index=False)
pd.set_option('display.width', 200)
print('Mean accuracy (%) over 9 targets, alpha =', ALPHA_MAIN, '| wins = W/T/L, p = Wilcoxon')
print(summary.round(3).to_string(index=False))

for (mode, clf), g in main.groupby(['mode', 'classifier']):
    print(f'\n== per subject | {mode} | {clf}')
    print(g.pivot(index='subject', columns='config', values='acc')[CONFIGS].round(2).to_string())

if len(ALPHAS) > 1:
    print('\nAlpha sensitivity (mean %):')
    print(results[results.config.str.startswith('gw')]
          .groupby(['mode', 'config', 'classifier', 'alpha'])['acc'].mean().unstack('alpha').round(2).to_string())

if len(diagnostics):
    print('\nGW coupling: diag_mass = share kept on the same channel (1 = identity); displacement in mm')
    print(diagnostics[diagnostics.alpha == ALPHA_MAIN]
          .groupby(['mode', 'config', 'subject'])[['diag_mass', 'displacement_mm']].mean().round(3).to_string())

shutil.make_archive('/kaggle/working/gw_alignment', 'zip', OUT)
print('\nSaved', sorted(p.name for p in OUT.iterdir()), '-> /kaggle/working/gw_alignment.zip')
'''


def cell(kind, text):
    lines = text.strip("\n").splitlines(keepends=True)
    base = {"cell_type": kind, "metadata": {}, "source": lines}
    if kind == "code":
        base.update(execution_count=None, outputs=[])
    return base


nb = {
    "cells": [cell("markdown", INTRO), cell("code", SETUP), cell("code", LOAD),
              cell("code", METHOD), cell("code", RUN), cell("code", REPORT)],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4, "nbformat_minor": 5,
}
OUT_NOTEBOOK.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote", OUT_NOTEBOOK)
