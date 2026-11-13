"""Build ra_data_characteristics.ipynb (Kaggle, CPU only) from ra_data_characteristics.py.

The notebook is self-contained: it writes the analysis module from an embedded copy,
downloads BCI IV-2a, BCI IV-2b and Zhou2016 through MOABB, and reads the OpenBMI caches
attached as Kaggle Datasets.

Run:  python analysis/make_ra_data_characteristics_notebook.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODULE = (HERE / "ra_data_characteristics.py").read_text(encoding="utf-8")

INTRO = """# Claim 1 – mục 4: đo đặc trưng dữ liệu (không cần GPU)

Đo trên **tín hiệu mà RA nhận** (tiền xử lý giống pipeline, trước RA), theo từng subject, cho BCI IV-2a, BCI IV-2b, Zhou2016 và OpenBMI. Không train mô hình nào.

| Giả thuyết | Đo gì |
|---|---|
| **H1** tương quan giữa kênh / tham chiếu chung | \\|r\\| trung bình giữa các kênh; tỉ lệ năng lượng ở thành phần chung (phần CAR sẽ trừ đi) |
| **H2** năng lượng ngoài băng mu/beta | tỉ lệ năng lượng < 4 Hz, 4–8, 8–30, 30–45 Hz; độ biến thiên giữa subject. Zhou đo 2 lần: pipeline (8–30 Hz) và **không lọc** |
| **H3** chất lượng tham chiếu RA | độ lệch giữa tham chiếu tính từ hai nửa trial; số điều kiện; trial/kênh; khoảng cách tham chiếu session fit và session đánh giá |
| **H4** kiểu khác biệt giữa subject | trong tangent space, tách phương sai thành: giữa subject (RA khử được), giữa lớp, tương tác subject × lớp (RA không khử được), phần dư |

**Thiết lập Kaggle:** Accelerator = **None (CPU)**, Internet = **On**.

**Add Input:**
- `openbmi-dataset-cuong`: OpenBMI có CAR (bắt buộc nếu muốn có OpenBMI).
- `openbmi-nocar-cuong`: OpenBMI không CAR (tùy chọn, khi đã tạo xong).

Không gắn dataset nào thì notebook vẫn chạy cho 2a, 2b, Zhou.

Kết quả lưu ở `/kaggle/working/ra_data_characteristics/` và file zip cùng tên.
"""

SETUP = """import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'moabb>=1.1'], check=True)
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import mne, moabb
print('mne', mne.__version__, '| moabb', moabb.__version__)
"""

LOADERS = r'''import json, shutil, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import mne
from moabb import datasets as moabb_datasets
from moabb.utils import set_download_dir

sys.path.insert(0, '/tmp')
import ra_data_characteristics as rdc

mne.set_log_level('ERROR')
DATA_DIR = Path('/tmp/mne_data'); DATA_DIR.mkdir(parents=True, exist_ok=True)
set_download_dir(str(DATA_DIR))
OUT = Path('/kaggle/working/ra_data_characteristics'); OUT.mkdir(parents=True, exist_ok=True)

# Same windows and filters as the training pipeline (configs/hada_tcformer.yaml):
# BCI IV-2a full MOABB interval, no filter; BCI IV-2b interval minus its last 0.5 s, no filter;
# Zhou2016 8-30 Hz and 0.5-5.0 s after the cue. fit = leading sessions used for the frozen RA reference.
SPECS = {
    'BCI IV-2a': dict(classes=('BNCI2014_001', 'BNCI2014001'), start=0.0, duration=4.0, fit=1, sessions=2),
    'BCI IV-2b': dict(classes=('BNCI2014_004', 'BNCI2014004'), start=0.0, duration=4.0, fit=3, sessions=5),
    'Zhou2016': dict(classes=('Zhou2016',), start=0.5, duration=4.5, fit=2, sessions=3),
}


def moabb_dataset(names):
    for name in names:
        if hasattr(moabb_datasets, name):
            return getattr(moabb_datasets, name)()
    raise ImportError(f'none of {names} in this MOABB version')


def epochs_from_raw(raw, dataset, start, duration, band):
    event_id = dataset.event_id
    stim = [ch for ch, kind in zip(raw.ch_names, raw.get_channel_types()) if kind == 'stim']
    if stim:
        events = mne.find_events(raw, stim_channel=stim[0], shortest_event=1, verbose=False)
    else:
        events, _ = mne.events_from_annotations(raw, event_id=event_id, verbose=False)
    codes = sorted(event_id.values())
    events = events[np.isin(events[:, 2], codes)]
    eeg = raw.copy().pick('eeg').load_data(verbose=False)
    eeg.apply_function(lambda values: values * 1e6, channel_wise=False)   # V -> uV, as the pipeline
    if band is not None:
        eeg.filter(band[0], band[1], verbose=False)
    sfreq = eeg.info['sfreq']
    tmin = float(dataset.interval[0]) + start
    present = {name: code for name, code in event_id.items() if code in set(events[:, 2])}
    epochs = mne.Epochs(eeg, events, event_id=present, tmin=tmin, tmax=tmin + duration - 1.0 / sfreq,
                        baseline=None, preload=True, verbose=False)
    y = np.searchsorted(codes, epochs.events[:, 2]).astype(np.int64)
    return epochs.get_data().astype(np.float32), y, sfreq


def load_moabb_views(name, spec, bands):
    """One View per band setting; raw files are read once per subject."""
    dataset = moabb_dataset(spec['classes'])
    subjects = {label: [] for label in bands}
    sfreq = None
    for subject in dataset.subject_list:
        started = time.time()
        data = dataset.get_data(subjects=[subject])[subject]
        session_names = sorted(data, key=rdc.natural_key)
        if len(session_names) != spec['sessions']:
            raise RuntimeError(f'{name} S{subject}: {len(session_names)} sessions, expected {spec["sessions"]}')
        for label, band in bands.items():
            sessions = []
            for session in session_names:
                parts = [epochs_from_raw(data[session][run], dataset, spec['start'], spec['duration'], band)
                         for run in sorted(data[session], key=rdc.natural_key)]
                sfreq = parts[0][2]
                sessions.append((np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])))
            subjects[label].append(rdc.SubjectData(subject, sessions, spec['fit']))
        shapes = [s[0].shape for s in subjects[next(iter(bands))][-1].sessions]
        print(f'{name} S{subject:02d}: sessions {shapes} ({time.time() - started:.0f} s)', flush=True)
    return [rdc.View(label, sfreq, desc, subjects[label]) for label, desc in
            ((label, f'{spec["start"]}-{spec["start"] + spec["duration"]} s after interval start, band {band}')
             for label, band in bands.items())]


def load_openbmi_views():
    """OpenBMI caches attached under /kaggle/input, grouped by their preprocessing signature."""
    groups = {}
    for path in sorted(Path('/kaggle/input').rglob('openbmi_S*.npz')):
        with np.load(path, allow_pickle=False) as data:
            signature = json.loads(str(data['signature']))
        car = 'CAR' if signature.get('common_average_reference', True) else 'no CAR'
        groups.setdefault((car, signature['sfreq'], signature['low_cut'], signature['high_cut']), {}) \
              .setdefault(int(path.stem[-2:]), path)
    views = []
    for (car, sfreq, low, high), files in sorted(groups.items()):
        if len(files) != 54:
            print(f'OpenBMI {car}: only {len(files)}/54 subjects attached, used as is', flush=True)
        subjects = []
        for subject in sorted(files):
            with np.load(files[subject], allow_pickle=False) as data:
                subjects.append(rdc.SubjectData(subject, [(data['X1'], data['y1']), (data['X2'], data['y2'])], 1))
        views.append(rdc.View(f'OpenBMI ({car}, {low:g}-{high:g} Hz)', float(sfreq),
                              f'{car}, {low}-{high} Hz, {sfreq} Hz, 0-4 s', subjects))
        print(f'OpenBMI {car}: {len(subjects)} subjects', flush=True)
    if not views:
        print('No OpenBMI cache attached: add openbmi-dataset-cuong (and openbmi-nocar-cuong) via Add Input.')
    return views
'''

RUN = r'''views = []
views += load_moabb_views('BCI IV-2a', SPECS['BCI IV-2a'], {'BCI IV-2a': None})
views += load_moabb_views('BCI IV-2b', SPECS['BCI IV-2b'], {'BCI IV-2b': None})
views += load_moabb_views('Zhou2016', SPECS['Zhou2016'],
                          {'Zhou2016 (pipeline, 8-30 Hz)': (8.0, 30.0), 'Zhou2016 (unfiltered)': None})
views += load_openbmi_views()

metrics, decomposition, psd = [], [], []
for view in views:
    started = time.time()
    print(f'== {view.name}: {len(view.subjects)} subjects, {view.sfreq:g} Hz', flush=True)
    m, d, p = rdc.analyse_view(view)
    metrics.append(m); decomposition.append(d); psd.append(p)
    print(f'   done in {(time.time() - started) / 60:.1f} min', flush=True)
metrics = pd.concat(metrics, ignore_index=True)
decomposition = pd.concat(decomposition, ignore_index=True)
psd = pd.concat(psd, ignore_index=True)
metrics.to_csv(OUT / 'subject_metrics.csv', index=False)
decomposition.to_csv(OUT / 'variance_decomposition.csv', index=False)
psd.to_csv(OUT / 'relative_psd.csv', index=False)
'''

SUMMARY = r'''# RA gain of the network (tab:ablation, fixed RA / per-session RA) for reference.
RA_GAIN = {'BCI IV-2a': '+5.83 / +7.45', 'BCI IV-2b': '+0.51 / +0.52',
           'Zhou2016 (pipeline, 8-30 Hz)': '+0.33 / +1.33', 'Zhou2016 (unfiltered)': 'not run',
           'OpenBMI (CAR, 4-38 Hz)': '-2.89 / +0.50', 'OpenBMI (no CAR, 4-38 Hz)': 'running'}
summary = rdc.summarise(metrics)
summary.loc['network RA gain (fixed / per-session)'] = [RA_GAIN.get(v, '') for v in summary.columns]
summary.to_csv(OUT / 'summary.csv')
pd.set_option('display.max_colwidth', 40, 'display.width', 250)
display(summary)
display(decomposition.round(3))
'''

PLOTS = r'''import matplotlib.pyplot as plt
order = list(dict.fromkeys(metrics['view']))

# H2: median relative power spectrum per view.
fig, ax = plt.subplots(figsize=(8, 4.5))
for view in order:
    curve = psd[psd['view'] == view].groupby('freq')['relative_psd'].median()
    ax.semilogy(curve.index, curve.values, label=view)
ax.axvspan(8, 30, color='0.9', zorder=0)
ax.set(xlabel='Frequency (Hz)', ylabel='Relative power (median over subjects)', title='H2: power spectrum seen by RA')
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT / 'h2_relative_psd.png', dpi=150); plt.show()

# H1-H4: per-subject distributions.
panels = [('h1_mean_abs_corr', 'H1: mean |r| between channels'),
          ('h1_common_mode_excess', 'H1: common-mode share (0 = none, 1 = all)'),
          ('h2_share_outside_8-30', 'H2: power share outside 8-30 Hz'),
          ('h3_split_half_distance', 'H3: split-half reference distance / sqrt(C)'),
          ('h3_session_shift', 'H3: fit vs evaluation reference / sqrt(C)'),
          ('h4_offset_over_interaction', 'H4: subject offset / subject x class')]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, (metric, title) in zip(axes.flat, panels):
    ax.boxplot([metrics.loc[metrics['view'] == v, metric] for v in order])
    ax.set_xticks(range(1, len(order) + 1), order, rotation=35, ha='right', fontsize=8)
    ax.set_title(title, fontsize=10)
fig.tight_layout(); fig.savefig(OUT / 'h1_h4_boxplots.png', dpi=150); plt.show()

# H4: variance decomposition before and after RA.
parts = ['eta2_subject', 'eta2_class', 'eta2_subject_x_class', 'eta2_residual']
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
for ax, kind in zip(axes, ['before RA', 'after frozen RA']):
    table = decomposition[decomposition['covariance'] == kind].set_index('view').loc[order, parts]
    table.plot.bar(stacked=True, ax=ax, legend=ax is axes[0])
    ax.set_title(f'H4: tangent-space variance, {kind}'); ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=35)
fig.tight_layout(); fig.savefig(OUT / 'h4_variance_decomposition.png', dpi=150); plt.show()

shutil.make_archive(str(OUT), 'zip', root_dir=OUT)
print('Saved', sorted(p.name for p in OUT.iterdir()), 'and', f'{OUT}.zip')
'''

READING = """## Cách đọc kết quả

| Nếu thấy | Thì |
|---|---|
| \\|r\\| và common-mode của **2a ≈ Zhou** (cùng đơn cực) nhưng Δ_RA rất khác | H1 **một mình không đủ** giải thích 2a và Zhou khác nhau |
| **2a** có tỉ lệ năng lượng ngoài 8–30 Hz cao và **biến thiên nhiều giữa subject**; **Zhou (unfiltered)** cũng cao nhưng **Zhou (pipeline)** thấp | Khác biệt 2a và Zhou đến từ **lựa chọn lọc tần số** (H2), không phải từ bản thân dữ liệu. Thí nghiệm tiếp theo: Zhou không lọc, có RA và không RA |
| OpenBMI có split-half distance cao nhất, trial/kênh thấp nhất | Tham chiếu RA của OpenBMI kém ổn định (H3) |
| 2a có `subject_over_interaction` (H4) cao nhất | Khác biệt giữa subject ở 2a chủ yếu là dịch chuyển chung, đúng loại RA sửa được |
| Sau RA, `eta2_subject` về ≈ 0 ở mọi bộ | Kiểm tra hợp lệ: RA khử đúng phần giữa subject |

Gửi lại `summary.csv`, `variance_decomposition.csv` và 3 hình (hoặc file zip) để phân tích tiếp.
"""


def cell(kind, source):
    base = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        base.update({"execution_count": None, "outputs": []})
    return base


def main():
    cells = [
        cell("markdown", INTRO),
        cell("code", SETUP),
        cell("code", "%%writefile /tmp/ra_data_characteristics.py\n" + MODULE),
        cell("code", LOADERS),
        cell("code", RUN),
        cell("code", SUMMARY),
        cell("code", PLOTS),
        cell("markdown", READING),
    ]
    notebook = {"cells": cells,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python"}},
                "nbformat": 4, "nbformat_minor": 5}
    path = HERE / "ra_data_characteristics.ipynb"
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
