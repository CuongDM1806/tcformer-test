"""Build ra_rotation.ipynb (Claim 1, item 12; Kaggle, CPU only).

Self-contained: embeds ra_data_characteristics.py, ra_interventions.py, ra_rotation.py and
ra_rotation_proxy.py, and reuses the MOABB loaders of the item-4 notebook.

Run:  python analysis/make_ra_rotation_notebook.py
"""
import json
from pathlib import Path

import make_ra_data_characteristics_notebook as item4

HERE = Path(__file__).resolve().parent
MODULES = ["ra_data_characteristics.py", "ra_interventions.py", "ra_rotation.py", "ra_rotation_proxy.py"]

INTRO = """# Claim 1, mục 12: sau RA có còn một phép quay giữa các subject không? (không cần GPU)

**Giả thuyết** (`docs/CLAIM1_RA_BCI2A.md`, mục 6a): RA làm trắng từng subject nên khử được phép trộn kênh A_s,
nhưng chỉ đến một **phép quay** U_s. Phép quay có C(C−1)/2 bậc tự do (3 với 3 kênh, 231 với 22 kênh),
nên phần lệch còn lại sau RA lớn dần theo số kênh.

**Căn chỉnh quay** (Procrustes, bước cuối của RPA): sau RA, xoay từng subject sao cho các tâm lớp
(trong tangent space log-Euclid) khớp với template của các subject nguồn. Subject nguồn dùng nhãn; subject đích
**không dùng nhãn**: tâm lớp lấy theo nhãn giả (tâm template gần nhất), fit phép quay trên session huấn luyện của đích
rồi áp cho session đánh giá.

Với mỗi bộ dữ liệu / tập kênh, MDM và TS-LR được chấm với 5 cấu hình: không RA, RA cố định, RA cố định + quay,
RA theo session, RA theo session + quay (LOSO như trong bài).

| Tập kênh | Mục đích |
|---|---|
| 2a: C3, Cz, C4 | 3 bậc quay |
| 2a: 8 và 14 kênh ngẫu nhiên (3 lần rút mỗi mức) | số bậc quay tăng dần |
| 2a: đủ 22 kênh | 231 bậc quay |
| 2b (3 lưỡng cực), Zhou (14 kênh) | đối chiếu |

**Dự đoán nếu giả thuyết đúng:**
1. `rotation_share` (phần lệch tâm lớp giữa subject sau RA là phép quay) đáng kể, và
2. `extra_rot` (= +quay − không quay) tăng theo số kênh của 2a, lớn nhất ở 22 kênh.

OpenBMI (62 kênh, 54 fold) không chạy ở đây vì quá chậm với phép quay 62×62.

**Thiết lập Kaggle:** Accelerator = **None (CPU)**, Internet = **On**. Không cần Add Input.
Kết quả lưu ở `/kaggle/working/ra_rotation/` và file zip cùng tên.
"""

LOADERS = item4.LOADERS.replace(
    "OUT = Path('/kaggle/working/ra_data_characteristics')",
    "OUT = Path('/kaggle/working/ra_rotation')",
) + "\nimport ra_interventions as rai\nimport ra_rotation_proxy as rrp\n"

RUN = r'''log = lambda msg: print(msg, flush=True)
results, diagnostics = [], []

def run(dataset, view, covs):
    started = time.time()
    results.append(rrp.rotation_table(dataset, view, covs, log=log))
    diagnostics.append(rrp.rotation_diagnostic(dataset, view, covs))
    log(f"    rotation_share {diagnostics[-1]['rotation_share']:.3f} ({(time.time() - started) / 60:.1f} min)")
    pd.concat(results, ignore_index=True).to_csv(OUT / 'subject_results.csv', index=False)
    pd.DataFrame(diagnostics).to_csv(OUT / 'rotation_diagnostic.csv', index=False)

# ---- BCI IV-2a: channel names from MOABB (EEG channels in the order the loader keeps)
dataset_2a = moabb_dataset(SPECS['BCI IV-2a']['classes'])
first = dataset_2a.get_data(subjects=[1])[1]
session = sorted(first, key=rdc.natural_key)[0]
raw = first[session][sorted(first[session], key=rdc.natural_key)[0]]
names = raw.copy().pick('eeg').ch_names
motor = [names.index(c) for c in ('C3', 'Cz', 'C4')]
log(f'2a channels ({len(names)}): {names}; C3/Cz/C4 at {motor}')

a_raw, = load_moabb_views('BCI IV-2a', SPECS['BCI IV-2a'], {'raw': None})
covs_2a = rai.to_covariances(a_raw, log=log)
del a_raw
run('BCI IV-2a', 'C3 Cz C4', rai.subset_channels(covs_2a, np.array(motor)))
rng = np.random.default_rng(0)
for n in (8, 14):
    for draw in range(3):
        channels = np.sort(rng.choice(len(names), n, replace=False))
        run('BCI IV-2a', f'{n} random', rai.subset_channels(covs_2a, channels))
run('BCI IV-2a', 'all 22', covs_2a)

# ---- BCI IV-2b and Zhou2016 (pipeline preprocessing)
b_raw, = load_moabb_views('BCI IV-2b', SPECS['BCI IV-2b'], {'raw': None})
run('BCI IV-2b', 'all 3', rai.to_covariances(b_raw, log=log))
del b_raw
z_pipe, = load_moabb_views('Zhou2016', SPECS['Zhou2016'], {'8-30': (8.0, 30.0)})
run('Zhou2016', 'all 14', rai.to_covariances(z_pipe, log=log))
del z_pipe

results = pd.concat(results, ignore_index=True)
diagnostics = pd.DataFrame(diagnostics)
'''

SUMMARY = r'''summary = rrp.summarise(results)
summary.to_csv(OUT / 'summary.csv', index=False)
diag = diagnostics.groupby(['dataset', 'view', 'n_channels'], sort=False, as_index=False).mean(numeric_only=True)
diag.to_csv(OUT / 'rotation_diagnostic_summary.csv', index=False)
pd.set_option('display.width', 250, 'display.max_columns', 40)
display(diag.round(3))
cols = ['dataset', 'view', 'n_channels', 'n', 'acc_none', 'acc_frozen', 'acc_frozen+rot', 'extra_rot_frozen',
        'p_extra_rot_frozen', 'acc_per_session', 'acc_per_session+rot', 'extra_rot_per_session', 'p_extra_rot_per_session']
for clf in rai.CLASSIFIERS:
    print(f'===== {clf}')
    display(summary[summary.classifier == clf][cols].round(3))
'''

PLOTS = r'''import matplotlib.pyplot as plt
two_a = summary[summary.dataset == 'BCI IV-2a'].sort_values('n_channels')
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for clf in rai.CLASSIFIERS:
    s = two_a[two_a.classifier == clf]
    axes[0].plot(s.n_channels, s.extra_rot_frozen, 'o-', label=f'{clf}, fixed RA')
    axes[0].plot(s.n_channels, s.extra_rot_per_session, 's--', label=f'{clf}, per-session RA')
    axes[1].plot(s.n_channels, s.gain_frozen, 'o-', label=f'{clf}: RA')
    axes[1].plot(s.n_channels, s['gain_frozen+rot'], 's--', label=f'{clf}: RA + rotation')
axes[0].axhline(0, color='0.4', lw=0.8)
axes[0].set(xlabel='channels (BCI IV-2a)', ylabel='accuracy points', title='Extra gain of rotation after RA')
axes[1].set(xlabel='channels (BCI IV-2a)', ylabel='accuracy points', title='Gain over no alignment (fixed RA)')
d = diag[diag.dataset == 'BCI IV-2a'].sort_values('n_channels')
axes[2].plot(d.n_channels, d.rotation_share, 'o-', label='BCI IV-2a')
for _, r in diag[diag.dataset != 'BCI IV-2a'].iterrows():
    axes[2].plot(r.n_channels, r.rotation_share, 'x', ms=10, label=r.dataset)
axes[2].set(xlabel='channels', ylabel='share', ylim=(0, 1), title='Class-mean mismatch after RA that is a rotation')
for ax in axes:
    ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(OUT / 'rotation_vs_channels.png', dpi=150); plt.show()

shutil.make_archive(str(OUT), 'zip', root_dir=OUT)
print('Saved', sorted(p.name for p in OUT.iterdir()), 'and', f'{OUT}.zip')
'''

READING = """## Cách đọc kết quả

| Nếu thấy | Thì |
|---|---|
| `extra_rot` của 2a tăng theo số kênh, dương và có ý nghĩa ở 22 kênh | Ủng hộ giả thuyết phép quay; làm tiếp mục 13 (mạng) |
| `extra_rot` ≈ 0 ở mọi số kênh | Phép quay còn lại không ảnh hưởng bộ phân loại; giả thuyết yếu đi |
| `rotation_share` lớn nhưng `extra_rot` ≈ 0 | Có phép quay, nhưng không ước lượng được nó cho đích khi không có nhãn (nhãn giả kém) |
| `extra_rot` âm | Nhãn giả của đích sai, phép quay xoay nhầm lớp |

Gửi lại file zip (`summary.csv`, `rotation_diagnostic_summary.csv`, `subject_results.csv`, hình).
"""


def main():
    cells = [item4.cell("markdown", INTRO), item4.cell("code", item4.SETUP)]
    for module in MODULES:
        cells.append(item4.cell("code", f"%%writefile /tmp/{module}\n" + (HERE / module).read_text(encoding="utf-8")))
    cells += [item4.cell("code", LOADERS), item4.cell("code", RUN), item4.cell("code", SUMMARY),
              item4.cell("code", PLOTS), item4.cell("markdown", READING)]
    notebook = {"cells": cells,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python"}},
                "nbformat": 4, "nbformat_minor": 5}
    path = HERE / "ra_rotation.ipynb"
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
