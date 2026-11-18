"""Build ra_interventions.ipynb (Claim 1, items 5 + 6; Kaggle, CPU only).

Self-contained: embeds ra_data_characteristics.py and ra_interventions.py, reuses the data
loaders of the item-4 notebook (MOABB for BCI IV-2a, IV-2b, Zhou2016; attached caches for OpenBMI).

Run:  python analysis/make_ra_interventions_notebook.py
"""
import json
from pathlib import Path

import make_ra_data_characteristics_notebook as item4

HERE = Path(__file__).resolve().parent
RDC = (HERE / "ra_data_characteristics.py").read_text(encoding="utf-8")
RAI = (HERE / "ra_interventions.py").read_text(encoding="utf-8")

INTRO = """# Claim 1, mục 5 + 6: can thiệp lên dữ liệu, đo lại lợi ích của RA bằng bộ phân loại proxy (không cần GPU)

Mỗi lần can thiệp bật hoặc tắt **một** đặc trưng trên **cùng một** bộ dữ liệu, rồi đo lại
**Δ_RA = Acc(có RA) − Acc(không RA)** bằng hai bộ phân loại trên ma trận hiệp phương sai:

- **MDM:** tâm lớp gần nhất trong tangent space log-Euclid, không có tham số học.
- **TS-LR:** vector tangent space, chuẩn hóa, rồi logistic regression.

Giao thức như trong bài: LOSO; subject nguồn dùng các session huấn luyện có nhãn; subject đích được chấm trên các session sau.
Mỗi cấu hình chạy với 3 kiểu RA: **không RA**, **RA cố định** (tham chiếu từ các session huấn luyện), **RA theo session**.
Không train mạng nào.

| Giả thuyết | Can thiệp | Dự đoán nếu giả thuyết đúng |
|---|---|---|
| H1 tham chiếu chung | 2a + CAR; Zhou + CAR; OpenBMI bỏ CAR (nếu gắn cache no-CAR) | Δ_RA của 2a/Zhou giảm; OpenBMI bỏ CAR thì Δ_RA tăng |
| H2 không lọc tần số | 2a lọc 8–30 Hz; Zhou không lọc | Δ_RA của 2a giảm; Zhou không lọc thì Δ_RA tăng |
| H3 ít trial để ước lượng tham chiếu | 2a: tham chiếu RA cố định chỉ từ 100 / 36 trial (5 lần rút ngẫu nhiên) | Δ_RA (cố định) của 2a giảm |

Mục 6: OpenBMI và RA theo session có sẵn trong cùng bảng.
Chưa làm: OpenBMI chỉ giữ ~20 kênh vận động (cache không lưu tên kênh).

**Thiết lập Kaggle:** Accelerator = **None (CPU)**, Internet = **On**.
**Add Input** (tùy chọn): `openbmi-dataset-cuong` (OpenBMI có CAR), `openbmi-nocar-cuong` (không CAR).

Kết quả lưu ở `/kaggle/working/ra_interventions/` và file zip cùng tên.
"""

SETUP = item4.SETUP

LOADERS = item4.LOADERS.replace(
    "OUT = Path('/kaggle/working/ra_data_characteristics')",
    "OUT = Path('/kaggle/working/ra_interventions')",
) + "\nimport ra_interventions as rai\n"

RUN = r'''results = []
log = lambda msg: print(msg, flush=True)

def run(dataset, intervention, covs, **kwargs):
    started = time.time()
    results.append(rai.ra_gain_table(dataset, intervention, covs, log=log, **kwargs))
    log(f'    ({(time.time() - started) / 60:.1f} min)')
    pd.concat(results, ignore_index=True).to_csv(OUT / 'subject_results.csv', index=False)

# ---- BCI IV-2a: H1 (CAR), H2 (8-30 Hz), H3 (fewer reference trials)
a_raw, a_filt = load_moabb_views('BCI IV-2a', SPECS['BCI IV-2a'], {'raw': None, '8-30': (8.0, 30.0)})
covs = rai.to_covariances(a_raw, log=log)
run('BCI IV-2a', 'original', covs)
run('BCI IV-2a', 'ref 100 trials', covs, ref_trials=100, repeats=5)
run('BCI IV-2a', 'ref 36 trials', covs, ref_trials=36, repeats=5)
run('BCI IV-2a', 'CAR', rai.to_covariances(a_raw, rai.common_average_reference, log=log))
run('BCI IV-2a', '8-30 Hz', rai.to_covariances(a_filt, log=log))
del a_raw, a_filt, covs

# ---- BCI IV-2b: reference only (bipolar, unfiltered)
b_raw, = load_moabb_views('BCI IV-2b', SPECS['BCI IV-2b'], {'raw': None})
run('BCI IV-2b', 'original', rai.to_covariances(b_raw, log=log))
del b_raw

# ---- Zhou2016: pipeline is 8-30 Hz; reverse H2 (unfiltered) and H1 (CAR)
z_pipe, z_raw = load_moabb_views('Zhou2016', SPECS['Zhou2016'], {'8-30': (8.0, 30.0), 'raw': None})
run('Zhou2016', 'original (8-30 Hz)', rai.to_covariances(z_pipe, log=log))
run('Zhou2016', 'unfiltered', rai.to_covariances(z_raw, log=log))
run('Zhou2016', 'CAR', rai.to_covariances(z_pipe, rai.common_average_reference, log=log))
del z_pipe, z_raw

# ---- OpenBMI (item 6): CAR cache is the pipeline; no-CAR cache reverses H1
for view in load_openbmi_views():
    name = 'original (CAR)' if 'no CAR' not in view.name else 'no CAR'
    run('OpenBMI', name, rai.to_covariances(view, log=log))

results = pd.concat(results, ignore_index=True)
results.to_csv(OUT / 'subject_results.csv', index=False)
'''

SUMMARY = r'''BASELINE = {'BCI IV-2a': 'original', 'BCI IV-2b': 'original',
            'Zhou2016': 'original (8-30 Hz)', 'OpenBMI': 'original (CAR)'}
BASELINE = {k: v for k, v in BASELINE.items() if ((results.dataset == k) & (results.intervention == v)).any()}
results = results[results.dataset.isin(BASELINE)]
summary = rai.summarise(results, BASELINE)
summary.to_csv(OUT / 'summary.csv', index=False)
pd.set_option('display.width', 250, 'display.max_columns', 30)
for clf in rai.CLASSIFIERS:
    print(f'===== {clf}')
    display(summary[summary.classifier == clf].drop(columns='classifier').round(4))
'''

PLOTS = r'''import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
labels = [f'{d} | {i}' for d, i in dict.fromkeys(zip(summary.dataset, summary.intervention))]
pos = np.arange(len(labels))
for ax, kind in zip(axes, ('frozen', 'per_session')):
    for k, clf in enumerate(rai.CLASSIFIERS):
        s = summary[summary.classifier == clf]
        ax.barh(pos + (k - 0.5) * 0.4, s[f'gain_{kind}'], height=0.4, label=clf)
    ax.axvline(0, color='0.3', lw=0.8)
    ax.set_yticks(pos, labels, fontsize=8); ax.invert_yaxis()
    ax.set_xlabel(f'Δ_RA ({kind}), accuracy points'); ax.set_title(f'RA gain, {kind} reference')
axes[0].legend()
fig.tight_layout(); fig.savefig(OUT / 'ra_gain_interventions.png', dpi=150); plt.show()

shutil.make_archive(str(OUT), 'zip', root_dir=OUT)
print('Saved', sorted(p.name for p in OUT.iterdir()), 'and', f'{OUT}.zip')
'''

READING = """## Cách đọc kết quả

Cột chính: `change_vs_baseline_frozen` / `_per_session` = Δ_RA sau can thiệp − Δ_RA gốc (cùng bộ dữ liệu, cùng subject), kèm `p_change_*` (Wilcoxon chính xác, ghép cặp theo subject).

| Nếu thấy | Thì |
|---|---|
| 2a + 8–30 Hz: Δ_RA giảm rõ; Zhou không lọc: Δ_RA tăng | Ủng hộ **H2** (lọc tần số) |
| 2a + CAR: Δ_RA giảm rõ; OpenBMI bỏ CAR: Δ_RA tăng | Ủng hộ **H1** (tham chiếu chung) |
| 2a ref 36 trial: Δ_RA (cố định) giảm, còn RA theo session không đổi | Ủng hộ **H3** (chất lượng tham chiếu) |
| MDM và TS-LR cho chiều khác nhau | Kết quả phụ thuộc bộ phân loại, chưa dùng được |

Đặc trưng nào làm Δ_RA của 2a giảm mạnh nhất (ở cả MDM và TS-LR) là ứng viên cho mục 7 (train lại SSM-DAN).
Gửi lại `summary.csv`, `subject_results.csv` và hình (hoặc file zip).
"""


def main():
    cells = [
        item4.cell("markdown", INTRO),
        item4.cell("code", SETUP),
        item4.cell("code", "%%writefile /tmp/ra_data_characteristics.py\n" + RDC),
        item4.cell("code", "%%writefile /tmp/ra_interventions.py\n" + RAI),
        item4.cell("code", LOADERS),
        item4.cell("code", RUN),
        item4.cell("code", SUMMARY),
        item4.cell("code", PLOTS),
        item4.cell("markdown", READING),
    ]
    notebook = {"cells": cells,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python"}},
                "nbformat": 4, "nbformat_minor": 5}
    path = HERE / "ra_interventions.ipynb"
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
