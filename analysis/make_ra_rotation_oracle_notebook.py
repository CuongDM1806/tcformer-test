"""Build ra_rotation_oracle.ipynb (Claim 1, item 15; Kaggle, CPU only).

Same as ra_rotation.ipynb (item 12), plus the oracle rotation: the target rotation fitted with the
TRUE labels of its fit sessions. Diagnostic only, not a method.

Run:  python analysis/make_ra_rotation_oracle_notebook.py
"""
import json
from pathlib import Path

import make_ra_data_characteristics_notebook as item4
import make_ra_rotation_notebook as item12

HERE = Path(__file__).resolve().parent

INTRO = """# Claim 1, mục 15: phép xoay "oracle" (chẩn đoán, không cần GPU)

Mục 12–13: sau RA vẫn còn một phép quay giữa các subject (tăng theo số kênh), nhưng xoay bù bằng **nhãn giả**
làm kết quả tệ đi. Chưa biết vì sao:

- **(a)** phép quay không phải phần mà DA sửa giúp RA, hoặc
- **(b)** phép quay đúng là phần đó, nhưng không ước lượng được khi nhãn giả sai.

Ở đây phép quay của subject đích được fit bằng **nhãn thật** của session huấn luyện của nó (session T với 2a), rồi áp
cho session đánh giá. **Đây là gian lận có chủ ý, chỉ để chẩn đoán**; nhãn của session đánh giá vẫn chỉ dùng để chấm điểm.

| Nếu thấy | Thì |
|---|---|
| `extra_oracle` dương, lớn, **tăng theo số kênh** của 2a | **(b)**: phép quay là cơ chế; thất bại ở mục 12–13 là do nhãn giả |
| `extra_oracle` ≈ 0 ở mọi số kênh | **(a)**: xoay đúng cũng không giúp; bỏ giả thuyết phép quay |

Cùng tập kênh như mục 12: 2a C3/Cz/C4, 8 và 14 kênh ngẫu nhiên (3 lần rút), đủ 22 kênh; 2b; Zhou.

**Thiết lập Kaggle:** Accelerator = **None (CPU)**, Internet = **On**. Không cần Add Input.
Kết quả lưu ở `/kaggle/working/ra_rotation_oracle/` và file zip cùng tên.
"""

LOADERS = item12.LOADERS.replace("OUT = Path('/kaggle/working/ra_rotation')",
                                 "OUT = Path('/kaggle/working/ra_rotation_oracle')")
RUN = item12.RUN.replace("rrp.rotation_table(dataset, view, covs, log=log)",
                         "rrp.rotation_table(dataset, view, covs, log=log, oracle=True)")
SUMMARY = item12.SUMMARY.replace(
    "'acc_per_session', 'acc_per_session+rot', 'extra_rot_per_session', 'p_extra_rot_per_session']",
    "'acc_frozen+oracle', 'extra_oracle_frozen', 'p_extra_oracle_frozen',\n"
    "        'acc_per_session', 'acc_per_session+rot', 'extra_rot_per_session', 'p_extra_rot_per_session',\n"
    "        'acc_per_session+oracle', 'extra_oracle_per_session', 'p_extra_oracle_per_session']")

PLOTS = r'''import matplotlib.pyplot as plt
two_a = summary[summary.dataset == 'BCI IV-2a'].sort_values('n_channels')
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
for ax, base in zip(axes, ('frozen', 'per_session')):
    for clf in rai.CLASSIFIERS:
        s = two_a[two_a.classifier == clf]
        ax.plot(s.n_channels, s[f'extra_oracle_{base}'], 'o-', label=f'{clf}: oracle rotation (true labels)')
        ax.plot(s.n_channels, s[f'extra_rot_{base}'], 's--', label=f'{clf}: rotation, pseudo-labels')
    ax.axhline(0, color='0.4', lw=0.8)
    ax.set(xlabel='channels (BCI IV-2a)', ylabel='accuracy points vs RA alone', title=f'Extra gain of rotation, {base} RA')
    ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(OUT / 'oracle_rotation_vs_channels.png', dpi=150); plt.show()

shutil.make_archive(str(OUT), 'zip', root_dir=OUT)
print('Saved', sorted(p.name for p in OUT.iterdir()), 'and', f'{OUT}.zip')
'''

READING = """## Cách đọc kết quả

Cột chính: `extra_oracle_frozen` (= RA cố định + xoay oracle − RA cố định) theo số kênh của 2a, kèm `p_extra_oracle_frozen`.
So với `extra_rot_frozen` (nhãn giả) để thấy phần mất đi do nhãn giả.

Gửi lại file zip `ra_rotation_oracle.zip`.
"""


def main():
    cells = [item4.cell("markdown", INTRO), item4.cell("code", item4.SETUP)]
    for module in item12.MODULES:
        cells.append(item4.cell("code", f"%%writefile /tmp/{module}\n" + (HERE / module).read_text(encoding="utf-8")))
    cells += [item4.cell("code", LOADERS), item4.cell("code", RUN), item4.cell("code", SUMMARY),
              item4.cell("code", PLOTS), item4.cell("markdown", READING)]
    notebook = {"cells": cells,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python"}},
                "nbformat": 4, "nbformat_minor": 5}
    path = HERE / "ra_rotation_oracle.ipynb"
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
