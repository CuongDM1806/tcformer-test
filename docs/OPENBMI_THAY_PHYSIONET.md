# OpenBMI thay cho PhysioNet trong bài báo (bản tạm)

Ghi chép tạm trong lúc các lần chạy OpenBMI chưa xong. Khi đủ kết quả, chuyển phần
"Bản nháp LaTeX" ở cuối vào `elsarticle-template-num.tex` và xoá mọi chỗ nhắc PhysioNet.

## 1. Vì sao đổi

| | PhysioNet MI (đang có trong bài) | OpenBMI / Lee2019_MI (thay thế) |
|---|---|---|
| Subject | 106 | 54 |
| Session | 1 session/subject | **2 session, ghi ở hai ngày khác nhau** |
| Protocol khả dụng | chỉ transductive (cùng trial vừa adapt vừa test) | **chronological**, giống BCI IV-2a/2b và Zhou2016 |
| Trạng thái trong bài | chỉ dùng cho ablation, các ô SSM-DAN còn trống | — |

PhysioNet buộc phải đánh giá transductive nên bài phải ghi riêng là "không so sánh trực tiếp".
OpenBMI có hai session nên chạy được đúng protocol chronological của bài: adapt trên session 1
không nhãn, test trên session 2. Như vậy kết quả OpenBMI so sánh được với ba dataset còn lại.

## 2. Protocol OpenBMI (theo code)

Branch `feature/hada-autodial-openbmi54`; cấu hình `preprocessing.openbmi` trong
`configs/hada_tcformer.yaml`; datamodule `datamodules/openbmi.py`.

- Dữ liệu: Lee2019_MI (OpenBMI), 54 subject, 62 kênh EEG, lấy mẫu gốc 1000 Hz.
  Mỗi session có một run offline có nhãn, 100 trial, 2 lớp (tay trái/phải, cân bằng).
- Tiền xử lý (theo pipeline cross-subject Ensemble-MI đã công bố): lọc 4–38 Hz, CAR,
  hạ mẫu 100 Hz, cửa sổ 0–4 s sau cue (400 mẫu), z-score. Không dùng augmentation S&R.
- LOSO 54 fold. Với mỗi target:
  - 53 subject nguồn: session 1 có nhãn để train (5.300 trial), session 2 có nhãn để
    validation (chọn checkpoint).
  - Target: session 1 không nhãn cho Stage 1 adaptation; session 2 không nhãn cho IM-TTA
    và là **tập test chính**.
  - RA: reference fit trên session 1 của từng subject, áp nguyên cho session 2.
- Huấn luyện: 125 epoch, batch 48, seed 0, Kaggle T4 x2 (2 fold song song).
  Khoảng 110 phút/fold.
- Mỗi fold full model cho hai kết quả: có IM-TTA (`test_acc`) và cùng trọng số không IM-TTA
  (`no_tta_acc`). Cột "Without test-time adaptation" lấy từ `no_tta_acc`, không cần chạy riêng.
- Test có 100 trial cân bằng 2 lớp nên kappa = 2·acc − 1 và accuracy có bước 1%.

Cột của `summary.csv`:
- job full / no_ra / no_da:
  `run, seed, subject_id, test_acc, test_kappa, no_tta_acc, no_tta_kappa, best_val_acc, train_time_min`
- job source-only (không có cột no_tta):
  `run, seed, subject_id, test_acc, test_kappa, best_val_acc, train_time_min`

## 3. Danh sách thí nghiệm cần cho bảng ablation

| Cột trong bảng ablation | Thư mục notebook | Job | Trạng thái |
|---|---|---|---|
| SSM-DAN | `kaggle_openbmi54_full` | job01 S1–S18 | chưa có kết quả |
| | | job02 S19–S37 | chưa có kết quả |
| | | job03 S38–S54 | **xong (17/17)** |
| Without test-time adaptation | (lấy `no_tta_acc` từ 3 job full) | — | xong S38–S54 |
| Source-only | `kaggle_openbmi54_source_only` | job01 S1–S18 | **xong (18/18)** |
| | | job02 S19–S37 | **18/19 (thiếu S37)** |
| | | job03 S38–S54 | chưa có kết quả |
| Without Riemannian alignment | `kaggle_openbmi54_no_ra` | job01–05 (12 target/job) | chưa có kết quả |
| Without domain adaptation | `kaggle_openbmi54_no_da` | job01–04 (13–14 target/job) | chưa có kết quả |

Khi có thêm `summary.csv`, dán vào mục 4 và cập nhật mục 5.

## 4. Kết quả đã có

### 4.1 Full model, job03 (S38–S54), seed 0

| Subject | Acc IM-TTA (%) | κ IM-TTA | Acc không IM-TTA (%) | κ không IM-TTA | best_val_acc | Train (phút) |
|---|---:|---:|---:|---:|---:|---:|
| 38 | 73 | 0.46 | 76 | 0.52 | 0.7551 | 111.2 |
| 39 | 86 | 0.72 | 81 | 0.62 | 0.7566 | 110.2 |
| 40 | 83 | 0.66 | 81 | 0.62 | 0.7600 | 110.6 |
| 41 | 77 | 0.54 | 78 | 0.56 | 0.7606 | 109.9 |
| 42 | 82 | 0.64 | 81 | 0.62 | 0.7551 | 109.7 |
| 43 | 85 | 0.70 | 81 | 0.62 | 0.7519 | 109.7 |
| 44 | 94 | 0.88 | 94 | 0.88 | 0.7538 | 108.9 |
| 45 | 95 | 0.90 | 94 | 0.88 | 0.7628 | 109.5 |
| 46 | 74 | 0.48 | 73 | 0.46 | 0.7532 | 108.6 |
| 47 | 86 | 0.72 | 80 | 0.60 | 0.7621 | 109.5 |
| 48 | 65 | 0.30 | 63 | 0.26 | 0.7628 | 109.2 |
| 49 | 82 | 0.64 | 79 | 0.58 | 0.7568 | 109.8 |
| 50 | 45 | −0.10 | 42 | −0.16 | 0.7515 | 114.2 |
| 51 | 69 | 0.38 | 65 | 0.30 | 0.7530 | 114.7 |
| 52 | 80 | 0.60 | 76 | 0.52 | 0.7496 | 114.2 |
| 53 | 77 | 0.54 | 68 | 0.36 | 0.7638 | 115.1 |
| 54 | 67 | 0.34 | 67 | 0.34 | 0.7545 | 111.3 |

### 4.2 Tóm tắt S38–S54 (n = 17, **tạm thời**)

| | Accuracy (%) | κ |
|---|---|---|
| SSM-DAN (có IM-TTA) | 77.65 ± 11.95 | 0.553 ± 0.239 |
| Không IM-TTA (cùng trọng số) | 75.24 ± 12.16 | 0.505 ± 0.243 |

- IM-TTA tăng trung bình +2.41 điểm (trung vị +2.0): tăng ở 13 subject, bằng ở 2 (S44 không
  đổi là 94; S54 là 67), giảm ở 2 (S38: 76 → 73; S41: 78 → 77).
- Wilcoxon hai phía (bỏ cặp bằng nhau, n = 15): p ≈ 0.006. Chỉ là con số sơ bộ trên 17/54
  subject; tính lại khi đủ 54.
- Accuracy từ 45% (S50, dưới mức ngẫu nhiên 50%) đến 95% (S45); 13/17 subject đạt ≥ 70%.
- Không đưa các số này vào bài trước khi có đủ 54 subject: 17 subject cuối không phải mẫu
  ngẫu nhiên của toàn dataset.

### 4.3 Source-only, seed 0

#### job01 (S1–S18)

| Subject | Acc (%) | κ | best_val_acc | Train (phút) |
|---|---:|---:|---:|---:|
| 1 | 76 | 0.52 | 0.8311 | 67.5 |
| 2 | 71 | 0.42 | 0.8281 | 67.4 |
| 3 | 98 | 0.96 | 0.8258 | 67.4 |
| 4 | 82 | 0.64 | 0.8298 | 67.0 |
| 5 | 82 | 0.64 | 0.8266 | 67.6 |
| 6 | 85 | 0.70 | 0.8230 | 68.0 |
| 7 | 74 | 0.48 | 0.8330 | 68.8 |
| 8 | 87 | 0.74 | 0.8281 | 68.9 |
| 9 | 75 | 0.50 | 0.8258 | 67.1 |
| 10 | 75 | 0.50 | 0.8298 | 66.9 |
| 11 | 70 | 0.40 | 0.8311 | 68.2 |
| 12 | 82 | 0.64 | 0.8313 | 68.3 |
| 13 | 64 | 0.28 | 0.8302 | 68.8 |
| 14 | 67 | 0.34 | 0.8300 | 67.6 |
| 15 | 78 | 0.56 | 0.8270 | 67.0 |
| 16 | 82 | 0.64 | 0.8313 | 67.2 |
| 17 | 60 | 0.20 | 0.8358 | 67.1 |
| 18 | 95 | 0.90 | 0.8275 | (bị cắt khi dán) |

Tóm tắt job01 (n = 18): accuracy 77.94 ± 9.93%, κ 0.559 ± 0.199; từ 60% (S17) đến 98% (S3);
15/18 subject ≥ 70%.

#### job02 (S19–S36; S37 chưa có)

| Subject | Acc (%) | κ | best_val_acc | Train (phút) |
|---|---:|---:|---:|---:|
| 19 | 75 | 0.50 | 0.8287 | 68.6 |
| 20 | 89 | 0.78 | 0.8217 | 69.5 |
| 21 | 95 | 0.90 | 0.8275 | 68.3 |
| 22 | 77 | 0.54 | 0.8215 | 68.8 |
| 23 | 84 | 0.68 | 0.8275 | 68.8 |
| 24 | 54 | 0.08 | 0.8398 | 69.4 |
| 25 | 97 | 0.94 | 0.8292 | 69.0 |
| 26 | 87 | 0.74 | 0.8302 | 68.8 |
| 27 | 91 | 0.82 | 0.8311 | 69.5 |
| 28 | 95 | 0.90 | 0.8296 | 69.8 |
| 29 | 81 | 0.62 | 0.8285 | 69.1 |
| 30 | 75 | 0.50 | 0.8317 | 69.8 |
| 31 | 90 | 0.80 | 0.8274 | 69.7 |
| 32 | 95 | 0.90 | 0.8296 | 69.9 |
| 33 | 96 | 0.92 | 0.8262 | 69.7 |
| 34 | 55 | 0.10 | 0.8306 | 70.4 |
| 35 | 85 | 0.70 | 0.8291 | 70.9 |
| 36 | 97 | 0.94 | 0.8308 | 71.0 |

Tóm tắt job02 (n = 18): accuracy 84.33 ± 13.11%, κ 0.687 ± 0.262; từ 54% (S24) đến
97% (S25, S36); 16/18 subject ≥ 70%. Khoảng 69.5 phút/fold.

#### Gộp source-only S1–S36 (n = 36, tạm thời)

Accuracy **81.14 ± 11.91%**, κ **0.623 ± 0.238**; từ 54% đến 98%; 31/36 subject ≥ 70%.
Hai job lệch nhau khoảng 6 điểm (77.9% so với 84.3%), cho thấy trung bình theo từng nhóm
subject dao động mạnh, nên chưa thể ngoại suy cho cả 54 subject.

Chưa so sánh được với full model: source-only đang có S1–S36, full đang có S38–S54,
**không trùng subject nào**. Phải so sánh từng cặp trên cùng subject: cần source-only job03
(S38–S54) hoặc full job01–02 (S1–S37).

### 4.4 Cần theo dõi: best_val_acc của full model thấp hơn source-only

`best_val_acc` là accuracy trên session 2 có nhãn của 53 subject nguồn (dùng để chọn
checkpoint). Tập validation gần như giống nhau giữa các fold nên giá trị này rất ổn định:

| | best_val_acc trung bình | Khoảng |
|---|---:|---|
| Source-only (S1–S36) | 0.829 | 0.822–0.840 |
| Full model (S38–S54) | 0.757 | 0.750–0.764 |

Full model thấp hơn khoảng 7 điểm trên **dữ liệu nguồn**, ổn định ở mọi fold. Có thể do:
(a) full model dùng RA còn source-only thì không, nên dữ liệu validation khác nhau;
(b) các loss DA làm giảm khả năng phân biệt lớp trên dữ liệu nguồn;
(c) cách tính validation khác nhau giữa hai pipeline (ví dụ thống kê BatchNorm/AutoDIAL).
Nếu là (b) thì full model có nguy cơ thua source-only trên OpenBMI. Cần kiểm tra ngay khi có
kết quả source-only và full trên cùng subject; job no_ra và no_da sẽ giúp tách (a) với (b).

## 5. Bảng ablation dự kiến (thay hàng PhysioNet)

| Dataset | Metric | SSM-DAN | Source-only | w/o RA | w/o DA | w/o TTA |
|---|---|---|---|---|---|---|
| OpenBMI | Acc. | (54 subject) | (job source-only) | (job no_ra) | (job no_da) | (`no_tta_acc`) |
| | κ | | | | | |

## 6. Những chỗ phải sửa trong bài khi đủ kết quả

File `docs/elsevier_bspc_template/original/elsarticle/elsarticle-template-num.tex`
(số dòng theo bản hiện tại):

1. Dòng 454: thay đoạn mô tả PhysioNet bằng đoạn OpenBMI (mục 7.1).
2. Dòng 479–483: thay hàng "PhysioNet MI" và comment TODO trong `tab:ablation` (mục 7.2).
3. Dòng 489: thay câu cuối nói về PhysioNet bằng câu kết quả OpenBMI (mục 7.3).
4. `references.bib`: xoá `schalk2004bci2000` và `goldberger2000physiobank` (chỉ PhysioNet
   dùng), thêm `lee2019openbmi` (mục 7.4).
5. Cân nhắc: vì OpenBMI theo đúng protocol chronological, có thể đưa OpenBMI thành dataset
   chính thứ tư (bảng dataset, bảng session allocation, bảng kết quả riêng). Việc này cần chạy
   thêm các baseline trên OpenBMI; nếu không chạy thì chỉ giữ OpenBMI trong ablation.

## 7. Bản nháp LaTeX

### 7.1 Đoạn mô tả dataset (thay dòng 454)

```latex
OpenBMI (Lee2019\_MI) \cite{lee2019openbmi} is added as a larger cohort for the ablation. It
contains 54 subjects recorded with 62 EEG channels in two sessions on different days, each with
one offline run of 100 left- versus right-hand MI trials. Following the cross-subject
preprocessing of the published Ensemble-MI pipeline, signals are band-pass filtered at
4--38~Hz, re-referenced to the common average, downsampled to 100~Hz, and cut to the 0--4-s
cue interval. Unlike PhysioNet-style single-session data, the two sessions allow the same
chronological protocol as the other datasets: in each of the 54 LOSO folds, the 53 source
subjects provide labeled session-1 trials for training and session-2 trials for checkpoint
selection, while the target's unlabeled session~1 is used for Stage~1 adaptation and its
session~2 for IM-TTA and testing. Riemannian alignment is fitted on session~1 of each subject.
All variants are trained for 125 epochs with a batch size of 48.
```

### 7.2 Hàng trong `tab:ablation` (thay dòng 479–483)

```latex
\multirow{2}{*}{OpenBMI}
 & Acc. & $XX.XX\pm XX.XX$ & $XX.XX\pm XX.XX$ & $XX.XX\pm XX.XX$ & $XX.XX\pm XX.XX$ & $XX.XX\pm XX.XX$ \\
 & $\kappa$ & $0.XXX\pm0.XXX$ & $0.XXX\pm0.XXX$ & $0.XXX\pm0.XXX$ & $0.XXX\pm0.XXX$ & $0.XXX\pm0.XXX$ \\
```

### 7.3 Câu kết quả (thay câu PhysioNet ở dòng 489)

```latex
On OpenBMI, ... (điền khi đủ 54 subject: SSM-DAN so với từng biến thể, kèm p-value Wilcoxon
và Holm như các dataset khác).
```

### 7.4 Mục tham khảo

```bibtex
@article{lee2019openbmi,
  author  = {Lee, Min-Ho and Kwon, O-Yeon and Kim, Yong-Jeong and Kim, Hong-Kyung and Lee, Young-Eun and Williamson, John and Fazli, Siamac and Lee, Seong-Whan},
  title   = {{EEG} dataset and {OpenBMI} toolbox for three {BCI} paradigms: an investigation into {BCI} illiteracy},
  journal = {GigaScience},
  year    = {2019},
  volume  = {8},
  number  = {5},
  pages   = {giz002},
  doi     = {10.1093/gigascience/giz002}
}
```

Metadata đã đối chiếu với Crossref (DOI 10.1093/gigascience/giz002).
Nếu nhắc tới pipeline Ensemble-MI trong bài thì cũng cần thêm citation cho nó.
