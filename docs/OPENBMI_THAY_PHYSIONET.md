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

**Đã rõ nguyên nhân (2026-11-12):** là (a). Khi tính RA theo từng session (mục 4.5),
`best_val_acc` của full model lên khoảng 0.85 (0.842–0.862).

### 4.5 Thử nghiệm RA theo từng session (2026-11-12, seed 0)

**Bối cảnh.** Full model dùng RA cố định: reference fit trên session 1 của từng subject rồi áp
nguyên cho session 2. Trên OpenBMI, năng lượng session 2 sau RA có trung vị gấp 1.55 lần
session 1 (20/54 subject gấp hơn 2 lần); BCI IV-2a chỉ 1.08–1.39 lần. Kết quả là full model
(80.61%) thấp hơn w/o RA (83.50%).

**Thay đổi.** Mỗi session dùng reference riêng, tính từ chính các trial **không nhãn** của
session đó (`preprocessing.<ds>.ra_reference: per_session`). Code nằm trên branch
`feature/ssm-dan-per-session-ra` (worktree `tcformer-per-session-ra`, commit `e485ee4c`),
notebook ở `kaggle_per_session_ra/`. Run: `openbmi_full_per_session_ra_*`.

**Trạng thái:** ✅ đủ 54/54 subject (7 subject cuối nhận ngày 2026-11-13, run `openbmi_full_per_session_ra_leftover`).

7 subject cuối (Acc %, IM-TTA / không IM-TTA): S25 90/91, S26 83/86, S38 78/78, S39 94/93, S40 86/86, S53 81/81, S54 66/67.

#### Tóm tắt trên đủ 54 subject (std ddof = 0, giống bảng trong bài)

| Cấu hình | Acc (%) | κ |
|---|---|---|
| **RA theo session, full** | **84.00 ± 9.56** | 0.680 ± 0.191 |
| RA theo session, không IM-TTA | 83.89 ± 9.57 | 0.678 |
| w/o RA | 83.50 ± 9.43 | 0.670 |
| RA cố định, full | 80.61 ± 10.26 | 0.612 |
| RA cố định, không IM-TTA | 76.65 ± 12.19 | 0.533 |

| So sánh (Wilcoxon hai phía) | Chênh lệch TB | Tăng/bằng/giảm | p |
|---|---|---|---|
| PS-RA so với RA cố định | +3.39 | 43/3/8 | 3.3e-06 |
| PS-RA so với RA cố định (đều không IM-TTA) | +7.24 | 45/6/3 | 5.3e-08 |
| PS-RA so với w/o RA | +0.50 | 26/5/23 | 0.61 |
| Tác dụng IM-TTA khi dùng PS-RA | +0.11 | 21/15/18 | 0.50 |
| Tác dụng IM-TTA khi dùng RA cố định | +3.96 | 38/8/8 | 3.6e-06 |
| PS-RA so với source-only (53, không S37; baseline vẫn RA cố định) | +3.85 | 37/4/12 | 0.00045 |
| PS-RA so với w/o DA (baseline vẫn RA cố định) | +4.15 | 41/2/11 | 5.4e-06 |

Kết luận không đổi so với bản 47 subject bên dưới.

#### Kết quả từng subject (Acc %, IM-TTA / không IM-TTA)

| S | PS-RA | PS-RA no TTA | RA cố định | w/o RA | | S | PS-RA | PS-RA no TTA | RA cố định | w/o RA |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|
| 1 | 86 | 84 | 83 | 81 | | 27 | 81 | 82 | 78 | 90 |
| 2 | 81 | 78 | 68 | 69 | | 28 | 97 | 97 | 95 | 98 |
| 3 | 99 | 98 | 99 | 97 | | 29 | 87 | 87 | 85 | 89 |
| 4 | 82 | 80 | 80 | 85 | | 30 | 82 | 84 | 88 | 84 |
| 5 | 89 | 90 | 85 | 93 | | 31 | 92 | 93 | 84 | 87 |
| 6 | 87 | 87 | 85 | 88 | | 32 | 97 | 97 | 92 | 95 |
| 7 | 83 | 83 | 81 | 74 | | 33 | 99 | 99 | 96 | 98 |
| 8 | 87 | 85 | 84 | 83 | | 34 | 61 | 60 | 69 | 63 |
| 9 | 79 | 81 | 82 | 77 | | 35 | 75 | 83 | 74 | 89 |
| 10 | 72 | 76 | 67 | 78 | | 36 | 98 | 99 | 93 | 91 |
| 11 | 69 | 73 | 71 | 69 | | 37 | 96 | 95 | 90 | 94 |
| 12 | 82 | 87 | 74 | 82 | | 41 | 79 | 78 | 77 | 78 |
| 13 | 68 | 69 | 70 | 70 | | 42 | 82 | 81 | 82 | 82 |
| 14 | 73 | 74 | 70 | 77 | | 43 | 87 | 87 | 85 | 86 |
| 15 | 90 | 90 | 77 | 76 | | 44 | 93 | 94 | 94 | 96 |
| 16 | 91 | 86 | 82 | 86 | | 45 | 97 | 96 | 95 | 94 |
| 17 | 80 | 76 | 71 | 67 | | 46 | 77 | 78 | 74 | 79 |
| 18 | 89 | 88 | 87 | 91 | | 47 | 96 | 96 | 86 | 93 |
| 19 | 89 | 89 | 86 | 85 | | 48 | 71 | 66 | 65 | 69 |
| 20 | 94 | 93 | 94 | 94 | | 49 | 83 | 82 | 82 | 90 |
| 21 | 100 | 98 | 99 | 99 | | 50 | 64 | 62 | 45 | 64 |
| 22 | 77 | 78 | 76 | 76 | | 51 | 72 | 67 | 69 | 73 |
| 23 | 81 | 81 | 71 | 83 | | 52 | 82 | 82 | 80 | 83 |
| 24 | 82 | 79 | 75 | 74 | | | | | | |

`best_val_acc` của run PS-RA: 0.842–0.862 (trước đây với RA cố định khoảng 0.75).

#### Tóm tắt trên 47 subject chung

| Cấu hình | Acc (%) | κ |
|---|---|---|
| **RA theo session, full** | **84.21 ± 9.81** | 0.684 |
| RA theo session, không IM-TTA | 84.00 ± 9.87 | 0.680 |
| w/o RA | 83.38 ± 9.80 | 0.668 |
| RA cố định, full | 80.74 ± 10.60 | 0.615 |
| RA cố định, không IM-TTA | 76.38 ± 12.65 | 0.528 |
| w/o DA (RA cố định) | 80.15 ± 10.31 | 0.603 |
| Source-only (46 subject, không có S37) | 79.70 ± 11.40 | 0.594 |

#### Wilcoxon theo cặp subject (hai phía)

| So sánh | Chênh lệch TB (trung vị) | Tăng/bằng/giảm | p |
|---|---|---|---|
| PS-RA so với RA cố định | +3.47 (+3.0) | 38/3/6 | < 0.0001 |
| PS-RA so với w/o RA | +0.83 (0.0) | 23/5/19 | 0.35 |
| PS-RA so với source-only (n = 46) | +4.26 (+4.0) | 34/4/8 | 0.0001 |
| PS-RA so với w/o DA | +4.06 (+3.0) | 37/1/9 | < 0.0001 |
| Tác dụng IM-TTA khi dùng PS-RA | +0.21 (0.0) | 20/12/15 | 0.32 |
| PS-RA so với RA cố định (đều không IM-TTA) | +7.62 (+6.0) | 40/5/2 | < 0.0001 |

#### Nhận xét

1. Giả thuyết lệch năng lượng giữa hai session được xác nhận: chỉ đổi reference RA, accuracy
   tăng ở 38/47 subject và `best_val_acc` tăng khoảng 10 điểm.
2. PS-RA ngang với w/o RA (+0.83, không có ý nghĩa thống kê): RA không còn gây hại, nhưng
   trên OpenBMI cũng không giúp thêm đáng kể.
3. ⚠️ Lợi ích của IM-TTA gần như biến mất (+0.21; với RA cố định là khoảng +4.4). Phần lớn
   tác dụng của IM-TTA trên OpenBMI trước đây là bù lại độ lệch do RA cố định. Nếu chuyển
   sang PS-RA thì phải viết lại phần đóng góp của IM-TTA.
4. So sánh với source-only và w/o DA **chưa công bằng** vì hai baseline này vẫn dùng RA cố
   định. Phải chạy lại cả hai với PS-RA trước khi khẳng định DA giúp tăng khoảng 4 điểm.
5. PS-RA là thiết lập transductive: cần thấy một lô trial không nhãn của session test trước
   khi suy luận. Phải nói rõ điều này trong bài.

#### Đối chiếu trên BCI IV-2a và 2b (seed 0)

BCI IV-2a, run `ssm_dan_per_session_ra_bcic2a_s01_s09` (Acc %, session E, 288 trial):

| | A01 | A02 | A03 | A04 | A05 | A06 | A07 | A08 | A09 | TB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| PS-RA, IM-TTA | 80.21 | 54.17 | 93.75 | 72.92 | 73.96 | 67.36 | 89.24 | 89.58 | 78.47 | 77.74 ± 11.72 (κ 0.703) |
| PS-RA, không IM-TTA | 79.17 | 54.86 | 92.71 | 71.18 | 63.89 | 64.24 | 82.99 | 88.19 | 76.39 | 74.85 ± 12.40 (κ 0.665) |
| RA cố định, IM-TTA (bài) | 79.51 | 52.43 | 93.40 | 73.96 | 69.44 | 65.97 | 88.89 | 86.11 | 75.35 | 76.12 ± 11.96 (κ 0.682) |

- PS-RA so với RA cố định (đều có IM-TTA): +1.62, tăng 8/9, Wilcoxon p = 0.023.
- IM-TTA khi dùng PS-RA: +2.89, tăng 8/9, p = 0.0078. Với RA cố định, bài ghi 76.12 so với 72.15 (+3.97).
  Trên 2a, IM-TTA **vẫn có đóng góp**, khác với OpenBMI (+0.21).
- 2b: PS-RA 84.10 ± 10.23 so với RA cố định 84.09 ± 10.42 (ngang nhau; 4 subject tăng, 4 giảm).

#### Việc còn lại

- ~~Chạy nốt S25, S26, S38–S40, S53, S54~~: xong 2026-11-13.
- ~~Chạy BCI IV-2a với PS-RA~~: xong, 2a tăng (xem trên). Còn Zhou2016
  (`ssm_dan_per_session_ra_zhou2016_s01_s04.ipynb`).
- 7 subject OpenBMI còn thiếu: `ssm_dan_per_session_ra_openbmi_leftover.ipynb` (commit `0c6f6039`).
- BCI IV-2b: lần chạy lại nhận ngày 2026-11-12 cho 84.10% (trước đây 84.09%). Cần kiểm tra
  `config.yaml` có `ra_reference: per_session` không, vì chưa có notebook PS-RA cho 2b.
- Nếu chuyển hẳn: chạy lại source-only, w/o DA, w/o IM-TTA trên OpenBMI với PS-RA; sửa
  claim (iv) so với RMETNet và phần IM-TTA trong cả bản EN lẫn VI.

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
