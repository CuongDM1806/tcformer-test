# RA ablation: kết quả trên mô hình source-only

Mô hình: full-Mamba source-only (hoisted scan), LOSO, seed 0, FP32.
Notebook: nhánh `feature/full-mamba-source-only-ra-ablation`, thư mục `kaggle_ra_ablation/`.
RA nghiêm ngặt: mỗi subject dùng một tham chiếu (Riemannian mean của covariance OAS) fit trên
session 1 (session huấn luyện) của chính subject đó, rồi áp cho session 2. Test = session 2 của target.

Accuracy tính bằng %; Δ = RA on − RA off (điểm %). κ của summary = (acc − 1/K) / (1 − 1/K).
Wilcoxon: kiểm định hạng có dấu theo cặp, hai phía, p chính xác (n = 9).

## 1. Kết quả theo subject

### 2a, 3 kênh (C3/Cz/C4), LR — ra06 / ra07

| Subject | RA off | RA on | Δ |
|---|---|---|---|
| S1 | 75.69 | 76.39 | +0.70 |
| S2 | 53.47 | 57.64 | +4.17 |
| S3 | 75.00 | 78.47 | +3.47 |
| S4 | 63.89 | 66.67 | +2.78 |
| S5 | 81.94 | 76.39 | −5.55 |
| S6 | 63.89 | 66.67 | +2.78 |
| S7 | 75.69 | 73.61 | −2.08 |
| S8 | 72.92 | 77.78 | +4.86 |
| S9 | 61.11 | 86.81 | +25.70 |
| **Mean ± SD** | **69.29 ± 9.11** | **73.38 ± 8.53** | **+4.09** |

### 2a, 3 kênh (C3/Cz/C4), 4 lớp — ra04 / ra05

| Subject | RA off | RA on | Δ |
|---|---|---|---|
| S1 | 54.86 | 65.28 | +10.42 |
| S2 | 27.78 | 32.64 | +4.86 |
| S3 | 65.62 | 77.43 | +11.81 |
| S4 | 42.71 | 45.83 | +3.12 |
| S5 | 38.19 | 48.26 | +10.07 |
| S6 | 34.38 | 43.75 | +9.37 |
| S7 | 42.36 | 40.62 | −1.74 |
| S8 | 53.47 | 52.78 | −0.69 |
| S9 | 43.40 | 57.99 | +14.59 |
| **Mean ± SD** | **44.75 ± 11.53** | **51.62 ± 13.61** | **+6.87** |

### 2a, 22 kênh, LR — ra02 / ra03

| Subject | RA off | RA on | Δ |
|---|---|---|---|
| S1 | 79.17 | 81.94 | +2.77 |
| S2 | 79.86 | 76.39 | −3.47 |
| S3 | 91.67 | 90.28 | −1.39 |
| S4 | 84.03 | 87.50 | +3.47 |
| S5 | 85.42 | 50.00 | −35.42 |
| S6 | 81.25 | 85.42 | +4.17 |
| S7 | 87.50 | 88.89 | +1.39 |
| S8 | 97.22 | 94.44 | −2.78 |
| S9 | 73.61 | 81.94 | +8.33 |
| **Mean ± SD** | **84.41 ± 7.09** | **81.87 ± 13.08** | **−2.55** |

S5 RA on: ghi chú ở mục 3.

### 2a, 22 kênh, 4 lớp — RA off (bảng chính của bài) / ra01

RA off lấy từ bảng source-only của bài báo (64.35; chạy bằng **sequential scan**). ra01 chạy bằng
**hoisted scan** (như mọi run ablation khác) — xem cảnh báo ở mục 2.

| Subject | RA off | RA on | Δ |
|---|---|---|---|
| S1 | 68.06 | 75.69 | +7.63 |
| S2 | 51.04 | 43.06 | −7.98 |
| S3 | 86.46 | 89.24 | +2.78 |
| S4 | 61.11 | 61.46 | +0.35 |
| S5 | 64.24 | 36.46 | −27.78 |
| S6 | 39.58 | 56.60 | +17.02 |
| S7 | 72.92 | 64.93 | −7.99 |
| S8 | 73.61 | 73.26 | −0.35 |
| S9 | 62.15 | 69.10 | +6.95 |
| **Mean ± SD** | **64.35 ± 13.55** | **63.31 ± 16.36** | **−1.04** |

### 2b, 3 kênh bipolar, LR — RA off (bảng chính của bài) / ra08

| Subject | RA off | RA on | Δ |
|---|---|---|---|
| S1 | 77.81 | 74.69 | −3.12 |
| S2 | 68.57 | 62.86 | −5.71 |
| S3 | 63.13 | 63.75 | +0.62 |
| S4 | 89.38 | 92.81 | +3.43 |
| S5 | 96.56 | 97.81 | +1.25 |
| S6 | 73.44 | 85.62 | +12.18 |
| S7 | 84.69 | 88.13 | +3.44 |
| S8 | 86.25 | 91.25 | +5.00 |
| S9 | 85.00 | 85.31 | +0.31 |
| **Mean ± SD** | **80.54 ± 10.66** | **82.47 ± 12.57** | **+1.93** |

## 2. Tổng hợp

| Cấu hình | RA off | RA on | Δ mean | Δ median | Tăng/giảm | Wilcoxon p |
|---|---|---|---|---|---|---|
| 2a 3ch LR | 69.29 | 73.38 | +4.09 | +2.78 | 7/2 | 0.152 |
| 2a 3ch 4 lớp | 44.75 | 51.62 | **+6.87** | **+9.37** | 7/2 | **0.020** |
| 2a 22ch LR | 84.41 | 81.87 | −2.55 | +1.39 | 5/4 | 0.793 |
| 2a 22ch 4 lớp ⚠ | 64.35 | 63.31 | −1.04 | +0.35 | 5/4 | 1.000 |
| 2b (3 kênh bipolar, LR) | 80.54 | 82.47 | +1.93 | +1.25 | 7/2 | 0.250 |

⚠ **Lệch scan mode:** RA off 64.35 chạy bằng sequential scan, ra01 (RA on) bằng hoisted scan. Hai scan
tương đương về toán nhưng thứ tự cộng số thực khác nhau nên quá trình train phân kỳ; cùng config, RA off với
hoisted scan cho 62.27 (không có số theo subject). Nếu so với 62.27 thì Δ mean ≈ +1.04 thay vì −1.04 — vẫn
nhỏ và không có ý nghĩa. Muốn so sạch cần chạy thêm **2a 22ch 4 lớp RA off bằng hoisted scan** (cùng nhánh
ablation). Cần kiểm tra thêm scan mode của RA off 2b (80.54).

Bảng 2×2 (Δ mean / Δ median):

| | LR | 4 lớp |
|---|---|---|
| 3 kênh | +4.09 / +2.78 (p = 0.15) | **+6.87 / +9.37 (p = 0.02)** |
| 22 kênh | −2.55 / +1.39 (p = 0.79) | −1.04 / +0.35 (p = 1.00) ⚠ |

Bỏ subject có |Δ| lớn nhất (kiểm tra độ nhạy, chỉ để tham khảo): 2a 22ch 4 lớp bỏ S5 → Δ mean +2.30
(p = 0.64); 2b bỏ S6 → Δ mean +0.65 (p = 0.46).

Nhận xét (1 seed, n = 9):

- **RA chỉ giúp có ý nghĩa ở một ô: 3 kênh, 4 lớp.** Với 22 kênh, RA không giúp cả bài LR lẫn bài 4 lớp.
  Giả thuyết trước ("số lớp là yếu tố quyết định") chỉ đúng khi có 3 kênh; bảng 2×2 cho thấy có **tương tác
  giữa số kênh và số lớp**, không phải một yếu tố đơn lẻ.
- **Mô hình không lặp lại phân tích MDM ở 22 kênh.** Với MDM, 2a 22ch 4 lớp là cấu hình RA giúp nhiều nhất
  (+12.2 pp, p = 0.004); với mô hình thì không. Giả thuyết (chưa kiểm chứng): với 22 kênh, lớp tích chập không
  gian của mạng đã học được phần lớn việc bù khác biệt covariance giữa subject, nên RA không còn thêm gì; với
  3 kênh, mạng có ít bậc tự do không gian hơn nên RA còn tác dụng.
- **2b:** +1.93 (p = 0.25), khớp với MDM (+1 pp, p = 0.50): RA không giúp có ý nghĩa ở 2b.
- **S5 của 2a bị RA làm hại nặng ở cả hai cấu hình 22 kênh** (LR −35.42, 4 lớp −27.78) nhưng không ở 3 kênh
  (LR −5.55, 4 lớp +10.07). Phù hợp với giải thích ở mục 3: tham chiếu RA fit trên session 1 không khớp
  session 2; lệch này nặng hơn khi whiten 22 kênh.
- **Hệ quả cho bài báo:** không nên viết "RA cải thiện kết quả trên BCI IV-2a" cho cấu hình 22 kênh dựa trên
  ablation source-only này. Nếu mô hình chính (HADA, 22 kênh) dùng RA, cần ablation RA on/off **trên chính mô
  hình đó** trước khi khẳng định vai trò của RA.
- Mới 1 seed; nên chạy thêm seed 1, 2 trước khi viết thành kết luận.

## 3. Ghi chú: S5, 22 kênh LR, RA on

Giữ nguyên kết quả trong bảng (không loại, không coi là lỗi). Log `seed0_S005.log` của ra03:

| Tập | Acc | Loss | κ |
|---|---|---|---|
| Session 2 (test, số chính thức) | 50.00 | 1.467 | 0.000 |
| Session 1+2 (phụ trợ) | 72.22 | 0.808 | 0.444 |

Mô hình đạt 94.44% trên session 1 (suy ra: 2 × 72.22 − 50.00). Session 1 cũng là session dùng để fit tham chiếu RA.
Trên session 2, mô hình dự đoán cùng một lớp cho mọi trial. Khi tắt RA, session 2 đạt 85.42%.
Vậy mức giảm của S5 nhiều khả năng đến từ lệch giữa hai session: tham chiếu RA fit trên session 1 không còn khớp với session 2.

## 4. PhysioNet MI, 4 lớp, RA off (batch size 108)

Run `physionet_4cls_ra_off_s001-s011`: subject S1–S11, seed 0, batch size 108. Mới có RA off; RA on còn chờ.
κ ở bảng này là κ do script tính trên từng subject (không phải công thức suy từ acc ở trên).

| Subject | Test acc | Test κ | Best val acc | Thời gian train (phút) |
|---|---|---|---|---|
| S1 | 68.89 | 0.5843 | 65.52 | 133.6 |
| S2 | 58.89 | 0.4516 | 65.52 | 134.2 |
| S3 | 73.33 | 0.6462 | 65.71 | 133.9 |
| S4 | 67.78 | 0.5707 | 64.00 | 134.5 |
| S5 | 43.33 | 0.2491 | 63.81 | 132.9 |
| S6 | 65.56 | 0.5422 | 64.38 | 132.9 |
| S7 | 77.78 | 0.7038 | 64.38 | 133.8 |
| S8 | 66.67 | 0.5564 | 63.43 | 133.9 |
| S9 | 71.11 | 0.6136 | 64.19 | 133.5 |
| S10 | 38.89 | 0.1872 | 65.14 | 133.8 |
| S11 | 52.22 | 0.3623 | 64.76 | 137.6 |
| **Mean ± SD** | **62.22 ± 12.50** | **0.497 ± 0.166** | **64.62 ± 0.77** | **134.1 ± 1.3** |
| Median | 66.67 | 0.556 | 64.38 | 133.8 |

Tổng thời gian train: 1474.6 phút (≈ 24.6 giờ) cho 11 subject.

Nhận xét: S5 (43.33) và S10 (38.89) thấp hơn hẳn các subject còn lại, kéo mean xuống dưới median khoảng 4.5 điểm.
Best val acc gần như không đổi giữa các subject (63.4–65.7), nên val không phản ánh độ khó của từng target.

### Run `physionet_4cls_ra_off_s012-s022` (job02)

Subject S12–S22, seed 0, đủ 11 subject.

| Subject | Test acc | Test κ | Best val acc | Thời gian train (phút) |
|---|---|---|---|---|
| S12 | 54.44 | 0.3952 | 65.14 | 135.4 |
| S13 | 55.56 | 0.4082 | 65.52 | 135.5 |
| S14 | 40.00 | 0.2051 | 65.71 | 135.0 |
| S15 | 83.33 | 0.7775 | 64.95 | 134.9 |
| S16 | 52.22 | 0.3624 | 64.00 | 135.6 |
| S17 | 58.89 | 0.4535 | 64.19 | 135.7 |
| S18 | 83.33 | 0.7776 | 64.00 | 135.9 |
| S19 | 65.56 | 0.5401 | 65.14 | 135.9 |
| S20 | 64.44 | 0.5266 | 65.52 | 135.4 |
| S21 | 51.11 | 0.3452 | 66.48 | 135.4 |
| S22 | 80.00 | 0.7337 | 64.76 | 135.0 |
| **Mean ± SD** | **62.63 ± 14.33** | **0.502 ± 0.190** | **65.04 ± 0.77** | **135.4 ± 0.3** |
| Median | 58.89 | | | |

Tổng thời gian train: 1489.7 phút (≈ 24.8 giờ) cho 11 subject.

### Run `physionet_4cls_ra_off_s023-s033` (job03)

Subject S23–S33, seed 0, đủ 11 subject.

| Subject | Test acc | Test κ | Best val acc | Thời gian train (phút) |
|---|---|---|---|---|
| S23 | 52.22 | 0.3613 | 64.95 | 131.1 |
| S24 | 47.78 | 0.3046 | 64.76 | 133.8 |
| S25 | 37.78 | 0.1669 | 64.76 | 131.5 |
| S26 | 37.78 | 0.1728 | 65.52 | 133.9 |
| S27 | 63.33 | 0.5114 | 63.43 | 132.0 |
| S28 | 55.56 | 0.4084 | 64.95 | 134.1 |
| S29 | 58.89 | 0.4518 | 65.14 | 132.3 |
| S30 | 52.22 | 0.3676 | 64.76 | 134.4 |
| S31 | 56.67 | 0.4230 | 64.95 | 132.1 |
| S32 | 85.56 | 0.8069 | 64.76 | 134.1 |
| S33 | 72.22 | 0.6299 | 64.19 | 129.3 |
| **Mean ± SD** | **56.36 ± 13.99** | **0.419 ± 0.187** | **64.74 ± 0.54** | **132.6 ± 1.6** |
| Median | 55.56 | | | |

Tổng thời gian train: 1458.6 phút (≈ 24.3 giờ) cho 11 subject.

### Run `physionet_4cls_ra_off_s034-s044` (job04)

Subject S34–S44, seed 0, đủ 11 subject.

| Subject | Test acc | Test κ | Best val acc | Thời gian train (phút) |
|---|---|---|---|---|
| S34 | 51.19 | 0.3492 | 64.19 | 135.9 |
| S35 | 80.00 | 0.7335 | 64.19 | 133.0 |
| S36 | 70.00 | 0.5995 | 64.57 | 133.9 |
| S37 | 70.24 | 0.6032 | 65.14 | 136.4 |
| S38 | 32.22 | 0.0985 | 64.57 | 134.4 |
| S39 | 47.78 | 0.3019 | 64.57 | 136.1 |
| S40 | 77.78 | 0.7026 | 63.62 | 135.4 |
| S41 | 58.33 | 0.4444 | 64.57 | 137.5 |
| S42 | 57.78 | 0.4380 | 63.81 | 133.5 |
| S43 | 74.44 | 0.6590 | 65.14 | 135.5 |
| S44 | 58.89 | 0.4529 | 65.90 | 129.8 |
| **Mean ± SD** | **61.70 ± 14.51** | **0.489 ± 0.193** | **64.57 ± 0.65** | **134.7 ± 2.1** |
| Median | 58.89 | | | |

Tổng thời gian train: 1481.4 phút (≈ 24.7 giờ) cho 11 subject.

S38 (32.22%, κ 0.099) thấp nhất trong tất cả các subject đã có, gần mức đoán ngẫu nhiên (25%).

### Run `physionet_4cls_ra_off_s045-s055` (job05)

Subject S45–S55, seed 0, đủ 11 subject.

| Subject | Test acc | Test κ | Best val acc | Thời gian train (phút) |
|---|---|---|---|---|
| S45 | 52.22 | 0.3639 | 64.19 | 130.8 |
| S46 | 41.11 | 0.2167 | 64.38 | 131.9 |
| S47 | 56.67 | 0.4190 | 64.19 | 130.6 |
| S48 | 74.44 | 0.6601 | 64.57 | 131.6 |
| S49 | 60.00 | 0.4673 | 64.57 | 131.8 |
| S50 | 57.78 | 0.4320 | 64.95 | 132.1 |
| S51 | 76.19 | 0.6825 | 65.14 | 131.7 |
| S52 | 72.22 | 0.6294 | 64.00 | 132.6 |
| S53 | 75.56 | 0.6741 | 64.76 | 131.4 |
| S54 | 67.78 | 0.5712 | 65.90 | 131.9 |
| S55 | 80.00 | 0.7337 | 64.57 | 130.9 |
| **Mean ± SD** | **64.91 ± 12.21** | **0.532 ± 0.163** | **64.66 ± 0.53** | **131.6 ± 0.6** |
| Median | 67.78 | | | |

Tổng thời gian train: 1447.3 phút (≈ 24.1 giờ) cho 11 subject.

### Run `physionet_4cls_ra_off_s056-s066` (job06)

Subject S56–S66, seed 0, đủ 11 subject (S66 có trong lần chạy resume). Batch size giả định 108.

| Subject | Test acc | Test κ | Best val acc | Thời gian train (phút) |
|---|---|---|---|---|
| S56 | 70.00 | 0.5999 | 64.57 | 132.3 |
| S57 | 86.67 | 0.8219 | 64.95 | 133.9 |
| S58 | 71.11 | 0.6144 | 65.14 | 132.3 |
| S59 | 56.67 | 0.4232 | 64.57 | 135.2 |
| S60 | 62.22 | 0.4944 | 64.57 | 132.3 |
| S61 | 80.00 | 0.7328 | 65.14 | 134.1 |
| S62 | 67.78 | 0.5701 | 64.76 | 133.0 |
| S63 | 43.33 | 0.2449 | 65.14 | 134.8 |
| S64 | 60.71 | 0.4762 | 65.14 | 132.5 |
| S65 | 65.56 | 0.5429 | 65.14 | 133.8 |
| S66 | 41.11 | 0.2124 | 65.14 | 132.7 |
| **Mean ± SD** | **64.11 ± 13.76** | **0.521 ± 0.184** | **64.93 ± 0.26** | **133.4 ± 1.1** |
| Median | 65.56 | | | |

Tổng thời gian train: 1466.9 phút (≈ 24.4 giờ) cho 11 subject.

### Run `physionet_4cls_ra_off_s067-s077` (job07)

Subject S67–S77, seed 0, đủ 11 subject (S77 có trong lần chạy sau). Batch size giả định 108.

| Subject | Test acc | Test κ | Best val acc | Thời gian train (phút) |
|---|---|---|---|---|
| S67 | 37.78 | 0.1704 | 65.71 | 131.1 |
| S68 | 63.33 | 0.5112 | 64.57 | 133.1 |
| S69 | 55.56 | 0.4051 | 65.71 | 131.3 |
| S70 | 90.00 | 0.8666 | 64.95 | 132.6 |
| S71 | 74.44 | 0.6598 | 64.19 | 131.4 |
| S72 | 61.90 | 0.4921 | 63.62 | 133.0 |
| S73 | 61.90 | 0.4921 | 64.38 | 131.6 |
| S74 | 42.86 | 0.2381 | 63.81 | 133.5 |
| S75 | 41.11 | 0.2149 | 65.33 | 131.7 |
| S76 | 42.86 | 0.2381 | 64.38 | 134.3 |
| S77 | 45.56 | 0.2720 | 64.57 | 130.0 |
| **Mean ± SD** | **56.12 ± 16.21** | **0.415 ± 0.216** | **64.66 ± 0.70** | **132.1 ± 1.3** |
| Median | 55.56 | | | |

Tổng thời gian train: 1453.6 phút (≈ 24.2 giờ) cho 11 subject.

S72/S73 và S74/S76 trùng cả acc lẫn κ: các subject này có 84 trial (đã bỏ trial bị cắt), nên acc rời rạc
(52/84 = 61.90%, 36/84 = 42.86%); và khi 4 lớp cân bằng (21 trial/lớp) thì κ = (acc − 0.25)/0.75 chỉ phụ thuộc
acc. Thời gian train khác nhau nên là các lần train riêng — không phải lỗi sao chép kết quả.

### Run `physionet_4cls_ra_off_s078-s088` (job08)

Subject S78–S87, seed 0. S88 không có trong kết quả vì thuộc nhóm bị loại (ghi ở 128 Hz; `exclude_subject_ids: [88, 92, 100]`).
Batch size giả định giống run S1–S11 (108) — cần đối chiếu với notebook job08.

| Subject | Test acc | Test κ | Best val acc | Thời gian train (phút) |
|---|---|---|---|---|
| S78 | 41.11 | 0.2152 | 64.95 | 135.3 |
| S79 | 52.22 | 0.3606 | 63.62 | 130.4 |
| S80 | 41.11 | 0.2165 | 64.57 | 130.9 |
| S81 | 67.78 | 0.5709 | 64.76 | 135.9 |
| S82 | 63.33 | 0.5097 | 65.14 | 131.2 |
| S83 | 41.11 | 0.2142 | 64.57 | 135.4 |
| S84 | 61.11 | 0.4822 | 64.38 | 131.3 |
| S85 | 64.44 | 0.5258 | 64.76 | 135.2 |
| S86 | 78.89 | 0.7186 | 64.57 | 130.8 |
| S87 | 41.11 | 0.2155 | 65.14 | 135.9 |
| **Mean ± SD** | **55.22 ± 13.78** | **0.403 ± 0.184** | **64.65 ± 0.44** | **133.2 ± 2.5** |
| Median | 56.66 | | | |

Tổng thời gian train: 1332.3 phút (≈ 22.2 giờ) cho 10 subject.

Nhận xét:

- 4 subject (S78, S80, S83, S87) có **cùng accuracy 41.11% = 37/90** nhưng κ khác nhau (0.214–0.217), nên
  dự đoán không giống hệt nhau. Có thể chỉ là trùng hợp (mỗi subject 90 trial nên accuracy chỉ nhận các giá trị
  rời rạc), nhưng nên xem confusion matrix của 4 subject này để chắc mô hình không dồn dự đoán về 1–2 lớp.
- Best val acc lại gần như không đổi (63.6–65.1), giống run S1–S11: val không phản ánh độ khó của target.

### Run `physionet_4cls_ra_off_s089-s099` (job09)

Subject S89–S99 trừ S92 (bị loại, ghi ở 128 Hz), seed 0. Batch size giả định 108 như các run trước.

| Subject | Test acc | Test κ | Best val acc | Thời gian train (phút) |
|---|---|---|---|---|
| S89 | 73.33 | 0.6441 | 64.95 | 133.2 |
| S90 | 78.89 | 0.7185 | 64.57 | 133.2 |
| S91 | 62.22 | 0.4971 | 65.14 | 133.3 |
| S93 | 56.67 | 0.4222 | 65.52 | 133.3 |
| S94 | 56.67 | 0.4214 | 65.71 | 134.8 |
| S95 | 50.00 | 0.3327 | 65.33 | 134.7 |
| S96 | 83.33 | 0.7777 | 65.52 | 133.7 |
| S97 | 60.00 | 0.4678 | 64.95 | 133.7 |
| S98 | 50.00 | 0.3319 | 65.71 | 133.2 |
| S99 | 44.44 | 0.2588 | 64.95 | 133.2 |
| **Mean ± SD** | **61.55 ± 13.01** | **0.487 ± 0.174** | **65.23 ± 0.38** | **133.6 ± 0.6** |
| Median | 58.34 | | | |

Tổng thời gian train: 1336.3 phút (≈ 22.3 giờ) cho 10 subject.

### Run `physionet_4cls_ra_off_s100-s109` (job10)

Subject S101–S109 (S100 bị loại, ghi ở 128 Hz), seed 0, đủ 9 subject.

| Subject | Test acc | Test κ | Best val acc | Thời gian train (phút) |
|---|---|---|---|---|
| S101 | 90.00 | 0.8670 | 64.95 | 132.3 |
| S102 | 61.90 | 0.4921 | 64.19 | 132.3 |
| S103 | 73.33 | 0.6440 | 64.19 | 132.2 |
| S104 | 49.43 | 0.3236 | 65.90 | 132.2 |
| S105 | 75.56 | 0.6729 | 65.33 | 132.2 |
| S106 | 45.56 | 0.2726 | 64.76 | 132.2 |
| S107 | 44.44 | 0.2597 | 65.33 | 132.0 |
| S108 | 70.00 | 0.5997 | 64.00 | 132.0 |
| S109 | 45.56 | 0.2737 | 65.14 | 131.1 |
| **Mean ± SD** | **61.75 ± 16.46** | **0.489 ± 0.220** | **64.87 ± 0.64** | **132.1 ± 0.4** |
| Median | 61.90 | | | |

Tổng thời gian train: 1188.5 phút (≈ 19.8 giờ) cho 9 subject.

### Gộp toàn bộ RA off (đủ 106 subject)

| Nhóm | n | Acc mean ± SD | Median | κ mean ± SD |
|---|---|---|---|---|
| S1–S11 (job01) | 11 | 62.22 ± 12.50 | 66.67 | 0.497 ± 0.166 |
| S12–S22 (job02) | 11 | 62.63 ± 14.33 | 58.89 | 0.502 ± 0.190 |
| S23–S33 (job03) | 11 | 56.36 ± 13.99 | 55.56 | 0.419 ± 0.187 |
| S34–S44 (job04) | 11 | 61.70 ± 14.51 | 58.89 | 0.489 ± 0.193 |
| S45–S55 (job05) | 11 | 64.91 ± 12.21 | 67.78 | 0.532 ± 0.163 |
| S56–S66 (job06) | 11 | 64.11 ± 13.76 | 65.56 | 0.521 ± 0.184 |
| S67–S77 (job07) | 11 | 56.12 ± 16.21 | 55.56 | 0.415 ± 0.216 |
| S78–S87 (job08) | 10 | 55.22 ± 13.78 | 56.66 | 0.403 ± 0.184 |
| S89–S99 trừ S92 (job09) | 10 | 61.55 ± 13.01 | 58.34 | 0.487 ± 0.174 |
| S101–S109 (job10) | 9 | 61.75 ± 16.46 | 61.90 | 0.489 ± 0.220 |
| **Tất cả** | **106** | **60.68 ± 13.89** | **60.36** | **0.476 ± 0.185** |

Đã đủ 106/106 subject. Mean thay đổi ít khi thêm subject
(51 → 74 → 85 → 96 → 105 → 106 subject: 60.55 → 61.16 → 61.35 → 60.78 → 60.87 → 60.68). Thời gian mỗi fold ổn định quanh 133 phút trên T4, và best val acc gần như hằng số
(63.4–65.9) ở mọi run.

## 5. Còn chờ

ra01 (2a 22ch 4 lớp, RA on) và ra08 (2b, RA on) đã xong — xem mục 1 và 2.

| Notebook | Cấu hình | So với |
|---|---|---|
| (mới) | 2a 22ch 4 lớp, **RA off, hoisted scan** | ra01 — để bỏ lệch scan mode |
| (kiểm tra) | Scan mode của RA off 2b (80.54) | ra08 |
| job01–10 / pn01–10 | PhysioNet 4 lớp, 106 subject, RA off / on (batch size 108) | RA off **đủ 106 subject** (mục 4); còn toàn bộ RA on |
| seed 1, 2 | Mọi cấu hình ở mục 2 | |
