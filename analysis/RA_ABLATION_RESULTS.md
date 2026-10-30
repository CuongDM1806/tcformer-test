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

## 2. Tổng hợp

| Cấu hình | RA off | RA on | Δ mean | Δ median | Tăng/giảm | Wilcoxon p |
|---|---|---|---|---|---|---|
| 2a 3ch LR | 69.29 | 73.38 | +4.09 | +2.78 | 7/2 | 0.152 |
| 2a 3ch 4 lớp | 44.75 | 51.62 | **+6.87** | **+9.37** | 7/2 | **0.020** |
| 2a 22ch LR | 84.41 | 81.87 | −2.55 | +1.39 | 5/4 | 0.793 |
| 2a 22ch 4 lớp | 64.35 | ra01 — chờ | | | | |
| 2b (3 kênh bipolar, LR) | 80.54 | ra08 — chờ | | | | |

Bảng 2×2 (Δ mean / Δ median):

| | LR | 4 lớp |
|---|---|---|
| 3 kênh | +4.09 / +2.78 (p = 0.15) | **+6.87 / +9.37 (p = 0.02)** |
| 22 kênh | −2.55 / +1.39 (p = 0.79) | chờ ra01 |

Nhận xét tạm thời:

- Cùng 3 kênh, RA giúp bài 4 lớp nhiều hơn và ổn định hơn bài LR (median +9.4 so với +2.8; chỉ 4 lớp có ý nghĩa).
  Theo κ trung bình thì hai mức tăng gần nhau (+0.092 so với +0.082) vì S9 kéo LR lên; khác biệt nằm ở median và độ ổn định.
- Ở bài LR, RA gần như không giúp, dù 3 hay 22 kênh. Thêm kênh không làm mức tăng ở LR lớn hơn.
- Đến giờ yếu tố quyết định trông giống **số lớp** hơn là số kênh. Cần ra01 để hoàn chỉnh bảng 2×2.
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

## 5. Còn chờ

| Notebook | Cấu hình | So với |
|---|---|---|
| ra01 | 2a 22ch 4 lớp, RA on | 64.35 (RA off đã có) |
| ra08 | 2b, RA on | 80.54 (RA off đã có) |
| job01–10 / pn01–10 | PhysioNet 4 lớp, 106 subject, RA off / on (batch size 108) | RA off S1–S11 đã có (mục 4); còn S12–S106 và toàn bộ RA on |
