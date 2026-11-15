# OpenBMI không CAR (kiểm tra giả thuyết H1 của Claim 1)

Ghi tạm kết quả các run OpenBMI **không áp CAR**. Khi đủ 54 subject sẽ tổng hợp vào
`CLAIM1_RA_BCI2A.md` và, nếu cần, vào bài.

## 1. Mục đích và thiết lập

**Câu hỏi.** RA có giúp OpenBMI không, khi thành phần tham chiếu chung còn nằm trong tín hiệu giống
BCI IV-2a? Đây là H1 trong [CLAIM1_RA_BCI2A.md](CLAIM1_RA_BCI2A.md).

**Thay đổi duy nhất:** `preprocessing.openbmi.common_average_reference: false`. Các bước còn lại
giữ nguyên: lọc 4–38 Hz, 100 Hz, cửa sổ 0–4 s, z-score, 125 epoch, batch 48, seed 0.

**Code.**
- Branch `feature/ssm-dan-openbmi-no-car` (commit `0a4b1a1d`), thư mục `kaggle_openbmi_no_car/`.
- Cache: Kaggle Dataset `openbmi-nocar-cuong`.

**Ba biến thể, mỗi biến thể 3 job** (S1–S18, S19–S37, S38–S54):

| Run | Mô tả |
|---|---|
| `openbmi_nocar_full_*` | RA cố định (reference = session 1) + AutoDIAL + IM-TTA |
| `openbmi_nocar_no_ra_*` | như full nhưng bỏ RA |
| `openbmi_nocar_per_session_ra_*` | mỗi session dùng reference RA riêng, không nhãn |

Mọi fold đều được chấm cả có IM-TTA (`test_*`) lẫn không IM-TTA (`no_tta_*`).

## 2. Trạng thái

| Biến thể | S1–S18 | S19–S37 | S38–S54 |
|---|---|---|---|
| Full (RA cố định) | **xong (18/18)** | **xong (19/19)** | **xong (17/17)** |
| No RA | **xong (18/18)** | **xong (19/19)** | chưa có |
| Per-session RA | chưa có | chưa có | chưa có |

## 3. Kết quả

### 3.1 Full, RA cố định, no-CAR (đủ 54 subject)

#### Job S1–S18 (run `openbmi_nocar_full_s001-s018`)

| S | Acc IM-TTA (%) | κ | Acc không IM-TTA (%) | κ | best_val_acc | Train (phút) |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 85 | 0.70 | 81 | 0.62 | 0.7585 | 110.1 |
| 2 | 74 | 0.48 | 57 | 0.14 | 0.7568 | 113.4 |
| 3 | 95 | 0.90 | 97 | 0.94 | 0.7513 | 115.7 |
| 4 | 83 | 0.66 | 71 | 0.42 | 0.7683 | 118.6 |
| 5 | 86 | 0.72 | 78 | 0.56 | 0.7558 | 115.9 |
| 6 | 88 | 0.76 | 84 | 0.68 | 0.7570 | 118.3 |
| 7 | 79 | 0.58 | 76 | 0.52 | 0.7638 | 116.5 |
| 8 | 88 | 0.76 | 83 | 0.66 | 0.7509 | 119.3 |
| 9 | 76 | 0.52 | 77 | 0.54 | 0.7504 | 117.7 |
| 10 | 74 | 0.48 | 61 | 0.22 | 0.7645 | 119.8 |
| 11 | 74 | 0.48 | 74 | 0.48 | 0.7598 | 109.9 |
| 12 | 79 | 0.58 | 74 | 0.48 | 0.7619 | 108.4 |
| 13 | 66 | 0.32 | 65 | 0.30 | 0.7628 | 109.1 |
| 14 | 75 | 0.50 | 63 | 0.26 | 0.7555 | 109.6 |
| 15 | 83 | 0.66 | 61 | 0.22 | 0.7630 | 111.6 |
| 16 | 82 | 0.64 | 81 | 0.62 | 0.7526 | 112.4 |
| 17 | 76 | 0.52 | 67 | 0.34 | 0.7543 | 115.1 |
| 18 | 84 | 0.68 | 78 | 0.56 | 0.7525 | 115.9 |

Tóm tắt job S1–S18 (n = 18; std ddof = 0):
- Có IM-TTA: **80.39 ± 6.72%**, κ 0.608 ± 0.134.
- Không IM-TTA: 73.78 ± 9.83%, κ 0.476 ± 0.197.
- `best_val_acc` trung bình 0.758 (0.750–0.768). Thời gian train trung bình 114.3 phút/fold.

#### Job S19–S37 (run `openbmi_nocar_full_s019-s037`)

| S | Acc IM-TTA (%) | κ | Acc không IM-TTA (%) | κ | best_val_acc | Train (phút) |
|---|---:|---:|---:|---:|---:|---:|
| 19 | 85 | 0.70 | 86 | 0.72 | 0.7611 | 116.0 |
| 20 | 89 | 0.78 | 86 | 0.72 | 0.7555 | 116.2 |
| 21 | 98 | 0.96 | 96 | 0.92 | 0.7474 | 116.7 |
| 22 | 80 | 0.60 | 71 | 0.42 | 0.7511 | 117.0 |
| 23 | 80 | 0.60 | 82 | 0.64 | 0.7583 | 117.7 |
| 24 | 68 | 0.36 | 61 | 0.22 | 0.7564 | 116.9 |
| 25 | 93 | 0.86 | 91 | 0.82 | 0.7362 | 117.5 |
| 26 | 87 | 0.74 | 83 | 0.66 | 0.7649 | 117.0 |
| 27 | 84 | 0.68 | 81 | 0.62 | 0.7479 | 116.9 |
| 28 | 95 | 0.90 | 93 | 0.86 | 0.7598 | 116.6 |
| 29 | 88 | 0.76 | 84 | 0.68 | 0.7779 | 117.0 |
| 30 | 84 | 0.68 | 76 | 0.52 | 0.7547 | 115.6 |
| 31 | 84 | 0.68 | 77 | 0.54 | 0.7592 | 116.7 |
| 32 | 98 | 0.96 | 98 | 0.96 | 0.7462 | 119.5 |
| 33 | 93 | 0.86 | 93 | 0.86 | 0.7479 | 111.1 |
| 34 | 64 | 0.28 | 63 | 0.26 | 0.7670 | 111.8 |
| 35 | 84 | 0.68 | 78 | 0.56 | 0.7558 | 109.1 |
| 36 | 97 | 0.94 | 97 | 0.94 | 0.7564 | 111.1 |
| 37 | 92 | 0.84 | 82 | 0.64 | 0.7500 | 107.1 |

Tóm tắt (n = 19): có IM-TTA 86.47 ± 8.98%; không IM-TTA 83.05 ± 10.30%.

#### Job S38–S54 (run `openbmi_nocar_full_s038-s054`)

| S | Acc IM-TTA (%) | κ | Acc không IM-TTA (%) | κ | best_val_acc | Train (phút) |
|---|---:|---:|---:|---:|---:|---:|
| 38 | 72 | 0.44 | 69 | 0.38 | 0.7485 | 108.1 |
| 39 | 82 | 0.64 | 80 | 0.60 | 0.7543 | 107.8 |
| 40 | 72 | 0.44 | 70 | 0.40 | 0.7494 | 108.9 |
| 41 | 67 | 0.34 | 65 | 0.30 | 0.7475 | 108.3 |
| 42 | 79 | 0.58 | 81 | 0.62 | 0.7549 | 107.5 |
| 43 | 88 | 0.76 | 85 | 0.70 | 0.7594 | 107.4 |
| 44 | 95 | 0.90 | 94 | 0.88 | 0.7519 | 107.9 |
| 45 | 96 | 0.92 | 96 | 0.92 | 0.7662 | 107.9 |
| 46 | 75 | 0.50 | 78 | 0.56 | 0.7532 | 108.1 |
| 47 | 89 | 0.78 | 82 | 0.64 | 0.7615 | 109.4 |
| 48 | 65 | 0.30 | 60 | 0.20 | 0.7632 | 108.7 |
| 49 | 89 | 0.78 | 84 | 0.68 | 0.7589 | 108.9 |
| 50 | 52 | 0.04 | 56 | 0.12 | 0.7591 | 110.3 |
| 51 | 65 | 0.30 | 72 | 0.44 | 0.7566 | 107.6 |
| 52 | 83 | 0.66 | 81 | 0.62 | 0.7600 | 108.8 |
| 53 | 70 | 0.40 | 65 | 0.30 | 0.7538 | 108.8 |
| 54 | 71 | 0.42 | 70 | 0.40 | 0.7681 | 105.4 |

Tóm tắt (n = 17): có IM-TTA 77.06 ± 11.66%; không IM-TTA 75.76 ± 10.87%.

#### Tóm tắt đủ 54 subject (std ddof = 0, κ = 2·acc − 1)

| Cấu hình | Acc (%) | κ |
|---|---|---|
| No-CAR, RA cố định, IM-TTA | **81.48 ± 10.08** | 0.630 ± 0.202 |
| No-CAR, RA cố định, không IM-TTA | 77.67 ± 11.10 | 0.553 ± 0.222 |
| *CAR, RA cố định, IM-TTA (bài)* | *80.61 ± 10.26* | *0.612 ± 0.205* |
| *CAR, RA cố định, không IM-TTA (bài)* | *76.65 ± 12.19* | *0.533 ± 0.244* |
| *CAR, w/o RA (bài)* | *83.50 ± 9.43* | *0.670 ± 0.189* |
| *CAR, PS-RA, IM-TTA* | *84.00 ± 9.56* | *0.680 ± 0.191* |

`best_val_acc` trung bình 0.757 (0.736–0.778). Train trung bình 112.7 phút/fold.

### 3.2 So sánh cùng subject (Acc %, Wilcoxon hai phía)

Số liệu CAR theo từng subject lấy từ `OPENBMI_THAY_PHYSIONET.md` (mục 4.1, 4.3, 4.5). Một số
baseline thiếu subject trong file đó, nên n khác nhau giữa các dòng.

| No-CAR full so với | n | TB no-CAR | TB baseline | Chênh lệch TB (trung vị) | Tăng/bằng/giảm | p |
|---|---:|---:|---:|---:|---|---:|
| CAR, RA cố định (thiếu S25, S26) | 52 | 81.15 | 80.40 | +0.75 (+1.0) | 29/4/19 | 0.23 |
| CAR, w/o RA (thiếu S25, S26, S38–S40, S53, S54) | 47 | 81.98 | 83.38 | −1.40 (−2.0) | 13/4/30 | 0.059 |
| CAR, PS-RA | 54 | 81.48 | 84.00 | −2.52 (−2.0) | 20/0/34 | 0.0021 |
| CAR, source-only (S1–S36) | 36 | 83.28 | 81.14 | +2.14 (+1.5) | 21/5/10 | 0.039 |
| Không IM-TTA: no-CAR RA cố định so với CAR PS-RA | 54 | 77.67 | 83.89 | −6.22 (−4.5) | 7/5/42 | < 0.0001 |
| Tác dụng IM-TTA trong run no-CAR | 54 | 81.48 | 77.67 | +3.81 (+3.0) | 41/5/8 | < 0.0001 |

### 3.3 Nhận xét (full đủ 54; chưa có no-CAR w/o RA nên chưa tính được Δ_RA)

1. **Bỏ CAR gần như không thay đổi RA cố định.**
   - No-CAR đạt 81.48%, CAR đạt 80.61%. So cùng 52 subject: +0.75, p = 0.23.
   - No-CAR vẫn thấp hơn CAR w/o RA (−1.40, p = 0.059) và thấp hơn rõ CAR PS-RA (−2.52, p = 0.002).
2. **Độ lệch giữa hai session vẫn là vấn đề chính, CAR không phải nguyên nhân.**
   - `best_val_acc` vẫn khoảng 0.76, giống CAR với RA cố định (khoảng 0.75); PS-RA đạt khoảng 0.85.
   - Không IM-TTA chỉ đạt 77.67%, kém PS-RA 6.2 điểm.
   - IM-TTA vẫn phải bù +3.8 điểm, gần bằng mức +3.96 khi có CAR.
3. **H1 chưa được ủng hộ** (đã kiểm tra trực tiếp ở mục 3.5–3.6).
   - Nếu tham chiếu chung là lý do RA giúp 2a, thì bỏ CAR phải làm RA cố định trên OpenBMI tốt lên
     rõ rệt. Điều đó không xảy ra.
   - Kết luận chắc chắn cần Δ_RA trên cùng dữ liệu no-CAR, tức là `openbmi_nocar_no_ra_*`.
   - Cũng cần biết PS-RA no-CAR có hơn PS-RA CAR không, tức là `openbmi_nocar_per_session_ra_*`.
4. **Hiệu ứng theo nhóm subject vẫn lớn.** Job S19–S37 đạt 86.5%, job S38–S54 đạt 77.1%, giống
   các run CAR. Khi so sánh luôn phải ghép cặp theo subject.

### 3.4 No RA, no-CAR, S1–S37 (run `openbmi_nocar_no_ra_s001-s018`, `_s019-s037`)

| S | Acc IM-TTA (%) | κ | Acc không IM-TTA (%) | κ | best_val_acc | Train (phút) |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 84 | 0.68 | 80 | 0.60 | 0.8172 | 107.9 |
| 2 | 72 | 0.44 | 65 | 0.30 | 0.8138 | 107.7 |
| 3 | 94 | 0.88 | 91 | 0.82 | 0.8168 | 107.6 |
| 4 | 83 | 0.66 | 84 | 0.68 | 0.8157 | 107.6 |
| 5 | 92 | 0.84 | 91 | 0.82 | 0.8121 | 108.1 |
| 6 | 90 | 0.80 | 91 | 0.82 | 0.8192 | 108.0 |
| 7 | 82 | 0.64 | 78 | 0.56 | 0.8155 | 108.0 |
| 8 | 83 | 0.66 | 88 | 0.76 | 0.8145 | 107.8 |
| 9 | 85 | 0.70 | 87 | 0.74 | 0.8100 | 108.3 |
| 10 | 72 | 0.44 | 74 | 0.48 | 0.8268 | 107.3 |
| 11 | 63 | 0.26 | 61 | 0.22 | 0.8183 | 107.5 |
| 12 | 85 | 0.70 | 85 | 0.70 | 0.8204 | 107.6 |
| 13 | 71 | 0.42 | 69 | 0.38 | 0.8145 | 109.5 |
| 14 | 79 | 0.58 | 76 | 0.52 | 0.8162 | 109.2 |
| 15 | 80 | 0.60 | 68 | 0.36 | 0.8115 | 110.3 |
| 16 | 81 | 0.62 | 83 | 0.66 | 0.8134 | 109.6 |
| 17 | 71 | 0.42 | 67 | 0.34 | 0.8147 | 115.1 |
| 18 | 92 | 0.84 | 91 | 0.82 | 0.8149 | 113.5 |
| 19 | 90 | 0.80 | 89 | 0.78 | 0.8125 | 112.5 |
| 20 | 87 | 0.74 | 90 | 0.80 | 0.8068 | 107.4 |
| 21 | 96 | 0.92 | 100 | 1.00 | 0.8094 | 108.1 |
| 22 | 78 | 0.56 | 77 | 0.54 | 0.8200 | 110.8 |
| 23 | 86 | 0.72 | 86 | 0.72 | 0.8138 | 107.6 |
| 24 | 71 | 0.42 | 67 | 0.34 | 0.8177 | 111.7 |
| 25 | 96 | 0.92 | 97 | 0.94 | 0.8134 | 108.3 |
| 26 | 91 | 0.82 | 94 | 0.88 | 0.8194 | 111.6 |
| 27 | 84 | 0.68 | 85 | 0.70 | 0.8109 | 107.2 |
| 28 | 95 | 0.90 | 92 | 0.84 | 0.8181 | 111.4 |
| 29 | 78 | 0.56 | 77 | 0.54 | 0.8136 | 107.6 |
| 30 | 84 | 0.68 | 85 | 0.70 | 0.8211 | 110.8 |
| 31 | 89 | 0.78 | 86 | 0.72 | 0.8221 | 109.8 |
| 32 | 97 | 0.94 | 96 | 0.92 | 0.8185 | 114.2 |
| 33 | 99 | 0.98 | 96 | 0.92 | 0.8179 | 108.3 |
| 34 | 68 | 0.36 | 66 | 0.32 | 0.8281 | 113.0 |
| 35 | 81 | 0.62 | 61 | 0.22 | 0.8175 | 108.5 |
| 36 | 91 | 0.82 | 90 | 0.80 | 0.8091 | 111.6 |
| 37 | 93 | 0.86 | 93 | 0.86 | (bị | cắt) |

Dòng S37 bị cắt khi dán (mất `best_val_acc` và thời gian train). Accuracy vẫn đủ.

**Tóm tắt (n = 37; std ddof = 0):**
- Có IM-TTA: **84.14 ± 8.93%**, κ 0.683 ± 0.179. Theo job: S1–S18 81.06 ± 8.28; S19–S37 87.05 ± 8.54.
- Không IM-TTA: 82.59 ± 10.82%, κ 0.652.
- `best_val_acc` trung bình 0.816 (0.807–0.828), cao hơn full RA cố định (khoảng 0.757).
- Train trung bình 109.5 phút/fold.

### 3.5 Δ_RA trên dữ liệu no-CAR (cùng subject S1–S37)

Δ_RA = Acc(full, RA cố định) − Acc(no RA). Wilcoxon hai phía.

| So sánh | n | TB A | TB B | Chênh lệch TB (trung vị) | Tăng/bằng/giảm | p |
|---|---:|---:|---:|---:|---|---:|
| **Δ_RA no-CAR, có IM-TTA** (full − no RA) | 37 | 83.51 | 84.14 | **−0.62** (0.0) | 16/4/17 | 0.27 |
| Δ_RA no-CAR, không IM-TTA | 37 | 78.54 | 82.59 | −4.05 (−4.0) | 8/1/28 | 0.0017 |
| Δ_RA CAR, có IM-TTA (RA cố định − w/o RA; thiếu S25, S26) | 35 | 81.74 | 83.77 | −2.03 (−1.0) | 13/4/18 | 0.050 |
| Δ_RA no-CAR so với Δ_RA CAR (cùng 35 subject) | 35 | −0.46 | −2.03 | +1.57 | | 0.18 |
| No RA: no-CAR so với CAR | 35 | 83.60 | 83.77 | −0.17 (+1.0) | 18/3/14 | 0.95 |
| CAR PS-RA so với no-CAR no RA | 37 | 85.08 | 84.14 | +0.95 (0.0) | 17/3/17 | 0.35 |

### 3.6 Kết luận về H1 (tạm, S1–S37)

1. **H1 bị bác trên OpenBMI.**
   - Khi bỏ CAR, RA cố định vẫn không giúp: Δ_RA = −0.62 (p = 0.27). Không IM-TTA thì RA còn gây
     hại: −4.05, p = 0.002.
   - Δ_RA của no-CAR không khác có ý nghĩa so với Δ_RA của CAR (−0.46 so với −2.03, p = 0.18).
   - Trên 2a, Δ_RA là +5.83 (RA cố định) và +7.45 (PS-RA). OpenBMI không đạt mức đó dù đã giữ thành
     phần tham chiếu chung.
2. **CAR gần như không ảnh hưởng tới mạng khi không dùng RA.** No-CAR no RA đạt 83.60 trên cùng
   35 subject, CAR w/o RA đạt 83.77 (p = 0.95).
3. **Lý do RA cố định gây hại vẫn là lệch giữa hai session.** `best_val_acc`: no RA khoảng 0.82,
   RA cố định khoảng 0.76. Không IM-TTA, RA cố định kém no RA 4 điểm.
4. **Hệ quả cho Claim 1.** Lợi ích của RA trên 2a nhiều khả năng đến từ yếu tố khác tham chiếu
   chung. Ứng viên còn lại:
   - H2: 2a không lọc tần số;
   - H3: chất lượng ước lượng reference;
   - H4: dạng khác biệt giữa các subject.

   Còn thiếu no-CAR no RA S38–S54 và PS-RA no-CAR để chốt.

## 4. Việc còn lại

- No-CAR no RA: còn job S38–S54 (S1–S37 đã xong; Δ_RA tạm thời ở mục 3.5).
- No-CAR per-session RA: cả 3 job.
- Bổ sung số từng subject còn thiếu trong `OPENBMI_THAY_PHYSIONET.md`:
  - CAR RA cố định: S25, S26;
  - CAR w/o RA: S25, S26, S38–S40, S53, S54;
  - source-only: S37–S54.

  Có các số này thì các so sánh ở mục 3.2 mới đủ n = 54.
