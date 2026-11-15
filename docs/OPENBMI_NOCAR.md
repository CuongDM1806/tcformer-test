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
| Full (RA cố định) | **xong (18/18)** | chưa có | chưa có |
| No RA | chưa có | chưa có | chưa có |
| Per-session RA | chưa có | chưa có | chưa có |

## 3. Kết quả

### 3.1 Full, RA cố định, no-CAR, job S1–S18 (run `openbmi_nocar_full_s001-s018`)

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

**Tóm tắt (n = 18; std với ddof = 0, giống bảng trong bài):**
- Có IM-TTA: **80.39 ± 6.72%**, κ 0.608 ± 0.134.
- Không IM-TTA: 73.78 ± 9.83%, κ 0.476 ± 0.197.
- `best_val_acc` trung bình 0.758 (0.750–0.768). Thời gian train trung bình 114.3 phút/fold.

### 3.2 So sánh cùng subject S1–S18 (Acc %, Wilcoxon hai phía)

Các cột "CAR" lấy từ `OPENBMI_THAY_PHYSIONET.md`: full RA cố định, w/o RA và PS-RA ở mục 4.5;
source-only ở mục 4.3.

| No-CAR full (80.39) so với | TB baseline | Chênh lệch TB | Tăng/bằng/giảm | p |
|---|---:|---:|---|---:|
| CAR, full RA cố định | 78.67 | +1.72 | 12/1/5 | 0.092 |
| CAR, w/o RA | 80.17 | +0.22 | 7/1/10 | 0.67 |
| CAR, PS-RA | 82.61 | −2.22 | 6/0/12 | 0.024 |
| CAR, source-only | 77.94 | +2.44 | 13/1/4 | 0.041 |
| No-CAR, cùng run nhưng không IM-TTA (tác dụng IM-TTA) | 73.78 | +6.61 | 15/1/2 | 0.0008 |

So sánh không IM-TTA: no-CAR RA cố định đạt 73.78, còn CAR PS-RA đạt 82.50. Chênh −8.72, p < 0.001.

### 3.3 Nhận xét tạm (mới 18/54 subject; chưa có no-CAR w/o RA nên chưa tính được Δ_RA)

1. **Bỏ CAR không làm RA cố định "hoạt động" như trên 2a.** No-CAR với RA cố định (80.39) chỉ
   ngang CAR w/o RA (80.17), và vẫn thấp hơn CAR với PS-RA (82.61, p = 0.024).
2. **Vấn đề lệch giữa hai session vẫn còn:**
   - `best_val_acc` vẫn khoảng 0.76, giống CAR với RA cố định (0.75), trong khi PS-RA đạt khoảng 0.85;
   - không IM-TTA chỉ đạt 73.78%;
   - IM-TTA phải bù tới +6.6 điểm.

   Như vậy độ lệch năng lượng giữa session, chứ không phải CAR, là nguyên nhân chính khiến RA
   cố định kém trên OpenBMI.
3. **Chưa kết luận được H1.** Phải có `openbmi_nocar_no_ra_s001_s018`
   (Δ_RA = full − no RA trên cùng dữ liệu no-CAR) và `openbmi_nocar_per_session_ra_s001_s018`.
   Nếu no-CAR có Δ_RA lớn rõ rệt hơn CAR (−2.89 với RA cố định, +0.50 với PS-RA), H1 mới
   được ủng hộ.

## 4. Việc còn lại

- No-CAR full: S19–S37, S38–S54.
- No-CAR no RA: cả 3 job (ưu tiên S1–S18 để tính Δ_RA ngay).
- No-CAR per-session RA: cả 3 job (ưu tiên S1–S18).
