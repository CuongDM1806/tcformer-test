# SSM-DAN (source-only) so với TCFormer: số trong bài báo

Các số dưới đây lấy từ `docs/elsevier_bspc_template/original/elsarticle/elsarticle-template-num.tex`;
bản tiếng Việt có cùng số. Thiết lập: LOSO, seed 0.

## 1. Độ chính xác

| Bộ dữ liệu | Mô hình | Acc (%) | κ | ITR (bit/phút) | p / p_Holm (so với SSM-DAN) |
|---|---|---|---|---|---|
| BCI IV-2a (4 lớp) | SSM-DAN source-only | 64.35 ± 12.77 ⚠ | 0.525 ± 0.170 | 8.22 ± 4.68 | 0.0039 / 0.0312 |
| | TCFormer | 60.80 ± 11.49 | 0.477 ± 0.153 | 6.80 ± 3.79 | 0.0039 / 0.0312 |
| BCI IV-2b (2 lớp) | SSM-DAN source-only | 80.54 ± 10.05 | 0.611 ± 0.201 | 5.07 ± 3.24 | 0.0156 / 0.1094 |
| | TCFormer | 81.18 ± 9.37 | 0.624 ± 0.187 | 5.17 ± 2.98 | 0.0820 / 0.3281 |
| Zhou2016 (3 lớp) | SSM-DAN source-only | 75.17 ± 3.51 | 0.627 ± 0.053 | 7.10 ± 1.22 | 0.1250 / 1.0000 |
| | TCFormer | 73.50 ± 6.22 | 0.603 ± 0.093 | 6.67 ± 2.07 | 0.1250 / 1.0000 |

Acc theo từng chủ thể:

| Bộ | Mô hình | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 |
|---|---|---|---|---|---|---|---|---|---|---|
| 2a | SSM-DAN source-only | 68.06 | 51.04 | 86.46 | 61.11 | 64.24 | 39.58 | 72.92 | 73.61 | 62.15 |
| 2a | TCFormer | 65.28 | 49.65 | 79.86 | 54.86 | 61.81 | 38.89 | 71.88 | 66.67 | 58.33 |
| 2b | SSM-DAN source-only | 77.81 | 68.57 | 63.13 | 89.38 | 96.56 | 73.44 | 84.69 | 86.25 | 85.00 |
| 2b | TCFormer | 78.44 | 69.64 | 64.38 | 90.62 | 95.00 | 75.94 | 86.87 | 85.31 | 84.38 |
| Zhou | SSM-DAN source-only | 72.00 | 78.67 | 71.33 | 78.67 | | | | | |
| Zhou | TCFormer | 68.00 | 78.67 | 66.67 | 80.67 | | | | | |

## 2. Chi phí tính toán

| Bộ dữ liệu | Mô hình | Số tham số (K) | Train giai đoạn 1 (phút) | Độ trễ CPU, batch 1 (ms) |
|---|---|---|---|---|
| 2a | SSM-DAN source-only | 98.4 | 47.01 ⚠ | 25.08 ⚠ |
| | TCFormer | 127.2 | 38.37 | 14.96 |
| 2b | SSM-DAN source-only | 94.6 | 36.10 | 13.68 |
| | TCFormer | 123.4 | 31.04 | 11.17 |
| Zhou2016 | SSM-DAN source-only | 96.8 | 6.29 | 13.36 |
| | TCFormer | 125.6 | 5.89 | 15.97 |

## 3. Nhận xét

- Độ chính xác: SSM-DAN source-only cao hơn TCFormer trên 2a (+3.55 điểm, dùng số 64.35 trong bài) và
  Zhou2016 (+1.67 điểm), nhưng thấp hơn một chút trên 2b (−0.64 điểm). Cả hai mô hình đều thấp hơn
  SSM-DAN đầy đủ (2b: 84.09).
- So giữa các bộ dữ liệu: TCFormer có độ chính xác và κ cao nhất trên 2b, nhưng ITR lại thấp nhất
  (5.17 so với 6.80 trên 2a và 6.67 trên Zhou2016), vì 2b chỉ có 2 lớp. Không nên nói TCFormer "tốt hơn"
  trên 2b.
- Số tham số: SSM-DAN source-only ít hơn TCFormer khoảng 23% trên cả ba bộ.
- Thời gian: SSM-DAN source-only train chậm hơn TCFormer trên cả ba bộ. Độ trễ CPU cũng chậm hơn trên 2a
  và 2b, nhưng nhanh hơn trên Zhou2016. Độ trễ 25.08 ms trên 2a cao gần gấp đôi 2b và Zhou dù số tham số
  gần như nhau, nên cần kiểm tra lại.

## 4. Các số cần cập nhật

**2a source-only (⚠).** Số 64.35 được chạy bằng **sequential scan**. Code của bài trên `main` dùng
**hoisted scan** (`configs/methods/ssm_dan.yaml`: `ssm_scan: hoisted`). Với hoisted scan, cùng config cho
**62.27** (xem `RA_ABLATION_RESULTS.md`, mục cảnh báo scan mode). Nếu dùng số này, SSM-DAN source-only chỉ
còn hơn TCFormer trên 2a 1.47 điểm, thay vì 3.55.

Hiện chưa có acc theo từng chủ thể, κ, ITR, p hay thời gian cho số 62.27. Cần chạy lại 2a source-only bằng
code `main` rồi mới cập nhật. Các chỗ phải sửa (cả bản EN lẫn VI):
- dòng source-only trong bảng 2a
- cột 2a trong bảng ablation
- dòng 2a trong bảng tính toán
- câu ITR "8.2 bit/phút" (khoảng 60% tăng) ở phần thảo luận

**2b source-only.** Chạy lại bằng code `main` (hoisted scan) cho kết quả khớp với bài:

| Chỉ số | Run mới | Bài báo |
|---|---|---|
| test_acc | 80.54 ± 10.05 | 80.54 ± 10.05 |
| test_kappa | 0.61 ± 0.20 | 0.611 ± 0.201 |
| Số tham số | 94 632 | 94.6K |
| train_time_min | 32.68 ± 0.14 | 36.10 |
| cpu_latency_ms | 11.16 ± 0.97 | 13.68 |

Acc và κ khớp với bài. Thời gian và độ trễ khác với bài, có thể do khác máy hoặc mức tải. Nếu cập nhật số
thời gian thì nên đo lại TCFormer trên cùng máy để so cho công bằng.
