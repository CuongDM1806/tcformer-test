# WDAN – tái lập (She et al., TNSRE 2023) trên BCI IV 2a, 4 lớp

Q. She et al., "Improved Domain Adaptation Network Based on Wasserstein Distance for
Motor Imagery EEG Classification", IEEE TNSRE 31:1137–1148, 2023 (DOI 10.1109/TNSRE.2023.3241846).
Dựng lại theo **toàn văn bài gốc**. Những chi tiết bài không nêu được ghi ở mục "Giả định".

## Cài đặt & chạy
```bash
pip install -r requirements.txt
python train.py                      # 9 subject, 200 epoch (lần đầu MOABB tự tải dữ liệu)
python train.py --no_da              # baseline: cùng mạng, tắt Wasserstein DA
python train.py --win_sec 2.0        # thử kích thước cửa sổ variance khác (Fig. 5 của bài)
python train.py --subjects 1 3 --epochs 100
python train.py --synthetic --epochs 2   # kiểm tra code, không cần dữ liệu
```
Kết quả: `results_summary.csv` (từng subject) và `results_A0x_history.csv` (theo epoch).

## Theo bài gốc
| Mục | Bài gốc | Vị trí trong bài |
|---|---|---|
| Giao thức | Test A0x: target = **session 2 của A0x** (không nhãn khi train, đồng thời là tập test); source = **session 1 của 8 subject còn lại** | II-E |
| Dữ liệu | 22 kênh EEG (bỏ 3 EOG), 250 Hz, toàn bộ trial, cắt 2–6 s | II-A, II-E |
| Tiền xử lý | Butterworth **bậc 4, 8–32 Hz**, rồi **exponential moving standardization** | II-B |
| Feature extractor | **CBAM** (W0, W1 ∈ R^{1×1}, conv không gian 1×1) → **6 dải con** 8–12, …, 28–32 Hz → **depthwise Conv2D (C,1)**, d bộ lọc mỗi dải → **variance layer** (cửa sổ không chồng lấp, w = 1) → **FC 64** | II-D-1, Table I |
| Domain discriminator | Critic Wasserstein + gradient penalty trên {h_s, h_t, điểm nội suy}; critic được tối ưu trước n bước | II-D-2, Eq. 5–7, Algorithm 1 |
| Classifier | **2 lớp FC** + softmax, chỉ học từ nhãn source | II-D-3, Eq. 8 |
| Cập nhật | θ_c theo L_cls; θ_f theo L_cls + L_wd (λ = 0 ở bước min) | Eq. 9, Algorithm 1 |
| Optimizer | Adam, **lr = 0.0005**, batch 64 | II-E |
| Báo cáo | Mean ± **sample std (ddof = 1)**, trung bình Cohen's κ từng subject | Table II |

Số tham chiếu (Table II): **77.60 ± 10.85 %, κ = 0.6951**. Std của bài là sample std: tính lại từ 9 subject
cho đúng 10.85. Còn (0.776 − 0.25)/0.75 = 0.7013 ≠ 0.6951, nên κ của bài là trung bình Cohen's κ từng subject.

## Giả định (bài không nêu)
1. Lọc **nhân quả**; EMS chạy trên tín hiệu liên tục từng run (đơn vị µV) với tham số mặc định của
   braindecode (factor_new = 1e-3, init_block_size = 1000, eps = 1e-4). `--zero_phase` để lọc hai chiều.
2. Bộ lọc dải con: FIR cố định 251 hệ số (firwin). Số bộ lọc d = 16, có BatchNorm sau depthwise conv.
3. Lớp ẩn classifier 64 (ELU, dropout 0.25), critic MLP 2 lớp ẩn 64 (LeakyReLU).
4. n_critic = 5, hệ số gradient penalty = 10, lr của critic = 0.0005, μ = 1 (Algorithm 1).
5. **Số epoch và tiêu chí dừng**: bài chỉ nói "until converged". Script chạy 200 epoch và in cả
   *last-epoch* (trung thực) và *best-epoch* (lạc quan, vì chọn epoch theo chính tập test).
6. CBAM hiểu theo đúng ký hiệu của bài: input là một feature map 1×C×T, nên attention kênh là một hệ số
   chung và attention không gian cho ra bản đồ C×T.

## Lưu ý khi so sánh
- Kích thước cửa sổ variance (w = 1) được bài chọn là giá trị cho accuracy cao nhất trong Fig. 5, đo trên
  chính session đánh giá; bài không mô tả tập validation riêng. Cửa sổ 2.0 chỉ đạt 69.83 %.
- Bảng so sánh của bài lấy số từ các bài khác với giao thức khác nhau (ví dụ C2CM, EEGNet, ConvNet là kết
  quả theo từng subject).
