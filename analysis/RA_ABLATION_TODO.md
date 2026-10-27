# RA ablation: các việc cần làm

Tình trạng hiện tại: đã có bằng chứng cơ chế ở mức covariance/MDM (`RA_DATA_DIFFERENCE_EVIDENCE.md`),
nhưng **chưa đủ cho ablation study**. Còn thiếu:

1. Bằng chứng trên chính mô hình (TCFormer).
2. Bằng chứng về nguồn gốc thiên lệch dự đoán trước RA.

## Bảng đầu việc

| # | Ưu tiên | Việc cần làm | Cấu hình / nội dung | Kiểm định | Kết quả mong đợi | Công sức | Trạng thái |
|---|---|---|---|---|---|---|---|
| 1 | Bắt buộc | Ablation RA on/off trên TCFormer (LOSO, 3 seed) | 2a 22ch 4 lớp (đã có, lấy lại số); 2a 22ch LR; 2a 3ch (C3/Cz/C4) LR; 2b LR (đã có) | Wilcoxon theo subject + Holm | Mức tăng nhờ RA cùng thứ tự với margin MDM: 4 lớp ≫ LR ≈ 2b | Trung bình (chỉ chạy 2 cấu hình mới) | ☐ |
| 2 | Cao | Tỉ lệ dự đoán và confusion matrix MDM theo từng subject, trước/sau RA | 2a 22ch 4 lớp, 2b LR; dùng epoch đã cache + `ra_mechanism_core.py` | Mô tả theo subject | Chỉ ra subject nào bị dồn dự đoán về Left/Tongue trước RA và RA sửa lại (thay cho class-wise test hậu nghiệm) | Thấp (1 script nhỏ) | ☐ |
| 3 | Cao | Tương quan giữa cơ chế MDM và mô hình | Spearman: normalized margin gain (MDM) vs accuracy gain nhờ RA (TCFormer), theo subject × cấu hình | Spearman ρ, p | ρ > 0, nối phân tích cơ chế với mô hình | Rất thấp (khi có việc 1) | ☐ |
| 4 | Tùy chọn | Thêm cấu hình 2a 3ch 4 lớp cho TCFormer | Chỉ làm nếu còn thời gian; MDM đã loại trừ số kênh | Wilcoxon + Holm | Xác nhận số kênh không phải nguyên nhân chính | Trung bình | ☐ |
| 5 | Bắt buộc | Viết lại kết luận và phụ lục | Xem mục "Cách viết" bên dưới | — | Lập luận không bị reviewer bắt lỗi | Thấp | ☐ |

## Cách viết kết luận

- Nói "lợi ích RA phụ thuộc cấu trúc lớp và khác biệt covariance giữa các subject". Không nói "2a tốt hơn 2b".
- Thừa nhận: khi khớp cả kênh và task (2a 3ch LR vs 2b LR), khác biệt không có ý nghĩa thống kê
  (accuracy p = 0.14; margin p = 0.42).
- Thiên lệch về Left trước RA là lỗi vị trí do nền covariance khác nhau giữa các subject (LOSO), không phải lỗi nhãn.
- Đưa vào phụ lục như kết quả âm: giả thuyết reference S3 của 2b (p_Holm = 0.47) và class-contrast cosine (p = 0.70).
- Ghi rõ hạn chế: 9 subject; MDM chỉ là bằng chứng cơ chế; class-wise test là phân tích hậu nghiệm.
