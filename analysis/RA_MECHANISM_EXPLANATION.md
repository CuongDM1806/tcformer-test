# Vì sao Riemannian Alignment giúp BCI IV-2a nhiều hơn BCI IV-2b?

## Kết luận chính

Kết quả hiện tại **không ủng hộ** cách giải thích đơn giản rằng RA tốt trên BCI IV-2a chỉ vì 2a có nhiều kênh hơn, nhiều lớp hơn, hoặc có độ lệch miền lớn hơn.

Bằng chứng phù hợp hơn là:

> RA hữu ích khi phần biến thiên hiệp phương sai bị loại bỏ là nhiễu không liên quan đến nhãn, nhờ đó cấu trúc lớp của người thử mới trở nên tương thích hơn với các tâm lớp học từ những người thử nguồn.

Trên BCI IV-2a, full RA làm độ chính xác chẩn đoán MDM tăng từ **30.32% lên 42.52%**, tức **+12.19 điểm phần trăm**, với kiểm định Wilcoxon ghép cặp theo người thử `p = 0.0039`. Trên BCI IV-2b, cùng phép đo chỉ tăng từ **56.66% lên 57.63%**, tức **+0.97 điểm**, `p = 0.4961`.

Vì vậy, yếu tố phân biệt tốt nhất trong thí nghiệm này không phải là **RA đã làm giảm bao nhiêu domain shift**, mà là **phần shift bị loại bỏ có giúp chuyển giao thông tin lớp hay không**.

## Các con số 30.32% và 42.52% là gì?

Đây là **chronological cross-subject Riemannian MDM accuracy** — độ chính xác của một bộ phân loại chẩn đoán Minimum Distance to Mean trên các ma trận hiệp phương sai, không phải accuracy của TCFormer/SSM-DAN.

Với mỗi người thử đích `t`:

1. Dùng các session đầu của từng người thử để ước lượng reference của RA.
2. Loại người thử `t` khỏi tập nguồn theo thiết kế LOSO.
3. Từ các người thử nguồn còn lại, tính Riemannian mean cho từng lớp.
4. Biến đổi các trial ở session đánh giá của `t` theo đúng reference được ước lượng từ session đầu của chính `t`.
5. Gán nhãn mỗi trial đánh giá bằng lớp có tâm Riemannian gần nhất.

Với covariance trial `C`, tâm lớp nguồn `M_k` và khoảng cách affine-invariant

```text
d_R(A, B) = || log eig(A^{-1}B) ||_2,
```

dự đoán MDM là

```text
y_hat = argmin_k d_R(C, M_k).
```

Accuracy cuối cùng trong bảng là trung bình accuracy của các người thử. Mức ngẫu nhiên là 25% cho BCI IV-2a bốn lớp, 50% cho BCI IV-2b hai lớp và 33.33% cho Zhou ba lớp.

MDM được dùng vì nó gần như không có khả năng học biểu diễn phức tạp. Nếu accuracy của MDM tăng sau RA, ta có bằng chứng trực tiếp ở mức hình học đầu vào rằng trial đích đã tiến gần hơn tới **đúng tâm lớp nguồn**, chứ không chỉ tiến gần phân phối nguồn theo nghĩa biên.

## `p = 0.0039` được tính như thế nào?

Mỗi người thử tạo ra một cặp số:

```text
(accuracy trước RA, accuracy sau full RA).
```

Kiểm định **Wilcoxon signed-rank hai phía, exact** được thực hiện trên chín cặp của BCI IV-2a và BCI IV-2b. Do đó:

- BCI IV-2a: `p = 0.00390625` cho thấy mức tăng MDM nhất quán giữa các người thử và có ý nghĩa thống kê ở ngưỡng 0.05.
- BCI IV-2b: `p = 0.49609375` không cung cấp bằng chứng rằng full RA cải thiện MDM một cách nhất quán.

Giá trị `p` không phải xác suất giả thuyết đúng và không đo kích thước hiệu ứng. Kích thước hiệu ứng ở đây được báo riêng bằng chênh lệch điểm phần trăm.

## Thiết kế thí nghiệm

Phân tích dùng dữ liệu thật tải qua MOABB và bám theo cách chia session theo thời gian:

| Dataset | Fit reference / nguồn | Đánh giá | Kênh | Lớp | Epoch và lọc |
|---|---|---|---:|---:|---|
| BCI IV-2a | session 1 | session 2 | 22 | 4 | 2–6 s, không band-pass bổ sung |
| BCI IV-2b | session 1–3 | session 4–5 | 3 | 2 | 3–7 s, không band-pass bổ sung |
| Zhou2016 | session 1–2 | session 3 | 14 | 3 | 0.5–5 s, 8–30 Hz |

Mỗi trial được biểu diễn bằng covariance OAS. Reference của mỗi người thử là affine-invariant Riemannian mean của covariance trong các session fit. Bốn biến thể được so sánh:

- `none`: không chuẩn hóa.
- `scalar`: chỉ chuẩn hóa một hệ số biên độ tổng thể.
- `diagonal`: chuẩn hóa phương sai riêng từng kênh, không xoay không gian.
- `full`: whitening đầy đủ `C -> R^(-1/2) C R^(-1/2)`.

Ngoài cấu hình gốc, BCI IV-2a còn được giảm có kiểm soát xuống:

- 3 kênh `C3/Cz/C4` để khớp số kênh của 2b;
- 2 lớp left/right để khớp số lớp của 2b;
- đồng thời 3 kênh và 2 lớp để tạo đối chứng gần nhất với 2b.

## Các phép đo và ý nghĩa

### 1. Marginal target-to-source shift

Khoảng cách giữa Riemannian mean chung của nguồn và mean chung của session đánh giá đích. Chỉ số này không dùng nhãn. Thấp hơn nghĩa là hai phân phối biên gần nhau hơn.

### 2. Same-class target-to-source shift

Với mỗi lớp, đo khoảng cách giữa tâm lớp nguồn và tâm cùng lớp của đích, sau đó lấy trung bình qua các lớp. Chỉ số này dùng nhãn thật để phân tích hậu nghiệm; thấp hơn nghĩa là alignment có điều kiện theo lớp tốt hơn.

### 3. Source-class separation

Khoảng cách trung bình giữa các cặp tâm lớp nguồn. Cao hơn nghĩa là các lớp nguồn tách nhau rõ hơn trong không gian covariance.

### 4. Conditional quality

```text
conditional quality = source-class separation / same-class shift.
```

Tỉ số lớn hơn cho biết khoảng cách giữa các lớp lớn so với sai lệch nguồn–đích trong cùng lớp.

### 5. Cross-subject MDM accuracy

Đây là phép kiểm tra quan trọng nhất: sau alignment, một classifier chỉ dựa trên tâm lớp nguồn có nhận dạng đúng trial đích tốt hơn không? Nó chuyển câu hỏi hình học thành một đại lượng gắn trực tiếp với nhãn.

## Kết quả chính

| Cấu hình | MDM trước RA | MDM full RA | Mức tăng | Wilcoxon `p` | Giảm same-class shift |
|---|---:|---:|---:|---:|---:|
| 2a, 22 kênh, 4 lớp | 30.32% | 42.52% | **+12.19 pp** | **0.0039** | 22.33% |
| 2a, 22 kênh, left/right | 53.78% | 62.04% | +8.26 pp | 0.0742 | 20.21% |
| 2a, 3 kênh, 4 lớp | 29.86% | 39.31% | **+9.45 pp** | **0.0078** | 60.80% |
| 2a, 3 kênh, left/right | 54.40% | 60.42% | +6.02 pp | 0.0977 | 60.56% |
| 2b, 3 kênh, left/right | 56.66% | 57.63% | +0.97 pp | 0.4961 | 61.65% |
| Zhou, 14 kênh, 3 lớp | 61.83% | 70.67% | +8.83 pp | 0.2500 | 36.75% |

`pp` là percentage points, tức điểm phần trăm. Ví dụ `42.52% - 30.32% = 12.19 pp` sau khi dùng các giá trị chưa làm tròn.

## Điều gì đã bị bác bỏ?

### “RA tốt hơn trên 2a vì 2a có domain shift lớn hơn”

Không phù hợp với kết quả. Full RA giảm same-class shift trên 2b khoảng **61.65%**, lớn hơn nhiều mức **22.33%** trên 2a-22ch, nhưng MDM của 2b gần như không tăng. Giảm mạnh khoảng cách hình học không tự động tạo ra lợi ích phân loại.

### “RA tốt hơn chỉ vì 2a có 22 kênh”

Không đủ để giải thích. Khi giảm 2a xuống đúng ba kênh `C3/Cz/C4`, full RA vẫn tăng MDM:

- +9.45 pp ở bài toán bốn lớp;
- +6.02 pp ở bài toán left/right.

Ở thiết kế khớp gần nhất `2a-3ch-LR` so với `2b-3ch-LR`, gain vẫn là `+6.02 pp` so với `+0.97 pp`. Tuy nhiên, kiểm định Mann–Whitney trực tiếp giữa hai bộ gain có `p = 0.3865`, nên đây là bằng chứng định hướng, chưa phải khác biệt liên-dataset có ý nghĩa thống kê.

### “RA tốt hơn chỉ vì 2a có bốn lớp”

Không đủ để giải thích. Khi chỉ giữ left/right trên 2a, gain vẫn lớn hơn 2b. Ngược lại, việc giảm từ bốn xuống hai lớp làm hiệu ứng nhỏ đi, cho thấy số lớp có thể điều chỉnh độ lớn nhưng không phải nguyên nhân duy nhất.

## Cách diễn giải có cơ sở nhất

RA là phép chuẩn hóa không giám sát nhãn. Nó có thể loại bỏ cả:

- nuisance covariance: khác biệt do người thử, phiên đo, gain điện cực hoặc cấu trúc tương quan không phục vụ phân loại;
- covariance có liên quan đến lớp: cấu trúc thực sự giúp phân biệt motor imagery.

Trên BCI IV-2a, phần bị loại bỏ dường như chủ yếu là nuisance: sau full RA, trial của người thử đích được phân loại tốt hơn rõ rệt bằng các tâm lớp nguồn. Trên BCI IV-2b, RA vẫn đưa các miền lại gần nhau rất mạnh, nhưng phần dịch chuyển được sửa không làm trial tiến về **đúng lớp** đủ nhất quán. Nói ngắn gọn:

```text
2a: alignment hình học -> alignment có ích cho nhãn -> accuracy tăng rõ
2b: alignment hình học -> chủ yếu loại bỏ shift không quyết định nhãn -> accuracy gần như giữ nguyên
```

So sánh `full` với `scalar` củng cố cách đọc này:

- 2a-22ch-4class: full cao hơn scalar **+4.82 pp**;
- 2b-3ch-LR: full thấp hơn scalar **−1.92 pp**;
- Zhou: full thấp hơn scalar **−4.00 pp**.

Điều đó gợi ý spatial whitening đầy đủ có ích riêng trên 2a, trong khi 2b và Zhou có thể chủ yếu cần sửa scale hoặc full whitening đã loại bỏ một phần cấu trúc hữu ích. Tuy nhiên, so sánh full–scalar trên 2a có `p = 0.375`, vì vậy đây chỉ là cơ chế gợi ý, chưa thể tuyên bố đã chứng minh thống kê.

## Zhou cần được diễn giải riêng

MDM trên Zhou tăng +8.83 pp nhưng kết quả mô hình SSM-DAN trước đó chỉ tăng khoảng +0.33 pp. Hai kết quả không mâu thuẫn:

- MDM đo lợi ích hình học trực tiếp ở đầu vào.
- Mô hình sâu còn có encoder, chuẩn hóa và domain adaptation nên có thể tự học phần bất biến mà RA cung cấp.
- Zhou chỉ có bốn người thử, nên Wilcoxon có độ mạnh thống kê rất thấp (`p = 0.25`).
- Scalar normalization còn tốt hơn full RA trong MDM, gợi ý whitening đầy đủ không phải lựa chọn tối ưu cho tập này.

Vì vậy, kết luận thận trọng là RA có thể cải thiện hình học covariance trên Zhou, nhưng lợi ích đó bị dư thừa với pipeline học sâu hiện tại và chưa đủ mẫu để kết luận chắc chắn.

## Quan hệ với kết quả mô hình chính

Phân tích này là **mechanistic diagnostic**, không thay thế ablation của mô hình. Các con số 30.32%, 42.52%, 56.66% và 57.63% không nên đưa vào bảng accuracy chính như kết quả TCFormer.

Cách dùng đúng trong bài báo:

1. Dùng ablation mô hình để khẳng định RA làm accuracy của mô hình tăng bao nhiêu.
2. Dùng phân tích MDM để giải thích vì sao: trên 2a, RA làm cấu trúc lớp cross-subject có khả năng chuyển giao tốt hơn.
3. Trình bày các đối chứng 3 kênh và 2 lớp để loại trừ giải thích đơn giản do số kênh hoặc số lớp.
4. Nói rõ Zhou là exploratory vì `n = 4`.

Một câu có thể dùng trong bài:

> Although full RA reduced class-conditional covariance shift more strongly on BCI IV-2b than on BCI IV-2a, this geometric reduction translated into a significant source-centroid classification gain only on BCI IV-2a (+12.19 percentage points, paired exact Wilcoxon p = 0.0039). The result suggests that RA is beneficial when the removed subject/session covariance is label-irrelevant, rather than merely when the marginal domain shift is large.

## Giới hạn

- Đây là phân tích hậu nghiệm trên ba dataset; nó cung cấp bằng chứng cơ chế nhưng chưa chứng minh quan hệ nhân quả tuyệt đối.
- Các kiểm định trong file summary chưa hiệu chỉnh multiple comparisons. Nếu đưa toàn bộ kiểm định vào bài, cần Holm correction hoặc xác định trước một kiểm định chính.
- BCI IV-2a và 2b khác nhau ngoài số kênh/lớp: protocol, số session, số trial và quần thể ghi. Đối chứng 3-kênh/2-lớp giảm nhiễu nhưng không biến chúng thành cùng một dataset.
- MDM là classifier tối giản và không mô phỏng toàn bộ TCFormer/SSM-DAN.
- Zhou chỉ có bốn người thử nên mọi kết luận thống kê đều yếu.
- Reference RA sử dụng các session fit của người thử đích, phù hợp thiết kế hiện tại; kết quả không đại diện cho thiết lập hoàn toàn không có dữ liệu target.

## File tái lập

- `ra_mechanism_evidence.ipynb`: notebook đã chạy, có output và diễn giải.
- `ra_mechanism_core.py`: tải dữ liệu, preprocessing, tính metric, kiểm định và sinh hình.
- `ra_mechanism_outputs/ra_mechanism_subject_results.csv`: kết quả từng người thử.
- `ra_mechanism_outputs/ra_mechanism_summary.csv`: bảng tổng hợp.
- `ra_mechanism_outputs/ra_mechanism_controlled_tests.json`: các kiểm định đối chứng.
- `ra_mechanism_outputs/ra_mechanism_summary.png` và `.pdf`: hình tổng hợp.

Dữ liệu thô và epoch cache được lưu ở `/tmp/tcformer-ra-data` và `/tmp/tcformer-ra-epochs`, không được commit vào repository.

## Chạy lại

Sau khi cài các dependency trong notebook, có thể chạy trực tiếp notebook hoặc:

```bash
python analysis/ra_mechanism_core.py
```

Mặc định script ghi kết quả vào `analysis/ra_mechanism_outputs/`. Lần chạy đầu cần tải dữ liệu công khai và sẽ lâu hơn; các lần sau sử dụng cache trong `/tmp`.
