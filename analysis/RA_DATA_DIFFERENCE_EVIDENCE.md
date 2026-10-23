# Bằng chứng dữ liệu: vì sao RA có ích trên BCI IV-2a hơn BCI IV-2b?

## Luận điểm được dữ liệu ủng hộ mạnh nhất

Kết quả mới cho thấy lợi ích của RA không đến từ việc giảm domain shift nói chung. RA có ích trên BCI IV-2a vì nó cải thiện **label geometry**: trial đích được kéo về phía tâm của đúng lớp nhiều hơn phía các lớp sai. Hiệu ứng này mạnh nhất trong bài toán bốn lớp gốc của 2a và gần như biến mất trên bài toán hai lớp của 2b.

Đây là khác biệt quan trọng:

```text
Geometric alignment: các domain gần nhau hơn.
Label-selective alignment: trial gần đúng lớp hơn tương đối so với lớp sai.
```

BCI IV-2b có geometric alignment rất mạnh nhưng label-selective alignment yếu. BCI IV-2a có geometric alignment nhỏ hơn, nhưng phần alignment đó tác động đúng vào biên phân lớp.

## Đại lượng chứng minh trực tiếp

Với trial `i` có nhãn thật `y_i`, định nghĩa MDM decision margin:

```text
margin_i = min_{k != y_i} d_R(C_i, M_k) - d_R(C_i, M_{y_i})
```

trong đó:

- `C_i` là covariance của trial đích;
- `M_k` là Riemannian centroid của lớp `k`, chỉ ước lượng từ các subject nguồn;
- `d_R` là affine-invariant Riemannian distance.

Ý nghĩa:

- `margin > 0`: trial gần đúng tâm lớp hơn mọi tâm sai;
- `margin < 0`: ít nhất một lớp sai gần hơn lớp đúng;
- `margin_after - margin_before > 0`: RA cải thiện hình học liên quan trực tiếp đến nhãn.

Để so sánh giữa bài toán có số kênh/lớp khác nhau, margin gain được chuẩn hóa bằng khoảng cách trung bình giữa các tâm lớp nguồn trước RA:

```text
normalized label-margin gain
    = (margin_full_RA - margin_none) / source-class separation_before_RA
```

Do đó kết quả không chỉ phản ánh việc khoảng cách trong không gian 22 chiều có độ lớn tự nhiên cao hơn không gian ba chiều.

## Kết quả quyết định

| Cấu hình | Normalized label-margin gain |
|---|---:|
| 2a, 22 kênh, 4 lớp | **0.1517** |
| 2a, 3 kênh, 4 lớp | **0.2264** |
| 2a, 22 kênh, left/right | 0.0160 |
| 2a, 3 kênh, left/right | 0.0665 |
| 2b, 3 kênh, left/right | **0.0287** |
| Zhou, 14 kênh, 3 lớp | 0.0470 |

Bốn kiểm định chính được hiệu chỉnh Holm:

| Kiểm định | Chênh lệch | `p` raw | `p_Holm` |
|---|---:|---:|---:|
| 2a-22ch-4class so với 2b-3ch-LR | +0.1230 | 0.0195 | **0.0389** |
| 2a-3ch-4class so với 2b-3ch-LR | +0.1977 | 0.0028 | **0.0110** |
| Trong 2a-22ch: 4-class so với left/right | +0.1357 | 0.0195 | **0.0389** |
| Trong 2a-3ch: 4-class so với left/right | +0.1599 | 0.0078 | **0.0234** |

Hai so sánh giữa dataset dùng exact independent permutation test hai phía. Hai so sánh trong cùng 2a dùng exact paired Wilcoxon hai phía. Tất cả vẫn có ý nghĩa sau Holm correction.

## Suy luận từ các đối chứng

### Không phải chỉ vì 2a có 22 kênh

Khi giảm 2a xuống đúng ba kênh `C3/Cz/C4` nhưng vẫn giữ bốn lớp, label-margin gain vẫn là `0.2264`, cao hơn 2b `0.0287`; kiểm định permutation cho `p_Holm = 0.0110`.

Do đó số kênh không phải nguyên nhân cần thiết của hiệu ứng. Hai mươi hai kênh có thể bổ sung spatial information, nhưng RA vẫn tạo label-selective alignment mạnh khi chỉ còn ba kênh.

### Cấu trúc bốn lớp là yếu tố phân biệt rõ nhất

Trong chính BCI IV-2a, giữ nguyên subject và số kênh nhưng chuyển từ bốn lớp sang left/right làm hiệu ứng margin giảm mạnh:

```text
22 kênh: 0.1517 -> 0.0160, p_Holm = 0.0389
 3 kênh: 0.2264 -> 0.0665, p_Holm = 0.0234
```

Như vậy dữ liệu ủng hộ rằng RA đặc biệt hữu ích khi phải đồng bộ nhiều covariance patterns khác nhau: lateralized hand imagery cùng các pattern midline của feet và tongue. Trên 2b, chỉ có left/right và ba bipolar derivations nên không gian lớp đơn giản hơn; phần shift được RA loại bỏ ít liên quan tới biên quyết định.

### Khi khớp cả kênh và task, khác biệt 2a–2b không còn rõ

Đối chứng gần nhất là:

```text
2a: C3/Cz/C4, left/right
2b: 3 bipolar channels, left/right
```

MDM accuracy gain là `+6.02 pp` so với `+0.97 pp`, nhưng independent test không có ý nghĩa (`p = 0.3865` trong phân tích accuracy; normalized-margin interaction `p = 0.4174`).

Không có ý nghĩa không chứng minh hai dataset tương đương, nhưng nó cho thấy ta không còn bằng chứng mạnh về một “dataset identity effect” sau khi bỏ cấu trúc bốn lớp. Đây là lý do luận điểm nên tập trung vào **task/class geometry**, không nên chỉ nói 2a có tín hiệu tốt hơn.

## Class-wise evidence trên 2a

MDM recall trước và sau full RA trên cấu hình 22-kênh/bốn-lớp:

| Lớp | Trước RA | Sau RA | Thay đổi |
|---|---:|---:|---:|
| Left hand | 58.49% | 49.38% | −9.10 pp |
| Right hand | 11.11% | 33.49% | **+22.38 pp** |
| Feet | 7.56% | 25.00% | **+17.44 pp** |
| Tongue | 44.14% | 62.19% | **+18.06 pp** |

Trước RA, class geometry cross-subject bị lệch mạnh: right hand và feet gần như không chuyển giao được bằng source centroids. RA làm accuracy tổng tăng không phải bằng cách tăng đều một chút, mà bằng cách phục hồi ba lớp đang bị misalignment nặng. Left-hand recall giảm, nên RA chưa tạo class geometry hoàn hảo; nó chủ yếu làm classifier bớt thiên lệch về một số lớp.

Các kiểm định từng lớp là phân tích hậu nghiệm và chưa được dùng làm bằng chứng xác nhận độc lập. Bằng chứng chính vẫn là margin tổng, kiểm định theo subject và Holm correction ở bảng trên.

## Các giả thuyết không được dữ liệu ủng hộ

### “2b không tốt vì reference trộn screening và feedback”

BCI IV-2b dùng session 1–2 không feedback, session 3–5 có feedback; cấu hình amplifier cũng khác giữa hai regime. Thí nghiệm đã thay chỉ reference RA:

```text
reference S1–3 pooled: full-RA gain = +0.97 pp
reference S3 feedback: full-RA gain = +3.68 pp
paired improvement:              +2.71 pp
```

Hướng thay đổi phù hợp với giả thuyết, nhưng không có ý nghĩa (`p = 0.2344`, `p_Holm = 0.4688`). Vì vậy protocol mixture có thể làm RA yếu đi nhưng không đủ mạnh để dùng làm kết luận chính.

### “RA làm class-contrast direction giữa subject giống nhau hơn”

Pairwise cosine của left/right class-contrast không tăng sau RA trên cả đối chứng 2a-3ch và 2b-3ch. Interaction test không có ý nghĩa (`p = 0.7039`). Giả thuyết này bị bác bỏ theo metric đã dùng.

### “Dataset có domain shift lớn hơn sẽ hưởng lợi nhiều hơn”

2b được full RA giảm same-class shift khoảng 61.65%, lớn hơn mức 22.33% của 2a-22ch, nhưng accuracy chỉ tăng 0.97 điểm. Vì vậy magnitude của shift không dự đoán lợi ích phân loại.

## Luận điểm nên dùng trong bài

> The benefit of Riemannian alignment was determined by label selectivity rather than by the magnitude of geometric domain contraction. On BCI IV-2a, full RA increased the true-versus-nearest-wrong Riemannian decision margin by 15.2% of the original source-class separation, compared with 2.9% on BCI IV-2b (exact two-sided permutation, Holm-adjusted p = 0.0389). The effect remained when BCI IV-2a was reduced to C3/Cz/C4 (22.6%, Holm-adjusted p = 0.0110), ruling out channel count as the sole explanation. Within BCI IV-2a, the effect was significantly stronger for the original four-class task than for the left-versus-right subset at both 22 and 3 channels, indicating that RA primarily benefits the richer multi-class covariance geometry rather than binary hand imagery alone.

Phiên bản ngắn bằng tiếng Việt:

> RA tốt trên BCI IV-2a không phải vì nó kéo các domain lại gần nhau nhiều hơn, mà vì nó sửa đúng phần covariance đang làm sai lệch biên phân lớp bốn lớp. Hiệu ứng vẫn tồn tại khi 2a bị giảm xuống ba kênh, nhưng giảm mạnh khi bài toán bị rút xuống chỉ còn left/right. Ngược lại, trên 2b, RA chủ yếu giảm shift chung mà gần như không tăng khoảng cách tương đối giữa đúng lớp và lớp sai.

## Giới hạn diễn giải

- Đây là bằng chứng cơ chế ở mức covariance/MDM, không phải phép trung gian nhân quả trực tiếp cho neural network.
- So sánh bốn lớp với hai lớp thay đổi chính bài toán và số lớp sai trong margin. Việc chuẩn hóa bằng source-class separation giảm vấn đề scale nhưng không làm hai task hoàn toàn đồng nhất.
- Kết luận đúng là “task/class geometry là yếu tố được dữ liệu ủng hộ mạnh nhất”, không phải “đã chứng minh sinh lý feet/tongue là nguyên nhân duy nhất”.
- Class-wise tests là hậu nghiệm; không nên trình bày chúng như bốn hypothesis độc lập chưa hiệu chỉnh.
- Chín subject vẫn là cỡ mẫu nhỏ. Exact tests và Holm correction giúp kiểm soát suy luận nhưng không thay thế replication.

## File tái lập

- `ra_mechanism_core.py`: tính covariance, RA, LOSO MDM, margin và các kiểm định.
- `ra_mechanism_evidence.ipynb`: notebook cơ chế ban đầu.
- `ra_data_difference_evidence.ipynb`: notebook tập trung vào khác biệt dữ liệu/task.
- `ra_mechanism_outputs/ra_mechanism_subject_results.csv`: kết quả subject-level, gồm margin và recall từng lớp.
- `ra_mechanism_outputs/ra_classwise_results.csv`: bảng class-wise.
- `ra_mechanism_outputs/ra_mechanism_controlled_tests.json`: exact tests và Holm correction.
- `ra_mechanism_outputs/ra_label_selectivity.png`: hình tổng hợp label-selective effect.
- `ra_dataset_difference_outputs/`: thí nghiệm reference/session và class-contrast bị bác bỏ hoặc chỉ có bằng chứng yếu.
