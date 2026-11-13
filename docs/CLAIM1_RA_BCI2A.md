# Claim 1 — RA cải thiện trên BCI IV-2a nhiều hơn các bộ dữ liệu khác

> Cập nhật: 2026-11-13. Số liệu lấy từ `tab:ablation` trong `elsarticle-template-num.tex`.

## 1. Phát biểu

Riemannian Alignment (RA) tăng accuracy rõ rệt và có ý nghĩa thống kê trên BCI IV-2a, trong khi trên BCI IV-2b, Zhou2016 và OpenBMI lợi ích gần 0 hoặc âm. Điều này đúng với **cả hai** kiểu tham chiếu RA:

- **RA frozen:** tham chiếu fit trên session 1, áp nguyên cho session sau.
- **RA per-session:** mỗi session dùng tham chiếu riêng.

**Mục tiêu:** chỉ ra **đặc trưng nào của dữ liệu** làm 2a khác các bộ còn lại, và chứng minh đặc trưng đó **gây ra** lợi ích của RA, không chỉ đi kèm với nó.

## 2. Kết quả chính (đã đủ)

Định nghĩa: **Δ_RA = Acc(có RA) − Acc(không RA)**.

| Dataset | Không RA | RA frozen | **Δ frozen** | RA per-session | **Δ per-session** |
|---|---|---|---|---|---|
| **BCI IV-2a** | 70.29 | 76.12 | **+5.83** (có ý nghĩa) | 77.74 | **+7.45** |
| BCI IV-2b | 83.58 | 84.09 | +0.51 (n.s.) | 84.10 | +0.52 |
| Zhou2016 | 84.00 | 84.33 | +0.33 (n.s.) | 85.33 | +1.33 (n = 4) |
| OpenBMI | 83.50 | 80.61 | −2.89 | 84.00 | +0.50 (n.s., p = 0.61) |

Chỉ 2a có Δ_RA lớn ở cả hai kiểu RA. Các bộ còn lại có Δ_RA ≤ +1.33 và không có ý nghĩa thống kê.

## 3. Những gì đã biết

Nguồn: `analysis/RA_MECHANISM_EXPLANATION.md`. Phân tích đo MDM (bộ phân loại theo tâm lớp hiệp phương sai), RA frozen, trên 2a, 2b, Zhou.

| Cấu hình | MDM trước RA | MDM sau RA | Mức tăng | Wilcoxon p |
|---|---:|---:|---:|---:|
| **2a, 22 kênh, 4 lớp** | 30.32 | 42.52 | **+12.19** | **0.0039** |
| 2a, 3 kênh C3/Cz/C4, 4 lớp | 29.86 | 39.31 | +9.45 | 0.0078 |
| 2a, 3 kênh, trái/phải | 54.40 | 60.42 | +6.02 | 0.098 |
| 2b, 3 kênh, trái/phải | 56.66 | 57.63 | +0.97 | 0.50 |
| Zhou, 14 kênh, 3 lớp | 61.83 | 70.67 | +8.83 | 0.25 (n = 4) |

**Đã loại trừ:**
- Số kênh và số lớp: 2a rút về 3 kênh và 2 lớp vẫn được lợi hơn 2b.
- "2a lệch giữa các subject nhiều nhất": lệch ban đầu của Zhou (0.72) còn lớn hơn 2a (0.65). RA giảm lệch trên 2b (61.65%) mạnh hơn trên 2a (22.33%), nhưng 2b vẫn không được lợi.

**Vì sao MDM chưa đủ:**
- MDM chỉ cho biết RA *có* giúp ở 2a, chưa cho biết **đặc trưng dữ liệu nào** gây ra điều đó.
- Zhou mâu thuẫn: MDM tăng +8.83 nhưng mạng chỉ tăng +0.33.
- Chưa có OpenBMI, chưa có RA per-session.
- Chỉ có 4 dataset, nên tương quan giữa các dataset (n = 4) không chứng minh được nguyên nhân. **Phải can thiệp:** bật/tắt từng đặc trưng trên **cùng một dataset** rồi xem Δ_RA thay đổi thế nào.

## 4. Đặc trưng dữ liệu của 4 bộ

Lấy từ `configs/hada_tcformer.yaml`, các datamodule và tài liệu dataset của MOABB (đã xác minh 2026-11-13).

| Đặc trưng | **BCI IV-2a** | BCI IV-2b | Zhou2016 | OpenBMI |
|---|---|---|---|---|
| Số kênh | 22 | 3 | 14 | 62 |
| Kiểu đạo trình / tham chiếu | **Đơn cực, tham chiếu mastoid trái** | Lưỡng cực (C3, Cz, C4) | **Đơn cực, tham chiếu mastoid trái** | CAR trong pipeline |
| Lọc trong pipeline | **Không lọc** (dữ liệu gốc 0.5–100 Hz) | Không lọc | 8–30 Hz | 4–38 Hz |
| Số lớp | 4 | 2 | 3 | 2 |
| Trial dùng để fit tham chiếu RA (frozen) | 288 (1 session) | 400 (session 1–3: 120 + 120 + 160) | ~300 (session 1–2, mỗi session 150) | **100** (1 session) |
| Trial / kênh khi fit tham chiếu | 13 | 133 | ~21 | **1.6** |
| Số subject source | 8 | 8 | 3 | 53 |

**Điểm nổi bật:**
- 2a là bộ duy nhất **vừa nhiều kênh đơn cực có chung một tham chiếu, vừa không lọc tần số**.
- **Zhou là đối chứng tự nhiên của 2a:** cùng kiểu tham chiếu (đơn cực, mastoid trái), nhiều kênh (14), nhưng **đã lọc 8–30 Hz**, và RA gần như không giúp (+0.33). Nếu chỉ riêng tham chiếu chung (H1) là đủ thì Zhou cũng phải được lợi. Điều này nghiêng về H2 (lọc tần số), hoặc H1 chỉ có tác dụng khi đi kèm H2. Cần lưu ý Zhou chỉ có 4 subject, và MDM trên Zhou vẫn tăng +8.83.
- OpenBMI ước lượng tham chiếu 62×62 từ chỉ 100 trial, lại sau CAR (ma trận hạng C−1).

## 5. Các giả thuyết đặc trưng cần kiểm tra

Mỗi giả thuyết gồm ba phần: **đo** đặc trưng trên cả 4 bộ (không train), **can thiệp** bằng cách bật/tắt đặc trưng đó trên cùng một bộ (không train, dùng bộ phân loại proxy), và **dự đoán** nếu giả thuyết đúng.

Bộ phân loại proxy: **MDM** và **TS-LR** (chiếu hiệp phương sai lên tangent space, rồi logistic regression). TS-LR mạnh hơn MDM và gần với mô hình học hơn. Cả hai chạy trong vài phút.

### H1. Tương quan không gian giữa các kênh (kiểu tham chiếu)

**Ý tưởng:** 2a dùng 22 kênh đơn cực chung một điện cực tham chiếu, nên các kênh tương quan rất mạnh. Thành phần chung này (common mode) và tương quan do dẫn truyền thể tích (volume conduction) khác nhau theo đầu và cách đặt điện cực của từng subject. RA làm trắng toàn bộ ma trận nên khử đúng phần này. 2b (lưỡng cực) và OpenBMI (CAR) đã tự khử phần lớn tương quan chung từ trước, nên RA còn ít việc để làm.

Bằng chứng gợi ý đã có: trên 2a, RA đầy đủ tốt hơn RA chỉ chỉnh thang đo +4.82 điểm MDM; trên 2b và Zhou thì ngược lại (−1.92, −4.00).

- **Đo:** trung bình |hệ số tương quan| giữa các kênh; tỉ lệ năng lượng nằm ở trị riêng lớn nhất; số điều kiện của hiệp phương sai. Tất cả tính theo subject.
- **Can thiệp:** áp **CAR** (hoặc đạo trình lưỡng cực) cho 2a rồi tính lại Δ_RA. Chiều ngược lại: bỏ CAR ở OpenBMI.
- **Dự đoán nếu đúng:** 2a có tương quan cao nhất. Sau khi áp CAR, Δ_RA của 2a giảm mạnh. OpenBMI bỏ CAR thì Δ_RA tăng.

### H2. Nhiễu tần số thấp và băng rộng (lọc tần số)

**Ý tưởng:** 2a không lọc, nên hiệp phương sai bị chi phối bởi năng lượng < 4 Hz (trôi nền, chuyển động mắt) và > 30 Hz (cơ), vốn khác nhau theo subject và không mang thông tin MI. RA chuẩn hóa phần năng lượng này. Zhou (8–30 Hz) và OpenBMI (4–38 Hz) đã lọc bỏ nó từ trước.

- **Đo:** tỉ lệ năng lượng trong băng mu/beta (8–30 Hz) so với tổng năng lượng; độ biến thiên giữa các subject của tỉ lệ này.
- **Can thiệp:** lọc 2a 8–30 Hz rồi tính lại Δ_RA. Chiều ngược lại: chạy Zhou không lọc.
- **Dự đoán nếu đúng:** 2a có tỉ lệ mu/beta thấp nhất và biến thiên nhiều nhất. Lọc 2a làm Δ_RA giảm. Bỏ lọc Zhou làm Δ_RA tăng.

### H3. Chất lượng ước lượng tham chiếu RA

**Ý tưởng:** RA chỉ tốt khi ước lượng được tham chiếu ổn định. OpenBMI phải ước lượng ma trận 62×62 từ 100 trial (1.6 trial/kênh), lại sau CAR (ma trận suy biến), nên tham chiếu nhiễu và phép làm trắng khuếch đại nhiễu. 2a có 13 trial/kênh.

- **Đo:** độ ổn định khi chia đôi: khoảng cách Riemann giữa tham chiếu tính từ nửa trial thứ nhất và nửa thứ hai, chia √C.
- **Can thiệp:** fit tham chiếu 2a từ 100 trial ngẫu nhiên (≈ 4.5 trial/kênh) hoặc 36 trial (1.6 trial/kênh, giống OpenBMI). Chiều ngược lại: OpenBMI chỉ giữ khoảng 20 kênh vận động.
- **Dự đoán nếu đúng:** tham chiếu OpenBMI kém ổn định nhất. 2a với ít trial thì Δ_RA giảm. OpenBMI với ít kênh thì Δ_RA tăng.

### H4. Kiểu khác biệt giữa các subject

**Ý tưởng:** RA chỉ khử được phần khác biệt giữa subject có dạng **dịch chuyển trung bình**, tức cả subject bị một phép biến đổi chung. Nó không khử được phần **tương tác subject × lớp**, tức mỗi subject có mẫu phân biệt lớp riêng. RA giúp nhiều khi phần dịch chuyển trung bình lớn so với tín hiệu lớp và phần tương tác nhỏ. Đây là phiên bản chính xác hơn của giả thuyết "lệch nhiều" đã bị bác.

- **Đo:** trong tangent space, phân rã phương sai thành bốn phần: giữa subject, giữa lớp, tương tác subject × lớp, và phần dư. Báo cáo η² của từng phần cho mỗi dataset.
- **Can thiệp:** không cần; đây là phép đo thuần.
- **Dự đoán nếu đúng:** 2a có tỉ lệ (giữa subject) / (tương tác) cao nhất. Theo từng subject, tỉ lệ này tương quan với Δ_RA của mạng (Spearman, gộp khoảng 76 subject).

### H5. Số subject source (yếu tố giao thức, không phải dữ liệu)

**Ý tưởng:** OpenBMI có 53 subject source, nên mạng có thể tự học được tính bất biến giữa subject mà không cần RA.

- **Can thiệp:** OpenBMI chỉ dùng 8 subject source ngẫu nhiên. Cần train mạng.
- **Dự đoán nếu đúng:** Δ_RA của OpenBMI tăng lên khi ít source.

## 6. Việc cần làm

| # | Việc | Cần train? | Trạng thái |
|---|---|---|---|
| 1 | Bảng Δ_RA trên 4 dataset, cả 2 kiểu RA (gồm Zhou PS-RA, OpenBMI PS-RA đủ 54) | — | ✅ Xong 2026-11-13 (đã cập nhật `tab:ablation` EN + VI) |
| 2 | MDM trước/sau RA frozen trên 2a, 2b, Zhou, kèm đối chứng 3 kênh và 2 lớp | Không | ✅ Đã có (`analysis/ra_mechanism_outputs/`) |
| 3 | Xác minh các ô "(kiểm tra)" ở mục 4: tham chiếu của Zhou, số trial fit tham chiếu của 2b và Zhou | Không | ✅ Xong 2026-11-13: Zhou tham chiếu mastoid trái; 2b 400 trial; Zhou ~300 trial |
| 4 | **Đo đặc trưng H1–H4** trên đủ 4 dataset, theo subject | Không | ⏳ Notebook sẵn sàng: `analysis/ra_data_characteristics.ipynb` (Kaggle CPU), chờ kết quả |
| 5 | **Can thiệp proxy** cho H1–H3 (MDM + TS-LR, RA và không RA, frozen và per-session) | Không | ❌ |
| 6 | Thêm OpenBMI và RA per-session vào phân tích MDM có sẵn, thêm TS-LR | Không | ❌ |
| 7 | **Xác nhận bằng mạng:** lấy đặc trưng có tác động lớn nhất ở mục 5, chạy SSM-DAN trên 2a đã tắt đặc trưng đó, có RA và không RA | Có (2 run × 9 subject) | ❌ |
| 8 | *(Nên có)* H5: OpenBMI với 8 subject source, có RA và không RA | Có | ❌ |

**Thứ tự ưu tiên:**
1. **4:** biết các bộ khác nhau ở đâu. Rẻ, chỉ tính hiệp phương sai.
2. **5 (+6):** tìm đặc trưng nào thật sự điều khiển Δ_RA.
3. **7:** khẳng định kết quả trên chính mô hình của bài.
4. **8:** chỉ cần nếu muốn chặn trước câu hỏi của reviewer.

Mục 3–6 có thể gom chung một notebook Kaggle, đọc dữ liệu từ cache có sẵn.

## 7. Câu kết luận dự kiến (điền khi có kết quả)

> RA improved accuracy substantially only on BCI IV-2a (+5.83 points with the fixed reference, +7.45 with the session-specific one), whereas its effect on the other datasets was at most 1.33 points and not significant. The difference was not explained by channel count, number of classes or the size of the inter-subject shift. Instead, BCI IV-2a is the only dataset with [đặc trưng tìm được, ví dụ: many monopolar channels sharing one reference / unfiltered broadband signals]; removing this property from BCI IV-2a reduced the RA gain from … to … points, and [chiều ngược lại] on [dataset] increased it from … to …. RA therefore helps when [cơ chế].
