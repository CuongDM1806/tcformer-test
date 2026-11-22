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
| 4 | **Đo đặc trưng H1–H4** trên đủ 4 dataset, theo subject | Không | ✅ 2a, 2b, Zhou xong 2026-11-18 (mục 6c). ⏳ OpenBMI: notebook đã push (`DATASETS = ['OpenBMI']`), chờ chạy với cache gắn vào |
| 5 | **Can thiệp proxy** cho H1–H3 (MDM + TS-LR, RA và không RA, frozen và per-session) | Không | ✅ 2a, 2b, Zhou xong 2026-11-18 (mục 6b): H1–H3 **không** giải thích được Δ_RA của 2a. ⏳ OpenBMI: bỏ CAR, tham chiếu 50 trial, 31/20/10 kênh ngẫu nhiên; notebook đã push, chờ chạy |
| 6 | Thêm OpenBMI và RA per-session vào phân tích proxy, thêm TS-LR | Không | ⚠ RA per-session + TS-LR xong; OpenBMI chạy chung với mục 4–5 ở trên |
| 7 | ~~Xác nhận bằng mạng: SSM-DAN trên 2a lọc 8–30 Hz, có RA và không RA~~ | Có | ⏸ **Hoãn**: mục 5 cho thấy lọc 8–30 Hz không làm Δ_RA giảm. Notebook vẫn có: nhánh `feature/ssm-dan-claim1-bandpass`, `notebooks/claim1_bandpass_bcic2a.ipynb` |
| 8 | *(Nên có)* H5: OpenBMI với 8 subject source, có RA và không RA | Có | ❌ |
| 9 | **2a source-only, hoisted scan, không RA, có số theo subject** (thay số 64.35 sequential trong bài; ghép với RA on 63.31 hoisted để so sạch) | Có (1 run × 9 subject) | ⏳ Notebook: nhánh `feature/full-mamba-source-only`, `kaggle_train_full_mamba_source_only_bcic2a_loso.ipynb`. Hiện chỉ có trung bình 62.27, không có số theo subject |
| 10 | **Bảng RA × DA:** source-only không RA / RA cố định / RA theo session trên 2b, Zhou, OpenBMI (ghép với `ssm_dan` và `ssm_dan-no_ra` để có 2×2) | Có (3 run × mỗi bộ) | ⚠ 2b: có `no_da-no_ra` và `no_ra` (mục 6e), thiếu `no_da` cùng đợt; Zhou, OpenBMI chưa có. Notebook: nhánh `feature/ssm-dan-source-only-ra`, `notebooks/source_only_ra_{bcic2b,zhou2016,openbmi}.ipynb` (variant `with_ra`) |
| 11 | Ô "chỉ DA, không IM-TTA" của 2a: đọc `no_tta_acc` từ output của run `ssm_dan-no_ra` đã có | Không | ❌ Cần output cũ của run đó |
| 12 | **Giả thuyết phép quay (proxy):** sau RA, đo độ lệch quay giữa subject theo số kênh (3 / 8 / 14 / 22); thêm căn chỉnh quay kiểu Procrustes (bước cuối của RPA, nhãn giả cho đích) sau RA, xem Δ_RA của MDM/TS-LR có tăng không | Không | ✅ Xong 2026-11-19 (mục 6d): rotation_share tăng theo số kênh; căn chỉnh quay không nhãn làm giảm accuracy trên 2a |
| 13 | **Giả thuyết phép quay (mạng):** source-only 2a 22 kênh với RA + căn chỉnh quay, không DA | Có (3 run: không RA / RA / RA + quay) | ✅ Đã chạy (mục 6d): RA + quay ≈ 50, RA 63.31, không RA 62.27 (RA so với không RA: +1.04, 6/9 subject, p = 0.73) |
| 15 | **Phép quay oracle (proxy):** fit phép quay của đích bằng nhãn thật của session huấn luyện (chỉ để chẩn đoán, không phải phương pháp) rồi áp cho session đánh giá. Nếu lợi ích lớn và tăng theo số kênh thì phép quay là cơ chế, thất bại ở mục 12–13 là do nhãn giả | Không | ⏳ Notebook: `analysis/ra_rotation_oracle.ipynb` (đã thử trên dữ liệu giả lập) |

**Thứ tự ưu tiên (cập nhật 2026-11-18):**
1. **9:** vừa sửa số 2a trong bài vừa cho phép so source-only ± RA sạch.
2. **4–5 OpenBMI** (CPU) và **10** (GPU): chạy song song.
3. **12:** rẻ, kiểm tra trực tiếp giả thuyết phép quay; **13** chỉ khi 12 ủng hộ.
4. **11, 8:** khi cần chặn câu hỏi của reviewer.

## 6a. Giả thuyết làm việc: RA khử phép trộn kênh nhưng để lại phép quay

Mô hình x = A_s z (z: nguồn não chung, A_s: phép trộn kênh của subject s). RA làm trắng nên A_s → R_s^(-1/2) A_s = U_s, một **phép quay** (xem `docs/RA_GIAI_THICH.md` mục 3.4). Lợi ích của RA cho mạng phụ thuộc vào:
1. **RA khử được bao nhiêu:** lớn khi nhiều kênh đơn cực, tương quan mạnh, hiệp phương sai lệch xa đẳng hướng (2a); nhỏ với 3 kênh lưỡng cực gần đẳng hướng (2b, log10 số điều kiện 0.79).
2. **Phép quay còn lại lớn đến đâu:** C(C−1)/2 bậc tự do, tức 3 với 3 kênh, 231 với 22 kênh.
3. **Tham chiếu có chính xác và ổn định qua session không:** kém ở OpenBMI (62×62 từ 100 trial, năng lượng trôi giữa session).

Khớp với: 2a 22 kênh source-only RA ≈ +1 (hoisted, 62.27 → 63.31) nhưng có DA thì +5.83 (DA sửa phần quay ở tầng đặc trưng); 2a 3 kênh source-only RA +6.87 (p = 0.02; ít bậc quay); 2b ≈ 0; OpenBMI RA cố định −2.89, theo session +0.50; RA theo session ≥ cố định ở mọi bộ.
Chưa khớp: Zhou (proxy lợi nhiều, mạng không; n = 4); 2a 22 kênh trái/phải source-only −2.55 (đang lệch scan).
**Đây là giả thuyết, chưa được chứng minh;** mục 9, 10, 12, 13 dùng để kiểm tra.

## 6b. Kết quả mục 5 (2026-11-18, `ra_interventions.ipynb`, Kaggle CPU)

Δ_RA (điểm %) = Acc(có RA) − Acc(không RA). "Đổi" = Δ_RA sau can thiệp − Δ_RA gốc, Wilcoxon ghép cặp theo subject.
MDM ở đây dùng khoảng cách log-Euclid (khác AIRM ở mục 3, nên số tuyệt đối hơi khác).

| Bộ | Can thiệp | MDM Δ frozen | MDM Δ per-session | TS-LR Δ frozen | TS-LR Δ per-session | Đổi so với gốc (MDM / TS-LR, frozen) |
|---|---|---|---|---|---|---|
| 2a | gốc | +12.11 (p=0.004) | +15.28 (p=0.008) | +8.33 (p=0.078) | +12.27 (p=0.008) | — |
| 2a | H3: tham chiếu 100 trial | +12.06 | +15.28 | +7.95 | +12.27 | −0.05 (p=1.0) / −0.39 (p=0.43) |
| 2a | H3: tham chiếu 36 trial | +11.99 | +15.28 | +7.52 | +12.27 | −0.12 (p=0.94) / −0.82 (p=0.20) |
| 2a | H1: CAR | +10.61 | +14.16 | +7.21 | +12.85 | −1.50 (p=0.36) / −1.12 (p=0.82) |
| 2a | H2: lọc 8–30 Hz | +13.66 | +17.25 | +8.18 | +13.16 | +1.54 (p=0.91) / −0.15 (p=0.91) |
| 2b | gốc | +1.42 | +5.95 (p=0.012) | +2.20 | +4.64 | — |
| Zhou | gốc (8–30 Hz) | +15.50 | +23.67 | **−7.33** | +6.17 | — |
| Zhou | H2 ngược: không lọc | +2.33 | +4.67 | +0.83 | +2.17 | −13.17 (p=0.125) / +8.17 (p=0.25) |
| Zhou | H1: CAR | +17.67 | +24.50 | −5.67 | +2.67 | +2.17 / +1.67 |

**Đọc kết quả:**
- **Không can thiệp nào làm Δ_RA của 2a giảm đáng kể.** CAR (H1) giảm nhiều nhất nhưng chỉ khoảng 1–1.5 điểm và không có ý nghĩa; lọc 8–30 Hz (H2) không giảm; ít trial cho tham chiếu (H3) gần như không đổi. Ở mức proxy, **H1, H2, H3 đều bị bác** như lời giải thích cho Δ_RA của 2a.
- **H2 bị bác cả ở chiều ngược:** Zhou không lọc không làm Δ_RA tăng; với MDM còn giảm mạnh (+15.50 → +2.33), vì bỏ lọc làm cả độ chính xác chung của Zhou sụt.
- **Zhou: hai proxy mâu thuẫn nhau** (MDM +15.50, TS-LR −7.33 với RA cố định), n = 4 nên p nhỏ nhất có thể là 0.125. Proxy không dùng được để kết luận gì về Zhou.
- **RA theo session luôn ≥ RA cố định** ở mọi bộ và mọi proxy, khớp với kết quả của mạng.
- **Hệ quả cho mục 7:** chạy SSM-DAN trên 2a đã lọc 8–30 Hz nhiều khả năng không làm Δ_RA giảm, nên không phân biệt được giả thuyết nào. Ứng viên còn lại: H4 (kiểu khác biệt giữa subject, cần kết quả mục 4) và H5 (số subject source).

## 6c. Kết quả mục 4 (2026-11-18, `ra_data_characteristics.ipynb`, Kaggle CPU)

Trung vị theo subject [IQR]. OpenBMI chưa có.

| Đặc trưng | 2a | 2b | Zhou (8–30 Hz) | Zhou (không lọc) |
|---|---|---|---|---|
| H1: \|r\| trung bình giữa kênh | **0.82** | 0.52 | 0.56 | 0.65 |
| H1: tỉ lệ năng lượng common-mode (vượt mức độc lập) | **0.82** | 0.51 | 0.54 | 0.53 |
| H2: tỉ lệ năng lượng ngoài 8–30 Hz | 0.61 | 0.64 | 0.13 | 0.90 |
| H3: trial / kênh | **13.1** | 133 | 20.9 | 20.9 |
| H3: log10 số điều kiện của tham chiếu | **3.49** | 0.79 | 2.66 | 2.37 |
| H3: độ lệch tham chiếu nửa–nửa / √C | 0.055 | 0.044 | 0.076 | 0.127 |
| H3: lệch tham chiếu session fit và session đánh giá / √C | 0.41 | 0.17 | 0.47 | 0.58 |
| H4: lệch subject / tương tác subject × lớp (theo subject) | 5.8 | 13.5 | 6.5 | 6.4 |

Phân rã phương sai tangent space (η²):

| Bộ | η² subject trước RA | η² subject sau RA cố định | subject / tương tác (trước RA) | lớp / tương tác (trước RA) |
|---|---|---|---|---|
| 2a | 0.507 | **0.190** | 42.5 | 0.74 |
| 2b | 0.571 | 0.038 | 63.0 | 0.45 |
| Zhou (8–30 Hz) | 0.409 | 0.038 | 37.9 | 3.89 |
| Zhou (không lọc) | 0.149 | 0.019 | 45.5 | 1.31 |

**Đọc kết quả:**
- **2a khác các bộ còn lại ở H1** (tương quan giữa kênh và thành phần chung cao nhất rõ rệt) **và ở số điều kiện của tham chiếu** (H3). Nhưng mục 5 cho thấy áp CAR lên 2a chỉ làm Δ_RA của proxy giảm 1–1.5 điểm (không có ý nghĩa), nên đặc trưng H1 **đi kèm** 2a nhưng không phải nguyên nhân chính.
- **H2 không phân biệt 2a với 2b:** cả hai đều không lọc, năng lượng ngoài 8–30 Hz tương đương (0.61 và 0.64).
- **H4 bị bác:** dự đoán là 2a có tỉ lệ subject / tương tác cao nhất, nhưng 2b cao hơn (63.0 so với 42.5), Zhou xấp xỉ (37.9).
- **Điểm khác lạ nhất của 2a: RA khử phần khác biệt giữa subject kém nhất.** Sau RA cố định, η² subject của 2a còn 0.190, trong khi 2b và Zhou chỉ còn khoảng 0.04. Nghĩa là trên 2a, RA chỉ khử được một phần khác biệt giữa subject, còn trên các bộ khác RA khử gần hết. Điều này **không giải thích được** vì sao RA giúp mạng nhiều nhất trên 2a.
- **Kết luận chung của mục 4 + 5:** không đặc trưng dữ liệu nào trong H1–H4 vừa riêng có ở 2a vừa điều khiển được Δ_RA. Ngoài ra, ở mức proxy RA giúp rõ cả trên Zhou (MDM +15.50) và 2b (RA theo session, MDM +5.95), trong khi mạng chỉ được lợi trên 2a. Như vậy **sự khác biệt nằm ở mức mạng**, không nằm ở dữ liệu: nhiều khả năng các lớp chuẩn hóa của mạng (z-score, AutoDIAL) đã tự làm phần việc của RA trên 2b và Zhou.

## 6d. Kết quả mục 12 và 13 (2026-11-19)

**Mục 12 (proxy, `ra_rotation.ipynb`).** `rotation_share` = phần lệch tâm lớp giữa subject sau RA cố định mà một phép quay khử được (đo bằng nhãn của mọi subject, chỉ để mô tả dữ liệu):

| Bộ / tập kênh | Số kênh | rotation_share |
|---|---|---|
| 2a C3 Cz C4 | 3 | 0.12 |
| 2a 8 kênh ngẫu nhiên | 8 | 0.34 |
| 2a 14 kênh ngẫu nhiên | 14 | 0.42 |
| 2a đủ | 22 | **0.52** |
| 2b | 3 | 0.30 |
| Zhou | 14 | 0.50 |

Lợi ích thêm của căn chỉnh quay **không nhãn** (`extra_rot` = RA + quay − RA, RA cố định; MDM / TS-LR):

| 2a 3 kênh | 2a 8 kênh | 2a 14 kênh | 2a 22 kênh | 2b | Zhou |
|---|---|---|---|---|---|
| +0.62 / −1.47 | −2.46 / **−4.32** (p=0.01) | −2.07 / −4.00 | −3.12 / −5.56 | +4.74 / +3.86 (n.s.) | −6.17 / −4.67 (n = 4) |

**Mục 13 (mạng, source-only 2a 22 kênh, `ra_rotation_bcic2a.ipynb`).** Chỉ có hình ma trận nhầm lẫn; trung bình đường chéo (độ nhạy trung bình, ≈ accuracy vì 2a cân bằng lớp) và macro-F1:

| Run | Trung bình đường chéo | macro-F1 |
|---|---|---|
| không RA (`source_only`, hoisted) | ≈ 61.3 | 58.72 |
| RA (`source_only-with_ra`) | ≈ 63.3 | 61.43 |
| RA + quay (`source_only-with_ra-ra_rotation`) | ≈ 50.0 | 47.91 |

Cần `test_acc` theo subject từ output của run để có số chính xác và kiểm định.

**Đọc kết quả:**
- **Phần 1 của giả thuyết được ủng hộ:** sau RA vẫn còn một phép quay giữa các subject, và phần lệch do quay tăng đều theo số kênh của 2a (0.12 → 0.52).
- **Phần 2 thất bại ở bước ước lượng:** căn chỉnh quay *không nhãn* làm giảm độ chính xác, giảm nhiều hơn khi nhiều kênh, ở cả proxy và mạng (−13 điểm). Lý do: phép quay của đích được fit theo nhãn giả, mà proxy chỉ đúng ~40% trên 2a 4 lớp, nên phép quay xoay theo nhãn sai. Trên 2b (2 lớp, nhãn giả tốt hơn) căn chỉnh quay lại giúp (+4–5, chưa có ý nghĩa).
- **Chưa phân biệt được:** "phép quay không phải nguyên nhân" với "phép quay là nguyên nhân nhưng không ước lượng được khi thiếu nhãn". Cần mục 15 (phép quay oracle).

## 6e. Kết quả mục 10: bảng RA × DA (2026-11-22)

Biến thể `no_da` bỏ Stage-1 DA nhưng **giữ IM-TTA** (khác source-only của bài, vốn bỏ cả IM-TTA). Seed 0, LOSO.

### BCI IV-2b (9 subject)

| Run | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | Acc | κ | ITR (bit/phút) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ssm_dan-no_da-no_ra` | 75.94 | 61.07 | 68.44 | 89.69 | 97.81 | 73.12 | 85.62 | 84.06 | 84.38 | 80.01 ± 10.72 | 0.600 ± 0.214 | 5.03 ± 3.53 |
| `ssm_dan-no_ra` | 80.00 | 68.21 | 66.56 | 89.06 | 98.44 | 83.75 | 89.69 | 88.44 | 83.75 | 83.10 ± 9.71 | 0.662 ± 0.194 | 5.94 ± 3.45 |
| Hiệu (DA khi không RA) | +4.06 | +7.14 | −1.88 | −0.63 | +0.63 | +10.63 | +4.07 | +4.38 | −0.63 | **+3.09** | | |

DA khi không có RA: +3.09 điểm, 6/9 subject tăng, Wilcoxon p = 0.098 (p_holm = 0.098).

Bảng 2×2 trên 2b (accuracy %):

| | Có DA | Không DA (giữ IM-TTA) |
|---|---|---|
| **Có RA (cố định)** | 84.09 (bài) | 83.35 (bài) |
| **Không RA** | 83.10 (run mới) / 83.58 (bài) | 80.01 (run mới) |

- **Lưu ý:** `ssm_dan-no_ra` lần này ra 83.10, khác số 83.58 trong bài. Nhiều khả năng do khác cách quét Mamba (sequential hay hoisted) hoặc khác code. Nên so các ô cùng một đợt chạy.
- **Δ_RA khi không DA** ≈ 83.35 − 80.01 = **+3.34**. **Δ_RA khi có DA** = +0.51 (bài) hoặc +0.99 (với run mới). Hai số đầu khác đợt chạy, chỉ dùng tham khảo.
- **Đọc sơ bộ:** trên 2b, RA và DA **thay thế nhau**. Bỏ một trong hai chỉ mất khoảng 0.5–1 điểm, bỏ cả hai mất khoảng 3–4 điểm. Điều này khớp với kết luận ở mục 6c: trên 2b, mạng (DA/AutoDIAL) đã tự làm phần việc của RA, nên Δ_RA của mạng đầy đủ gần 0.
- **Còn thiếu:** `no_da` (có RA) chạy cùng đợt để có Δ_RA sạch khi không DA; bảng tương tự cho Zhou và OpenBMI.

## 7. Câu kết luận dự kiến (điền khi có kết quả)

> RA improved accuracy substantially only on BCI IV-2a (+5.83 points with the fixed reference, +7.45 with the session-specific one), whereas its effect on the other datasets was at most 1.33 points and not significant. The difference was not explained by channel count, number of classes or the size of the inter-subject shift. Instead, BCI IV-2a is the only dataset with [đặc trưng tìm được, ví dụ: many monopolar channels sharing one reference / unfiltered broadband signals]; removing this property from BCI IV-2a reduced the RA gain from … to … points, and [chiều ngược lại] on [dataset] increased it from … to …. RA therefore helps when [cơ chế].
