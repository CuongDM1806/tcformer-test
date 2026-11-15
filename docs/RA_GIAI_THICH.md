# Riemannian Alignment (RA): định nghĩa, hai kiểu reference, và cơ chế toán học

Tài liệu này ghi lại RA đang dùng trong SSM-DAN, hai kiểu reference (cố định và theo session),
và lý do toán học vì sao nên áp dụng. Công thức viết dạng văn bản thường.

Ký hiệu:
- C: số kênh. T: số mẫu thời gian.
- X_i: trial i, ma trận C × T.
- P_i: ma trận hiệp phương sai của trial i, C × C, đối xứng xác định dương (SPD).
- s: chỉ số subject. k: chỉ số session.
- Aᵀ: chuyển vị. A^(1/2), A^(-1/2): căn bậc hai và căn nghịch đảo của ma trận SPD, tính qua
  phân rã trị riêng.

---

## 1. RA làm gì (code: `datamodules/base.py`)

### 1.1 Ba bước

**Bước 1. Hiệp phương sai từng trial** (`_riemannian_whitener`):

```
P_i = OAS( X_iᵀ ).covariance_          (C × C)
```

Mỗi trial cho đúng một ma trận, tính trên trục thời gian của chính trial đó. OAS là ước lượng
co rút (shrinkage) để P_i luôn khả nghịch khi số mẫu ít.

**Bước 2. Tâm (reference) = trung bình Riemann của các P_i trong tập fit** (`_riemannian_mean`):

```
R = argmin_M  Σ_i d(M, P_i)²
d(A, B) = sqrt( Σ_k log²(λ_k) ),   λ_k = trị riêng của A^(-1/2) · B · A^(-1/2)      (AIRM)
```

Cách tính là lặp điểm cố định (Karcher mean):

```
M ← (1/N) Σ P_i                               khởi tạo bằng trung bình cộng
lặp tối đa 50 lần:
    G = (1/N) Σ log( M^(-1/2) · P_i · M^(-1/2) )
    nếu ||G||_F < 1e-7: dừng
    M ← M^(1/2) · exp(G) · M^(1/2)
R = M
```

Tập fit gồm **mọi trial của một subject** (một session, hoặc các session calibration). Mọi lớp
được gộp chung, **không dùng nhãn**.

**Bước 3. Làm trắng** (`_riemannian_align_many`):

```
X_i ← R^(-1/2) · X_i        ⇒      P_i ← R^(-1/2) · P_i · R^(-1/2)
```

### 1.2 Mỗi subject có tâm riêng

RA là bước tiền xử lý, chạy một lần trước khi train. Mỗi subject (hoặc mỗi session của mỗi
subject) có tâm riêng. Không có tâm chung cho mọi subject, không có tâm theo lớp. Sau RA, trial
của mọi subject source được **gộp lại**. Khi train, batch trộn ngẫu nhiên mọi subject, và mạng
chỉ phân biệt source với target.

---

## 2. Hai kiểu reference (`preprocessing.<dataset>.ra_reference`)

| | RA cố định (`calibration`, mặc định, bài đang dùng) | RA theo session (`per_session`) |
|---|---|---|
| Tâm được fit trên | session calibration (session sớm) | **từng session riêng** |
| Session sau dùng tâm | **mượn** tâm của session calibration | tâm của **chính nó** |
| Số tâm mỗi subject | 1 | số session |
| Cần trial không nhãn của session test trước khi suy luận | không | **có** (transductive) |

Tâm được fit trên tập nào theo từng dataset:

| Dataset | RA cố định | RA theo session |
|---|---|---|
| BCI IV-2a | session T (288 trial), áp cho T và E | T và E, mỗi session một tâm |
| BCI IV-2b | session 1–3 (khoảng 400 trial), áp cho 1–5 | 5 tâm, mỗi session một tâm |
| Zhou2016 | session 1–2 (khoảng 300 trial), áp cho 1–3 | 3 tâm |
| OpenBMI | session 1 (100 trial), áp cho 1 và 2 | 2 tâm |

Vai trò của từng phần dữ liệu (OpenBMI):

| Phần dữ liệu | Dùng làm gì | RA cố định | RA theo session |
|---|---|---|---|
| Source, session 1 | train (có nhãn) | tâm session 1 | tâm session 1 |
| Source, session 2 | validation | tâm session 1 | tâm session 2 |
| Target, session 1 | DA không nhãn (Stage 1) | tâm session 1 | tâm session 1 |
| Target, session 2 | IM-TTA và test | tâm session 1 | tâm session 2 |

Tập train source giống hệt nhau ở hai kiểu. Hai kiểu chỉ khác nhau ở session 2 (validation
source và test target).

---

## 3. Cơ chế toán học: vì sao RA giảm khác biệt giữa các subject

### 3.1 Mô hình sinh dữ liệu

EEG trên da đầu là trộn tuyến tính của các nguồn trong não (dẫn truyền thể tích):

```
x(t) = A_s · z(t)
```

- z(t): hoạt động nguồn. Hiệp phương sai của z phụ thuộc lớp MI: Σ_c với lớp c.
- A_s: ma trận trộn C × C khả nghịch, **riêng cho subject s**. Nó gộp hình dạng đầu, độ dẫn điện
  của da và sọ, vị trí điện cực, trở kháng, điện cực tham chiếu và hệ số khuếch đại.

Khi đó hiệp phương sai của trial i (lớp c) của subject s là:

```
P_i = A_s · Σ_c · A_sᵀ
```

Phần mang thông tin lớp là Σ_c, chung cho mọi người ở mức lý tưởng. Phần gây khác biệt giữa
subject là A_s. Mạng train trên subject source gặp A_s khác với A_target, nên phân bố đầu vào bị
lệch (covariate shift).

### 3.2 Tính chất 1: khoảng cách AIRM bất biến với phép đồng dạng

Với mọi ma trận khả nghịch W:

```
d( W A Wᵀ, W B Wᵀ ) = d(A, B)
```

Chứng minh: trị riêng của (W A Wᵀ)^(-1) (W B Wᵀ) = W^(-ᵀ) A^(-1) B Wᵀ đồng dạng với A^(-1) B, nên
các λ_k không đổi.

Hệ quả: **hình dạng bên trong cụm trial của một subject không phụ thuộc A_s.**

```
d(P_i, P_j) = d(A_s Σ_ci A_sᵀ, A_s Σ_cj A_sᵀ) = d(Σ_ci, Σ_cj)
```

Khoảng cách giữa các trial, và vì vậy độ tách giữa các lớp, đã là đại lượng không phụ thuộc
subject. Thứ khác nhau giữa các subject chỉ là **vị trí** của cả cụm trên đa tạp SPD.

### 3.3 Tính chất 2: trung bình Riemann đồng biến với phép đồng dạng

```
mean_R( W P_i Wᵀ ) = W · mean_R( P_i ) · Wᵀ
```

Hệ quả của tính chất 1: nếu M cực tiểu Σ d(M, P_i)² thì W M Wᵀ cực tiểu Σ d(·, W P_i Wᵀ)².
Trung bình cộng không có tính chất này theo nghĩa AIRM.

Áp vào mô hình. Giả sử các lớp có tỉ lệ cân bằng, và đặt G = mean_R(Σ_c), là "tâm" chung của các
nguồn và không phụ thuộc subject. Khi đó:

```
R_s = mean_R( A_s Σ_c A_sᵀ ) = A_s · G · A_sᵀ
```

Tâm của subject s chứa đúng **A_s**, tức đúng phần gây khác biệt.

### 3.4 Làm trắng khử A_s, chỉ còn một phép quay

Sau RA:

```
P̃_i = R_s^(-1/2) · A_s Σ_c A_sᵀ · R_s^(-1/2)
```

Đặt U_s = R_s^(-1/2) · A_s · G^(1/2). Ta có:

```
U_s · U_sᵀ = R_s^(-1/2) · A_s G A_sᵀ · R_s^(-1/2) = R_s^(-1/2) · R_s · R_s^(-1/2) = I
```

nên **U_s là ma trận trực giao** (một phép quay hoặc phản xạ). Viết lại:

```
P̃_i = U_s · ( G^(-1/2) · Σ_c · G^(-1/2) ) · U_sᵀ
```

Phần ở giữa, G^(-1/2) Σ_c G^(-1/2), **giống hệt nhau cho mọi subject**. Toàn bộ khác biệt do A_s
(thang năng lượng từng kênh, tương quan giữa kênh, tham chiếu, khuếch đại) bị khử. Chỉ còn lại
một phép quay U_s. Cụ thể:

1. **Tâm của mọi subject trùng nhau tại I.** mean_R(P̃_i) = U_s · I · U_sᵀ = I.
2. **Năng lượng tổng và phổ trị riêng quanh tâm giống nhau giữa subject.** Phép quay không đổi
   trị riêng, nên độ phân tán của cụm, đo bằng d(I, P̃_i), không phụ thuộc subject.
3. **Phần còn lại, phép quay U_s, RA không khử được.** Đây là phần các thành phần DA ở mức đặc
   trưng (AutoDIAL, adversarial, MMD) và IM-TTA phải xử lý tiếp.

Đây là lý do toán học để áp dụng RA: với mô hình trộn tuyến tính, RA là phép biến đổi **không
dùng nhãn, dạng đóng, chi phí thấp** khử đúng phần lệch do A_s, mà không làm méo khoảng cách giữa
các trial, tức không làm mất thông tin lớp (tính chất 1).

### 3.5 Ảnh hưởng tới mạng

Mạng học trên tín hiệu X̃ = R_s^(-1/2) X, không học trực tiếp trên P_i. Hiệp phương sai của X̃
là P̃_i, nên mọi bộ lọc không gian mà mạng học (lớp tích chập theo kênh) gặp dữ liệu có:
- năng lượng từng kênh và tương quan giữa kênh đã chuẩn hóa về cùng một mốc I cho mọi subject;
- cùng hình dạng phân bố quanh mốc đó.

Nhờ vậy cùng một bộ lọc dùng được cho subject mới mà không phải học lại thang và tương quan riêng
của từng người.

---

## 4. Vì sao RA cố định có thể sai, và RA theo session sửa được

### 4.1 Mô hình có lệch giữa các ngày

Ma trận trộn không chỉ phụ thuộc subject mà còn phụ thuộc lần ghi: điện cực đặt lại, trở kháng,
gel, trạng thái mệt mỏi. Session k của subject s:

```
P_i = A_(s,k) · Σ_c · A_(s,k)ᵀ ,      R_(s,k) = A_(s,k) · G · A_(s,k)ᵀ
```

### 4.2 RA cố định: tâm session 1 áp cho session 2

```
P̃_i (session 2) = R_(s,1)^(-1/2) · A_(s,2) Σ_c A_(s,2)ᵀ · R_(s,1)^(-1/2)
```

Trung bình của session 2 sau RA:

```
H_s = R_(s,1)^(-1/2) · R_(s,2) · R_(s,1)^(-1/2)
```

- Nếu A_(s,2) = A_(s,1) thì H_s = I, và RA cố định đúng.
- Nếu không, H_s ≠ I. Session 2 của subject s nằm lệch khỏi I một khoảng d(I, H_s), theo hướng
  **khác nhau giữa các subject**. Mạng được train trên dữ liệu quanh I (session 1 source), nhưng
  validation (session 2 source) và test (session 2 target) lại nằm lệch.

Trị riêng của H_s đo trực tiếp độ lệch này. Ví dụ, trace(H_s)/C là tỉ lệ năng lượng trung bình
của session 2 so với session 1 sau RA.

Số đo được (`OPENBMI_THAY_PHYSIONET.md` mục 4.5):

| Dataset | Năng lượng session sau / session calibration, sau RA cố định |
|---|---|
| BCI IV-2a | 1.08–1.39 |
| OpenBMI | trung vị **1.55**; 20/54 subject > 2 |

### 4.3 RA theo session: mỗi session có tâm riêng

```
P̃_i (session k) = R_(s,k)^(-1/2) · A_(s,k) Σ_c A_(s,k)ᵀ · R_(s,k)^(-1/2)
                = U_(s,k) · ( G^(-1/2) Σ_c G^(-1/2) ) · U_(s,k)ᵀ
```

Lập luận ở mục 3.4 áp cho từng session riêng, nên mọi session của mọi subject đều có trung bình
đúng bằng I, **bất kể A_(s,2) khác A_(s,1) bao nhiêu**. Lệch giữa các ngày bị khử cùng cách với
lệch giữa các subject. Chỉ còn lại phép quay U_(s,k).

### 4.4 Bằng chứng thực nghiệm (seed 0)

| Dataset | RA cố định | RA theo session | Không RA |
|---|---|---|---|
| BCI IV-2a | 76.12 | **77.74** (+1.62, p = 0.023, 8/9 tăng) | 70.29 |
| BCI IV-2b | 84.09 | 84.10 | 83.58 |
| Zhou2016 | 84.33 | 85.33 (n = 4) | 84.00 |
| OpenBMI | 80.61 | **84.00** (+3.39, p < 0.001, 43/54 tăng) | 83.50 |

Riêng OpenBMI:
- `best_val_acc` tăng từ khoảng 0.75 lên khoảng 0.85, đúng như dự đoán từ mục 4.2, vì
  validation là session 2.
- Không IM-TTA: 83.89 so với 76.65 (+7.24).
- Lợi ích của IM-TTA giảm từ +3.96 xuống +0.11. Với RA cố định, phần lớn việc IM-TTA làm là bù
  H_s ≠ I. RA theo session đã khử phần này từ đầu vào.

---

## 5. Khi nào RA giúp nhiều, khi nào ít

Từ mục 3: lợi ích của RA lớn khi **khác biệt do A_s lớn so với khác biệt do lớp**, và **tâm R
được ước lượng chính xác**. RA giúp ít khi:

1. **Phần khác biệt của A_s đã bị khử từ trước.** Ví dụ tín hiệu lưỡng cực (2b) hoặc lọc tần số
   hẹp đã bỏ phần năng lượng biến thiên theo subject (giả thuyết H2 trong `CLAIM1_RA_BCI2A.md`).
2. **Tâm ước lượng kém.** Số trial mỗi kênh thấp: OpenBMI có 100 trial cho 62 kênh, khoảng
   1.6 trial/kênh, so với 2a có 288/22, khoảng 13. Sau CAR, ma trận còn suy biến hạng một (H3).
   Kết quả no-CAR (`OPENBMI_NOCAR.md`): bỏ CAR làm Δ_RA của RA cố định tốt lên từ −2.64 thành
   −0.19 (p = 0.018), nhưng vẫn không dương.
3. **Khác biệt giữa subject không có dạng trộn tuyến tính chung.** Ví dụ mỗi người dùng chiến
   lược MI khác nhau, tức Σ_c riêng cho từng subject (H4). Mục 3.4 cần Σ_c chung; nếu không có
   thì RA không khử được phần này.
4. **Giả định tỉ lệ lớp cân bằng bị vi phạm** trong tập fit. Khi đó R_s bị kéo về lớp chiếm đa
   số. Các tập dùng ở đây đều cân bằng lớp.
5. **Lệch giữa các ngày lớn mà dùng RA cố định** (mục 4.2). Đây là trường hợp OpenBMI.

---

## 6. Lưu ý về protocol

- Cả hai kiểu RA **không dùng nhãn** ở bất kỳ đâu, nên không có rò rỉ nhãn.
- RA cố định chỉ cần dữ liệu calibration của target (target session 1, không nhãn). Session test
  được biến đổi bằng ma trận đã có sẵn, nên phù hợp suy luận online.
- RA theo session cần **cả lô trial không nhãn của session test** để tính R_(target,2) trước khi
  phân loại. Đây là thiết lập transductive, cùng loại truy cập dữ liệu với IM-TTA. Bài đã ghi điều
  này trong phần Limitations. Muốn dùng online thì phải ước lượng R dần theo từng trial
  (running Riemannian mean).
- So sánh RA theo session với các baseline chỉ công bằng khi baseline có cùng quyền truy cập
  session test. Source-only và w/o DA trên OpenBMI hiện vẫn dùng RA cố định, cần chạy lại.

## 7. Tham chiếu code

| Việc | Vị trí |
|---|---|
| Hiệp phương sai OAS và tâm | `datamodules/base.py`, `_riemannian_whitener` |
| Trung bình Riemann (Karcher) | `datamodules/base.py`, `_riemannian_mean` |
| Làm trắng | `datamodules/base.py`, `_riemannian_align_many` |
| Chọn kiểu reference | `datamodules/base.py`, `_ra_per_session` (branch `feature/ssm-dan-per-session-ra`) |
| Áp theo dataset | `datamodules/bcic4_2a.py`, `bcic4_2b.py`, `zhou2016.py`, `openbmi.py` |
