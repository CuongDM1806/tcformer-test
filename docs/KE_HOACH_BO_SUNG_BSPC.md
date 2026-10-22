# Kế hoạch bổ sung thống kê, domain shift và hình minh họa cho SSM-DAN (nộp BSPC)

- Bài báo: [elsarticle-template-num.tex](elsevier_bspc_template/original/elsarticle/elsarticle-template-num.tex) / [.pdf](elsevier_bspc_template/original/elsarticle/elsarticle-template-num.pdf). Tên bài: *Domain-Adaptive Selective State-Space Learning for Cross-Subject Motor-Imagery EEG Decoding*
- Hướng dẫn tác giả: Guide for authors của BSPC (file PDF nằm ngoài repo, trong thư mục `Tcformer/`)
- Code chính: thư mục `tcformer-autodial/` (nhánh `feature/hada-no-ssm`). Các đường dẫn code bên dưới đều thuộc thư mục đó.
- Kết quả và checkpoint (seed 0):
  - `autodial_bcic2a_autodial_s01_s09_results.zip`: 76.12%, đúng số trong bài
  - `autodial_bcic2b_autodial_s01_s09_results.zip`: 84.09%
  - `autodial_zhou2016_autodial_mi8_30hz_t0p5_5p0_s01_s04_results.zip`: 84.33%
  - `ssm_selection_align_bcic2a_autodial_s01_s09_results.zip`: run 2a khác (77.08%), **không** phải số trong bài
- Ngày viết: 2026-10-22

---

## 0. Tóm tắt

Hiện bài chỉ có 4 loại bằng chứng: Acc, κ, Wilcoxon (seed 0) và ablation. Với một bài domain adaptation, reviewer BSPC gần như chắc chắn sẽ hỏi 4 câu:

1. **Có thật sự giảm được domain shift không?** Hiện chưa có đại lượng nào đo shift, cũng chưa có hình feature space.
2. **Có thật sự "learn how strongly to align" không?** Đây là đóng góp chính (α của AutoDIAL), nhưng bài chưa đưa ra giá trị α nào và chưa cho thấy α liên quan gì đến mức shift.
3. **Kết quả có ổn định không?** Chỉ chạy 1 seed, n = 9/9/4, chưa hiệu chỉnh đa so sánh, chưa có effect size và khoảng tin cậy. Riêng Zhou2016 thì không thể ra kết quả có ý nghĩa (p nhỏ nhất đạt được là 0.125).
4. **Mô hình học được gì về mặt sinh lý?** Chưa có topomap hay saliency.

**Điểm thuận lợi:** checkpoint Stage 1 của SSM-DAN đầy đủ đã được lưu cho cả 3 dataset (`checkpoints/subject_X_best-*.ckpt` trong các file zip). IM-TTA chạy sau huấn luyện (`tcformer-autodial/train_pipeline.py:163`). Vì vậy cùng một checkpoint cho ra cả **"Stage 1 UDA" (trước TTA)** lẫn **"SSM-DAN" (sau TTA)**. Khoảng 60% phân tích bên dưới **không cần train lại**.

---

## 1. Thống kê: những gì còn thiếu và cách làm

### 1.1 Làm được ngay, không cần train lại (dùng số trong Bảng 3–7)

| # | Phương pháp | Mục đích | Cách làm |
|---|---|---|---|
| S1 | **Hiệu chỉnh đa so sánh Holm–Bonferroni** | Mỗi bảng có 10 so sánh với SSM-DAN, Wilcoxon chưa hiệu chỉnh sẽ bị reviewer bắt lỗi | `statsmodels.stats.multitest.multipletests(p, method='holm')`; thêm cột p_Holm vào bảng |
| S2 | **Effect size**: matched-pairs rank-biserial r (đi cùng Wilcoxon) và Hedges' g_av | Với n = 9, p-value không nói lên mức cải thiện lớn hay nhỏ | r = (W⁺ − W⁻)/(W⁺ + W⁻) |
| S3 | **CI 95% của hiệu trung bình** (bootstrap theo subject, 10 000 lần, BCa) | Ví dụ "+11.8 điểm [7.9; 15.6]" thuyết phục hơn nhiều so với chỉ ghi p | `scipy.stats.bootstrap` |
| S4 | **Kiểm định Friedman, post-hoc Nemenyi, Critical Difference (CD) diagram** (Demšar 2006) | Đây là chuẩn khi so nhiều phương pháp trên nhiều subject. **Gộp 22 subject-fold của cả 3 dataset** thì có đủ power, giải quyết được vấn đề n = 4 của Zhou2016 | Dùng thư viện `autorank` (tự chọn test và vẽ CD) hoặc `scikit-posthocs` |
| S5 | **Bayesian signed-rank test với ROPE ±1%** (Benavoli 2017) | Trả lời được câu "không significant có phải là tương đương không" trên 2b và Zhou2016, đúng chỗ bài đang yếu | `baycomp.two_on_multiple` |
| S6 | **Mức may rủi điều chỉnh** (adjusted chance level, Müller-Putz 2008) theo từng subject | Chuẩn của BCI. Ví dụ 2a có 288 trial, 4 lớp, α = 0.05 thì ngưỡng khoảng 29.9%. Cần chứng minh mọi subject đều vượt mức may rủi | Lấy bách phân vị 95% của phân phối nhị thức `binom.ppf(0.95, n, 1/K)/n` |
| S7 | **ITR (bits/min)** và **macro-F1 / balanced accuracy** | BSPC và cộng đồng BCI quen đọc ITR. F1 thì phản ánh được lớp bị TTA làm lệch | Tính từ confusion matrix đã lưu |

Ví dụ khung code (tạo `analysis/stats_tables.py` và nhập số từ bảng):

```python
import numpy as np, scipy.stats as st
from statsmodels.stats.multitest import multipletests
from autorank import autorank, plot_stats

def rank_biserial(a, b):
    d = np.asarray(a) - np.asarray(b); d = d[d != 0]
    r = st.rankdata(np.abs(d)); return (r[d > 0].sum() - r[d < 0].sum()) / r.sum()

def boot_ci(a, b):
    res = st.bootstrap((np.asarray(a) - np.asarray(b),), np.mean,
                       n_resamples=10000, method="BCa", random_state=0)
    return res.confidence_interval

# p_holm = multipletests(p_raw, method="holm")[1]
# df: hàng = 22 subject-fold (2a+2b+Zhou), cột = phương pháp
# res = autorank(df, alpha=0.05); plot_stats(res)   # -> CD diagram
```

### 1.2 Cần dự đoán từng trial: chạy lại inference từ checkpoint, không train lại

Hiện pipeline chỉ lưu Acc/κ/confmat, **không lưu dự đoán từng trial**. Cần thêm phần này (xem mục 4).

| # | Phương pháp | Mục đích |
|---|---|---|
| S8 | **McNemar test (exact) theo từng subject**: Stage 1 UDA so với sau TTA, rồi gộp bằng Stouffer hoặc Fisher | Cho thấy TTA cải thiện có ý nghĩa *bên trong* từng subject, kể cả trên Zhou2016 (150 trial/subject) |
| S9 | **GLMM logistic mức trial**: `correct ~ method + (1 \| dataset/subject)` | Phương pháp mạnh nhất cho thiết kế lặp đo. Nó dùng hết khoảng 20 000 trial thay vì chỉ 22 con số, nên kết quả Zhou2016 có thể đạt ý nghĩa thống kê. Python dùng `pymer4`, còn R dùng `lme4::glmer` |
| S10 | **Calibration: ECE và reliability diagram**, trước và sau TTA | IM-TTA tối thiểu hóa entropy nên dễ làm mô hình *quá tự tin*. Reviewer chuyên về TTA hay hỏi điểm này. Chứng minh được TTA không làm calibration tệ đi là một điểm cộng |

**Lưu ý:** GLMM và McNemar *so với baseline* cần dự đoán từng trial của baseline. Nếu checkpoint baseline chưa được giữ lại trên Kaggle thì phải chạy lại baseline (mục 1.3). Nếu không có điều kiện, chỉ làm S8/S9 trong nội bộ các biến thể SSM-DAN là đã đủ ý.

### 1.3 Bắt buộc train lại: chạy nhiều seed (quan trọng nhất)

Chính Discussion và Conclusion của bài đã tự viết "repeated-seed experiments are still needed". Reviewer sẽ trích lại câu này. **Cần làm cho xong trước khi nộp.**

- **Chạy thêm seed:** ít nhất 5 seed (0–4) cho các cấu hình **SSM-DAN đầy đủ** (cho ra cả Stage 1 UDA và SSM-DAN), **source-only**, **w/o RA** và **w/o DA**.
- **Seed cho baseline:** tối thiểu 3 seed cho các baseline chính (TCFormer, T-TIME, RMETNet). Nếu không đủ GPU thì giữ 1 seed và ghi rõ điều đó trong Limitations.
- **Cách báo cáo:**
  - Mỗi subject lấy trung bình qua các seed. Kiểm định vẫn tính trên **subject** (đơn vị phân tích), không gộp seed×subject thành n = 45, vì như vậy là pseudo-replication.
  - Báo cáo thêm SD giữa các seed, viết dạng "76.1 ± 12.0 (subject) / ± 0.8 (seed)".
  - Có thể vẽ boxplot của 5 seed theo từng subject.
- **Chi phí ước tính (theo Bảng 8):**

  | Dataset | Thời gian mỗi seed (SSM-DAN) | Ghi chú |
  |---|---|---|
  | BCI IV-2a | khoảng 9 giờ | Vừa 1 session Kaggle (giới hạn 12 giờ), nên mỗi session chạy 1 seed |
  | BCI IV-2b | khoảng 8 giờ | |
  | Zhou2016 | khoảng 50 phút | |

- **Sensitivity analysis (khuyến nghị):** quét trọng số loss (0.1/0.5 → ×0.5, ×2) chỉ trên 2a với 1 seed. Phần này cần train lại. Còn quét `im_tta_lr` và `im_tta_steps` thì **không cần train**.

---

## 2. Chứng minh domain shift được giảm (phần quan trọng nhất)

Mục tiêu là chứng minh lần lượt từng tầng: RA giảm shift ở input, AutoDIAL + MMD + adversarial giảm shift ở feature, TTA giảm shift ở output.

### 2.1 Đo shift ở mức input (RA): không cần train

- **D1. Heatmap khoảng cách Riemann giữa các subject** (9×9) giữa các ma trận covariance trung bình, trước và sau RA. Dùng `pyriemann.utils.distance.distance_riemann`. Đặt hai heatmap cạnh nhau: sau RA, các phần tử ngoài đường chéo gần như bằng 0.
- **D2.** Biểu đồ cột thể hiện khoảng cách source → target của từng target subject, trước và sau RA, cho cả 3 dataset.

### 2.2 Đo shift ở mức feature: dùng feature `q` (64-d) và output AutoDIAL

So sánh 3 trạng thái:

| Trạng thái | Nguồn dữ liệu | Có cần train? |
|---|---|---|
| (a) source-only | Checkpoint source-only | Cần checkpoint source-only. Nếu Kaggle chưa lưu thì **phải train lại** (2a mất khoảng 7 giờ) |
| (b) Stage 1 UDA | Checkpoint hiện có, chưa chạy TTA | Không |
| (c) sau TTA | Checkpoint hiện có, có chạy TTA | Không |

Các đại lượng cần tính và báo cáo thành một bảng mới, "Domain discrepancy", theo từng dataset:

- **MMD² (multi-kernel Gaussian)** giữa feature source và target. Có thể dùng lại class `MKMMD` sẵn có trong `tcformer-autodial/models/hada_tcformer.py:177`.
- **Proxy A-distance:** d_A = 2(1 − 2ε), với ε là lỗi của một logistic regression phân biệt source và target (5-fold CV). Đây là chỉ số chuẩn, xuất phát từ bài DANN của Ganin.
- **CORAL distance:** ‖C_s − C_t‖²_F.
- **Mức căn chỉnh có điều kiện theo lớp:** cosine giữa centroid lớp k của source và của target, cùng silhouette score theo lớp trên target. Phần này chứng minh không chỉ phân phối biên được kéo lại gần mà các lớp cũng khớp nhau, tránh câu hỏi về negative transfer.

### 2.3 Hình feature space

- **F1. t-SNE/UMAP 3 cột × 2 hàng** cho 1–2 subject tiêu biểu (ví dụ S3 là subject tốt, S2 là subject khó):
  - Ba cột là source-only, Stage 1 UDA và sau TTA.
  - Hàng trên tô màu theo domain (source/target), hàng dưới tô màu theo lớp.
  - Trong hình, ghi d_A và MMD của từng panel.
- **F2. Scatter plot "shift ban đầu so với mức cải thiện"**: trục x là d_A ban đầu (source-only), trục y là ΔAcc (SSM-DAN − source-only), mỗi điểm là một subject của cả 3 dataset, kèm Spearman ρ. Hình này **giải thích kết quả "dataset dependent"**: 2b có shift nhỏ nên gain nhỏ. Nó biến điểm yếu hiện tại thành một phát hiện khoa học.

### 2.4 Chứng minh claim "learn how strongly to align" (α của AutoDIAL): không cần train

- **F3.** Biểu đồ cột hoặc heatmap α đã học, kích thước (subject × 4 nhóm feature: 3 nhánh CNN và 1 nhánh SSM) cho từng dataset. Lấy từ checkpoint bằng `model.autodial_alpha()` (`tcformer-autodial/models/hada_tcformer.py:357`).
- **F4.** Scatter plot α so với d_A ban đầu của từng subject. Nếu thấy tương quan dương (shift lớn thì α lớn), đó là **bằng chứng trực tiếp cho đóng góp chính** của bài. Nếu không có tương quan thì cần viết lại claim cho trung thực.
- Ngoài ra nên vẽ quỹ đạo α qua các epoch nếu có log. Hiện log chưa có nên cần thêm logging, và phần này **cần train lại**. Ưu tiên thấp.

### 2.5 Phân tích TTA: không cần train

- **F5. Động học TTA:** accuracy, entropy trung bình H(p) và entropy biên H(p̄) theo từng bước TTA (0 → 5 pass, có thể kéo dài tới 20 để thấy điểm bão hòa hoặc collapse).
- **F6. Phân phối lớp dự đoán** trước và sau TTA so với phân phối thật (đều nhau), chứng minh TTA không làm sụp về một lớp.
- **F7. Ngân sách dữ liệu TTA:** accuracy khi TTA chỉ dùng 10/25/50/100% số trial của Te. Hình này rất có giá trị thực tiễn cho BCI vì cho biết cần bao nhiêu trial chưa nhãn.
- **F8.** Reliability diagram trước và sau TTA (S10).

---

## 3. Hình kết quả và diễn giải sinh lý (BSPC đánh giá cao)

| Hình | Nội dung | Có cần train? |
|---|---|---|
| F9 | **Paired slope plot / dumbbell plot:** mỗi đường là một subject, đi qua Source-only → +RA → Stage 1 UDA → +TTA. Thay hoặc bổ sung cho Bảng 7, giúp thấy ngay subject nào được lợi và subject nào bị negative transfer | Không. Nếu có seed thì vẽ thêm error bar |
| F10 | **CD diagram** (S4) trên 22 fold | Không |
| F11 | **Forest plot:** hiệu trung bình và CI 95% của SSM-DAN so với từng baseline, theo từng dataset, cộng một dòng gộp (random-effects meta-analysis) | Không |
| F12 | **Confusion matrix:** source-only, SSM-DAN và hiệu giữa hai cái (ma trận Δ), cho thấy lớp nào được cải thiện (thường là feet/tongue) | Ma trận SSM-DAN đã có, source-only cần checkpoint |
| F13 | **Topomap saliency** (Integrated Gradients theo kênh) cho từng lớp, dùng `mne.viz.plot_topomap`. Kỳ vọng: tay trái tập trung ở C4, tay phải ở C3, chân ở Cz. Đây là **bằng chứng sinh lý**, rất hợp với BSPC | Không |
| F14 | **Saliency theo thời gian, hoặc độ lớn Δt của selective scan, theo thời gian trong trial**: cho thấy SSM "chọn" đúng cửa sổ ERD (khoảng 0.5–2.5 s sau cue). Cụ thể hóa được lý do dùng Mamba | Không. Cần hook để lấy Δt |
| F15 | Trọng số softmax của 4 nhóm trong grouped classifier, tức mức đóng góp của nhánh SSM so với CNN | Không |
| — | **Graphical abstract** (BSPC khuyến khích) và **Highlights** (bắt buộc, 3–5 ý, mỗi ý ≤ 85 ký tự, nộp thành file riêng có chữ "highlights" trong tên) | — |

---

## 4. Cần sửa code gì

**(1) Lưu dự đoán và feature từng trial.** Sửa `tcformer-autodial/train_pipeline.py`, ngay trước `adapt_to_target` (khoảng dòng 165) và sau `trainer.test` (khoảng dòng 168). Thêm hàm `dump_outputs(model, loader, path)` lưu vào `results/.../outputs/subject_{id}_{pre|post}_tta.npz` các trường sau:
- `logits`, `probs`, `y_true`
- `q` (feature trước AutoDIAL) và `q_hat` (sau AutoDIAL), lấy bằng `register_forward_hook` lên layer AutoDIAL
- `domain` (0 = source val, 1 = target Ta, 2 = target Te)

Gọi hàm này cho cả source-validation loader và target Ta loader, để có đủ dữ liệu tính MMD/A-distance.

**(2) Giữ checkpoint cho mọi biến thể và baseline:** source-only, no-RA, no-DA, các baseline. Nhớ tải về từ Kaggle **trước khi session hết hạn**.

**(3) Thêm script phân tích mới** trong thư mục `analysis/`, độc lập với phần train:
- `reinfer_from_ckpt.py`: load checkpoint, dựng lại datamodule (vẫn chỉ fit RA trên Ta), chạy Stage 1, chạy TTA với số bước và tỉ lệ dữ liệu thay đổi được, rồi dump `.npz`. Dùng cho F5–F8, S8–S10 và các hình của mục 2.
- `shift_metrics.py`: MMD, A-distance, CORAL, khoảng cách Riemann (D1, D2, F2, F4).
- `stats_tables.py`: Holm, effect size, CI, Friedman/CD, Bayesian, GLMM, chance level, ITR.
- `figures.py`: t-SNE/UMAP, slope plot, forest plot, topomap (MNE), reliability diagram.

**(4) Lưu ý về tính tái lập của TTA.** `tcformer-autodial/models/hada_tcformer.py:483` đặt BN ở chế độ train với `track_running_stats=False`, tức dùng thống kê theo batch. Kết quả sau TTA vì vậy phụ thuộc thứ tự và kích thước batch. Khi chạy lại inference cần:
- cố định seed và `shuffle=False`;
- **kiểm tra lại rằng tái tạo đúng 76.12% / 84.09% / 84.33%** trước khi phân tích thêm (sai lệch ≤ 0.5% là chấp nhận được);
- ghi chi tiết này vào phần Method.

**(5) Thêm `--seed` vào notebook Kaggle**, mỗi seed ghi vào một thư mục riêng. Có thể viết `summarize_seeds.py` để gộp kết quả.

---

## 5. Thứ tự làm đề xuất

| Ưu tiên | Việc | Train lại? | Thời gian ước tính |
|---|---|---|---|
| 1 | S1–S7, F9–F11 (từ số liệu có sẵn) | Không | 1 ngày |
| 2 | Code dump `.npz` và `reinfer_from_ckpt.py`, kiểm tra tái tạo 76.12 / 84.09 / 84.33 | Không | 1–2 ngày |
| 3 | α (F3, F4), D1/D2, TTA (F5–F8), McNemar (S8), saliency (F13, F14) | Không | 2–3 ngày |
| 4 | Train source-only có lưu checkpoint để có t-SNE (F1), bảng discrepancy, F2, F12 | **Có** (khoảng 7 giờ / 6 giờ / 40 phút cho 2a / 2b / Zhou) | 1–2 ngày GPU |
| 5 | Chạy 5 seed cho SSM-DAN và các ablation, tính lại toàn bộ bảng | **Có** (khoảng 17 giờ/seed cho 2a + 2b) | 1–2 tuần trên Kaggle |
| 6 | GLMM so với baseline (cần dự đoán từng trial của baseline), sensitivity của trọng số loss | **Có** | Tùy điều kiện |

---

## 6. Sửa trong bản thảo

- **Mục 3.6 Evaluation measures:** viết lại thành mục "Statistical analysis", gồm các nội dung:
  - Wilcoxon exact kèm hiệu chỉnh Holm trong từng nhóm so sánh;
  - rank-biserial r và CI 95% bootstrap theo subject;
  - Friedman + Nemenyi trên 22 subject-fold;
  - GLMM ở mức trial;
  - McNemar theo subject cho TTA;
  - mức may rủi điều chỉnh (adjusted chance level);
  - α = 0.05;
  - subject là đơn vị phân tích, seed được lấy trung bình trước khi kiểm định.
- **Mục Results mới:** thêm 4.7 "Domain discrepancy analysis" (bảng MMD/A-distance/CORAL, F1, F2), 4.8 "Learned alignment strength" (F3, F4), 4.9 "Test-time adaptation behaviour" (F5–F8) và 4.10 "Neurophysiological interpretation" (F13, F14).
- **Caption các Bảng 3–5:** còn sót cụm "—: not yet available" mà trong bảng không còn dấu "—", cần xóa.
- **Discussion:** dùng F2 và F4 để giải thích vì sao gain phụ thuộc dataset (2b có ít kênh và shift giữa các session nhỏ), thay cho câu chung chung hiện tại.

---

## 7. Checklist theo dõi

- [x] S1: Holm, đã thêm cột p_Holm vào Bảng 3–5 và sửa mục 3.6, 4.1, 4.2 trong `docs/elsevier_bspc_template/original/elsarticle/elsarticle-template-num.tex` (bản sao lưu: `.tex.bak-holm`; script: `analysis/holm_tables.py` trong repo này)
- [ ] S2–S7: effect size, CI, Friedman/CD, Bayesian, chance level, ITR/F1
- [ ] Sửa `train_pipeline.py` để dump `.npz`
- [ ] `analysis/reinfer_from_ckpt.py`, tái tạo đúng số trong bài
- [ ] F3/F4: α của AutoDIAL
- [ ] D1/D2: khoảng cách Riemann trước/sau RA
- [ ] F5–F8: phân tích TTA (động học, phân phối lớp, ngân sách dữ liệu, calibration)
- [ ] S8: McNemar theo subject
- [ ] F13/F14: topomap và saliency theo thời gian
- [ ] Train source-only có lưu checkpoint → F1, F2, F12, bảng discrepancy
- [ ] Chạy 5 seed (SSM-DAN, source-only, w/o RA, w/o DA)
- [ ] Cập nhật bản thảo (mục 3.6, mục Results 4.7–4.10, Discussion, caption bảng)
- [ ] Highlights (file riêng) và Graphical abstract
