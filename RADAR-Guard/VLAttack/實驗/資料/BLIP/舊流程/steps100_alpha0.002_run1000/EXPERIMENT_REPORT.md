# VLAttack on BLIP-VQA 與 TC-VAD 兩階段防禦實驗報告

> **合併版（實驗一 + 實驗二）**：[`../../../../成果報告/2026-09-19_COMBINED_EXPERIMENT_REPORT.md`](../../../../../文件/成果報告/2026-09-19_COMBINED_EXPERIMENT_REPORT.md)

實驗目錄：`BLIP_attack/output/VQA/steps100_alpha0.002_run1000/`
執行環境：RTX 5060 (sm_120)、`vlattack_gpu` (Python 3.10 + PyTorch 2.11.0+cu128)
完成時間：2026-09-13

---

## 摘要

本實驗在 VQAv2 上對 BLIP-VQA 施行 1000 張規模的 VLAttack 黑盒攻擊（ASR 46.20%），
並評估以 TC-VAD（caption 語意離散度）為基礎的兩階段偵測與高斯去雜訊緩解流程。

三項主要發現：

1. **原設計的 Stage 2（高斯模糊後以 caption-SVM 重評風險）不僅無效，而且有害。**
   端到端 Youden J 從 0.232 降至 0.217。根因是機制矛盾：模糊會消除偵測所依賴的差異。
2. **改以「答案翻轉」作為 Stage 2 判準，將 Youden J 由 0.232 提升至 0.387（+67%）**，
   且機制更簡單（不需訓練、不需 caption 模型）。
3. **偵測與緩解的最佳參數不同**（θ_det = σ1.5 vs θ\* = σ0.8），證實兩目標須拆開最佳化。

此外，Stage 1 的 AUROC 在樣本量放大 3.3 倍後幾乎不變（0.6695 → 0.6655），
顯示 0.67 是該方法的能力上限，而非資料量不足所致。

---

## 1. 實驗設定

### 1.1 攻擊

| 項目 | 設定 |
|------|------|
| 攻擊方法 | VLAttack（NeurIPS 2023），`method='BSA'` |
| 白盒梯度來源 | BLIP pretrain (`model_base.pth`) |
| 黑盒攻擊目標 | BLIP VQA (`model_vqa.pth`) |
| 最佳化 | PGD，L∞ budget 0.125，steps 100，alpha 0.002 |
| 影像尺寸 | 480 × 480 |
| 推論方式 | rank，`k_test=128`，候選答案 3129 個 |
| 樣本數 | 1000（皆為模型原本答對的樣本） |

### 1.2 TC-VAD 偵測器

| 階段 | 機制 |
|------|------|
| Stage 1 | 每張圖做 k=10 次隨機 ±5° 旋轉 → BLIP 產 caption → MiniLM 句向量 → 4 維角距離特徵（`d_center`、`d_iqr`、`d_tail`、`d_q95`）→ SVM RBF（GroupKFold OOF）→ risk score |
| 風險分段 | low < 0.3；0.3 ≤ mid < 0.7；high ≥ 0.7 |
| 路由 | low 放行、high 攔下、**mid 交給 Stage 2** |

樣本規模：1000 張圖 × (乾淨 + 對抗) = **2000 列**。

---

## 2. 攻擊結果

| 指標 | 數值 |
|------|------|
| 攻擊成功數 | **462 / 1000** |
| 攻擊成功率 (ASR) | **46.20%** |
| 平均 SSIM（原圖 vs 對抗圖） | **0.8003** |

### 2.1 各防禦下的殘存 ASR

分母為原本攻擊成功的 462 筆。

| 防禦手段 | 殘存攻擊數 | 殘存 ASR |
|----------|-----------|----------|
| mild resize (0.9–1.0) | 344 / 462 | 74.46% |
| strong resize (0.5–0.7) | 206 / 462 | 44.59% |
| gaussian 3×3 | 264 / 462 | 57.14% |
| **gaussian 5×5** | **171 / 462** | **37.01%** |
| resize + gaussian | 191 / 462 | 41.34% |

單一防禦中 **gaussian 5×5 最有效**（把攻擊壓到 37%），這也是後續 SSS 與 θ_det
固定 kernel 為 5×5、僅掃 sigma 的依據。

### 2.2 樣本規模穩健性

| 樣本數 | ASR | Mean SSIM |
|--------|-----|-----------|
| 200 | 42.00% | 0.8037 |
| **1000** | **46.20%** | **0.8003** |

放大 5 倍後 ASR 略升、SSIM 幾乎不變，攻擊強度穩定。

---

## 3. Stage 1 偵測結果

| 樣本規模 | 配置 | 模型 | AUROC | Recall | Precision | F1 | 中風險率 |
|---------|------|------|-------|--------|-----------|-----|---------|
| 300 對 | B2 | LogisticRegression | 0.6384 | 0.5290 | 0.5820 | 0.5542 | 94.20% |
| 300 對 | P1 | SVC_RBF | **0.6695** | 0.5500 | 0.6250 | 0.5845 | 43.50% |
| 1000 對 | B2 | LogisticRegression | 0.6384 | 0.5290 | 0.5820 | 0.5542 | 94.20% |
| 1000 對 | P1 | SVC_RBF | **0.6655** | 0.4910 | 0.6865 | 0.5723 | 39.30% |

風險分段（1000 對、P1）：low 868、**mid 786**、high 346。

### 關鍵觀察

**AUROC 未隨資料量改善**（0.6695 → 0.6655）。樣本放大 3.3 倍後可分性完全沒動，
說明 0.67 是「旋轉探針 + caption 語意離散度」的**方法上限**，不是訓練資料不足。

Precision 由 0.6250 升至 0.6865、中風險率由 43.5% 降至 39.3%，代表大樣本讓
決策邊界更穩定，但**天花板未被突破**。

特徵位移方向 `Δμ = [0.0325, 0.0096, 0.0160, 0.0539]`（300 對）四維皆為正，
方向符合假設：對抗圖在旋轉探針下的 caption 離散度確實較高。

---

## 4. SSS 去雜訊緩解模組

### 4.1 定義

$$SSS(\hat{x}) = \bigl[\Delta ASR \cdot IQP \cdot SP\bigr]^{1/3}$$

| 因子 | 定義 |
|------|------|
| ΔASR | 去雜訊後攻擊成功率下降幅度 |
| IQP | 影像品質保留度 = geo-mean(SSIM, 1−LPIPS, LapVar 比值, 1−BRISQUE/100) |
| SP | 語意保留度 = geo-mean(CLIP 相似度, MSS caption 相似度) |

基準為**未去雜訊的對抗輸入**（部署時可得）；對乾淨圖的版本另記為 `*_oracle` 上界。

### 4.2 θ 掃描結果（1000 對，中風險池 786 筆）

| σ | ΔASR | IQP | SP | **SSS** | 乾淨圖答案改變率 |
|---|------|-----|-----|---------|-----------------|
| 0.3 | 0.018 | 0.900 | 0.984 | 0.253 | 0.51% |
| 0.5 | 0.073 | 0.734 | 0.961 | 0.372 | 2.53% |
| **0.8** | **0.479** | 0.460 | 0.901 | **0.583** | 2.78% |
| 1.0 | 0.570 | 0.375 | 0.879 | 0.573 | 4.05% |
| 1.5 | 0.697 | 0.293 | 0.852 | 0.558 | 5.32% |

**θ\* = 5×5, σ = 0.8**，SSS = 0.5834。

### 4.3 θ\* 的穩健性

| 樣本規模 | θ\* | SSS | ΔASR | IQP | SP |
|---------|-----|-----|------|-----|-----|
| 300 對 | 5×5, σ=0.8 | 0.5998 | 0.510 | 0.462 | 0.916 |
| 1000 對 | 5×5, σ=0.8 | 0.5834 | 0.479 | 0.460 | 0.901 |

兩種規模都選中 σ=0.8，各項因子幾乎重合，**SSS 的參數選擇是穩健的**。

### 4.4 逐張自適應 θ（消融）

per-sample ΔASR = **0.7333**（高於 set-level 的 0.479），但 σ 分布高度集中在 0.3
（662 / 786），因為多數樣本本就未被騙、SSS 退化為僅最大化 IQP·SP。
此設定需 oracle 標籤，不可部署，僅作為上界參考。

---

## 5. Stage 2 設計：失敗與修正

### 5.1 原設計（caption-SVM 重評）— 不採用

流程：中風險樣本 → 高斯模糊（θ\*）→ 重抽 4 維特徵 → 原 SVM 重評風險。

結果（1000 對）：

| 指標 | Stage 1 單獨 | 加 Stage 2 |
|------|-------------|-----------|
| TPR | 0.289 | 0.328 |
| FPR | 0.057 | 0.111 |
| **TPR − FPR** | **0.232** | **0.217** |

**唯一讓效果變差的組合。** TPR 多 3.9pp，FPR 卻多 5.4pp。

### 5.2 失敗根因：機制矛盾

中風險池上的 AUROC 診斷（300 對）：

```
stage1 risk              AUROC = 0.5244
stage2 risk (absolute)   AUROC = 0.4653   <- 低於隨機
乾淨圖平均風險 0.4810  vs  對抗圖 0.4628   <- 方向相反
```

高斯模糊對兩類影像的作用相反：

- **對抗圖**被模糊 → 擾動被洗掉 → caption 趨穩 → 風險**下降**
- **乾淨圖**被模糊 → 單純劣化 → caption 變亂 → 風險**上升**

Stage 1 用旋轉是**放大**類間差異的探針，模糊則是**消除**差異的處理。
將消除差異的操作接在偵測鏈上，本質是自我抵消——**淨化越成功，偵測訊號越消失**。
這不是參數問題，任何 σ 都無法修正。

### 5.3 修正設計：答案翻轉偵測器

```
pred_before = VQA(x, question)
pred_after  = VQA(blur(x, θ_det), question)
is_adversarial  ⟺  pred_after ≠ pred_before
```

**原理**：乾淨圖的正確答案由**影像真實內容**支撐，輕度模糊不改變「照片裡有什麼」，
答案穩定。對抗圖的錯誤答案由**擾動**支撐，擾動一被模糊洗掉，答案便塌回正確解。

**關鍵轉念**：此設計把「淨化是否發生」本身當成證據。
同一個物理現象，原設計視為訊號流失，新設計視為訊號本身。

---

## 6. θ_det 掃描（偵測目標，1000 對）

目標函數為 Youden J = TPR − FPR，全部 786 筆中風險樣本（395 乾淨 + 391 對抗）。

| σ | TPR | FPR | **Youden J** | 僅看被騙樣本的 TPR |
|---|-----|-----|--------------|-------------------|
| 0.3 | 0.020 | 0.005 | 0.015 | 0.036 |
| 0.5 | 0.079 | 0.025 | 0.054 | 0.133 |
| 0.8 | 0.325 | 0.028 | 0.297 | 0.612 |
| 1.0 | 0.366 | 0.041 | 0.325 | 0.739 |
| **1.5** | **0.407** | 0.053 | **0.353** | **0.830** |

**θ_det = 5×5, σ = 1.5**

### 兩目標的最佳參數不同

| 目標 | 參數 | 理由 |
|------|------|------|
| 緩解 | θ\* = σ **0.8** | 需交付可用影像，模糊過強會毀畫質（IQP 隨 σ 快速下降） |
| 偵測 | θ_det = σ **1.5** | 模糊圖僅為探針、不交付，畫質無關，可放心加強以提高翻轉率 |

300 對時 θ_det = σ1.0，1000 對時移至 σ1.5，與 θ\*（恆為 σ0.8）差距拉大，
**直接實證兩目標必須拆開最佳化**。

### 運作方式

先用 θ_det 當探針做偵測判斷；判定為攻擊後，再用 θ\* 產出實際交付的淨化影像。

---

## 7. 多 σ 翻轉計數（最佳方案）

將「5 個 σ 中有幾個會使答案翻轉」作為 0–5 的連續分數。

```
AUROC = 0.7035
best threshold >= 1   TPR = 0.4731   FPR = 0.0759   J = 0.3972
```

**AUROC 0.7035 超越 Stage 1 整套 caption + SVM 的 0.6655**，
但完全不需訓練、不需 caption 模型、不需 MiniLM，僅需 5 次 VQA 推論。

---

## 7.5 緩解 ASR 彙總（θ\*，mitigation_summary）

**目標**：加高斯後，攻擊是否仍成功（`VQA(adv) ≠ original_prediction`）。

由 `output/mitigation_summary.py` 讀 `sss_per_theta_raw.csv`（σ=θ\*）、`attack_dual_defense_log.json`、`tcvad_stage1_4_academic_dataset.csv`，寫入本目錄 `mitigation_summary.csv`。

```bash
python BLIP_attack/output/mitigation_summary.py --experiment_root BLIP_attack/output/VQA/steps100_alpha0.002_run1000
```

重跑 flip 端到端（含 σ=θ\* 一行）：

```bash
python BLIP_attack/output/answer_flip_detector.py --experiment_root BLIP_attack/output/VQA/steps100_alpha0.002_run1000
```

§4.2 的 ΔASR 僅覆蓋 Mid 已騙子集；下表為 `mitigation_summary.txt`（σ=θ\*=0.8）：

| cohort | n | 糊前 ASR | 糊後 ASR | 下降 (pp) |
|--------|---|----------|----------|-----------|
| Mid 對抗 | 391 | 42.20% | 28.64% | 13.55 |
| Mid 已騙子集 | 165 | 100% | 52.12% | 47.88 |
| 1000 對抗，僅 Mid 糊 | 1000 | 46.20% | 40.90% | 5.30 |
| 462 成功子集，僅 Mid 糊 | 462 | 100% | 82.90% | 17.10 |
| **路由**（Low 放／Mid 糊／High 攔） | 1000 | **46.20%** | **25.10%** | **21.10** |

---

## 8. 端到端總對照

路由規則：low 放行、high 攔下、mid 交給各偵測器。

**偵測**指標（非緩解 ASR）。`detector_comparison.csv` 另含 `stage1 + answer-flip (sigma=0.8, theta*)` 與 §6 θ_det 對照。

### 1000 對（2000 列）

| Pipeline | TPR | FPR | **TPR − FPR** |
|----------|-----|-----|---------------|
| Stage 1 單獨（僅 high） | 0.289 | 0.057 | 0.232 |
| Stage 1 + caption-SVM Stage 2（原設計） | 0.328 | 0.111 | **0.217** |
| Stage 1 + 答案翻轉（σ=1.5） | 0.448 | 0.078 | **0.370** |
| Stage 1 + 答案翻轉（σ=0.8，θ\*） | 0.416 | 0.068 | **0.348** |
| **Stage 1 + 多 σ 翻轉計數（≥1）** | **0.474** | **0.087** | **0.387** |

### 兩種樣本規模對照

| Pipeline | 300 對 (J) | 1000 對 (J) |
|----------|-----------|------------|
| Stage 1 單獨 | 0.207 | 0.232 |
| + caption-SVM Stage 2 | 0.197 | 0.217 |
| + 答案翻轉 | 0.340 | 0.370 |
| + 多 σ 翻轉計數 | 0.343 | 0.387 |

**所有結論在 3.3 倍樣本下皆複現，且大樣本表現更佳。**

最佳方案相對 Stage 1 單獨：Youden J **+67%**，TPR **+18.5pp**，FPR 僅 **+3.0pp**。

---

## 9. 方法學結論

**直接觀測模型行為優於間接觀測語意表徵。**
Stage 1 透過 caption 語意離散度間接推測擾動存在，AUROC 上限 0.67 且不隨資料量改善。
答案翻轉直接觀測「模型決策是否依賴脆弱的高頻擾動」，以更簡單的機制達到 0.7035。

**去雜訊與偵測是對立目標，不可共用參數。**
去雜訊的成功（移除擾動）在定義上即消除了偵測所依賴的類間差異。
正確做法是拆成兩條獨立的線，各自最佳化，並把「淨化是否發生」反過來當作偵測訊號。

---

## 10. 已知限制

1. **TPR 上限受攻擊有效性制約。** 中風險池 391 筆對抗樣本中僅 165 筆真正騙到模型
   （翻轉機制僅對這些有效，其上 TPR 達 0.830）。另外 226 筆攻擊未改變答案，
   翻轉機制無從偵測。此為機制固有限制。
2. **Stage 1 仍是整體瓶頸。** 868 筆被判 low 而直接放行，其中的對抗樣本永遠不會
   進入 Stage 2。提升整體 TPR 需改善 Stage 1 或調整 low/mid 門檻。
3. **單一攻擊方法。** 僅測試 VLAttack，未涵蓋其他攻擊族系。
4. **單一探針手法。** Stage 1 僅使用 `rotate_5`，未與 `scale_crop_mild` 做跨手法泛化。

---

## 11. 檔案索引

### 攻擊產物

| 檔案 | 內容 |
|------|------|
| `attack_dual_defense_log.json` | 1000 筆完整紀錄（5.7 MB） |
| `attack_rate_summary.json` / `.txt` | ASR 與各防禦殘存率 |
| `visual_results/` | 12,000 張對照圖 |
| `run_console.log` | 攻擊完整執行紀錄 |
| `run_info.txt` | 參數與環境 |

### Stage 1 / 分析

| 檔案 | 內容 |
|------|------|
| `tcvad_stage1_4_academic_dataset.csv` | 2000 列特徵 + OOF risk + 分段 |
| `tcvad_stage1_metrics.csv` | B2 / P1 的 AUROC 等指標 |
| `academic_plots/`、`risk_tables/`、`teacher_examples/` | 圖表與交叉分析 |

### SSS 緩解

| 檔案 | 內容 |
|------|------|
| `sss_grid_search.csv` | 各 σ 的 ΔASR / IQP / SP / SSS |
| `sss_best_theta.json` | θ\* = 5×5, σ0.8 |
| `sss_per_sample.csv` | 逐張自適應 θ（消融） |
| `sss_per_theta_raw.csv` | 786 × 5 筆原始明細（含去雜訊前後預測） |

### Stage 2 與偵測器

| 檔案 | 內容 |
|------|------|
| `stage2_gaussian_results.csv` / `_summary.csv` | 原設計結果（對照組） |
| `theta_det_sweep.csv` | θ_det 掃描 |
| `theta_det.json` | θ_det = 5×5, σ1.5 + 多 σ 分數 |
| `flip_count_per_sample.csv` | 逐張翻轉計數 |
| `detector_comparison.csv` | 端到端四組對照 |

### 對照組

| 路徑 | 內容 |
|------|------|
| `results_300pairs/` | 300 對版本的完整結果（49 個檔案） |
| `../steps100_alpha0.002/` | 早期 200 張攻擊結果（未被覆蓋） |

### 程式碼

| 檔案 | 說明 |
|------|------|
| `attack_vqa.py` | 攻擊入口（新增 `--run_suffix`、`--steps`、`--alphas`、`--num_samples`） |
| `attack/VQA/adv_attack_blip_vla.py` | VLAttack 實作（修 resume、參數化樣本上限） |
| `attack/VQA/attack_rate_summary.py` | ASR 計算（新增） |
| `output/evaluate_best_stage1_new.py` | Stage 1（新增 `--limit`、修路徑） |
| `output/sss_denoise.py` | SSS 模組（新增） |
| `output/evaluate_stage2_gaussian.py` | 原 Stage 2（改 OOF 評分、吃 θ\*） |
| `output/answer_flip_detector.py` | 答案翻轉偵測器與 θ_det 掃描（新增） |
| `output/run_full_pipeline.py` | 五段一鍵執行 |

---

## 附錄：實作過程修正的關鍵問題

| 問題 | 修正 |
|------|------|
| RTX 5060 為 sm_120，PyTorch 2.4.1+cu121 不支援 | 建立 Python 3.10 + torch 2.11.0+cu128 環境 |
| `blip_pretrain.py` momentum encoder 詞表大小不符（30522 vs 30524） | 補 `text_encoder_m.resize_token_embeddings()` |
| `BLIP_Decoder` 將 ViT 回傳的 tuple 當 tensor 用 | `forward` / `generate` 皆改為解包 |
| Windows cp950 無法輸出 emoji | 全部改為 ASCII 標記 |
| Stage 2 的 SVM 在全量資料上 fit 後評同一批樣本（資料洩漏） | 改為 GroupKFold OOF 評分 |
| `load_caption_model()` 會 `os.chdir(BLIP_ROOT)`，使相對路徑失效 | 所有路徑先 `.resolve()` 為絕對路徑 |
| 攻擊 resume 時 `self.logs` 被重設 | 移除重設，改以 `finished_ids` 跳過 |

### 一個非顯而易見但關鍵的細節

攻擊的 `save_adv_image()` 使用 **ImageNet** 的 mean/std 反正規化存圖，
但 VQA dataloader 實際使用 **0.5/0.5**。因此要還原模型真正看到的張量，
必須以 ImageNet 常數逆運算，而非 0.5。此點已在 `sss_denoise.py` 中註解鎖定。

驗證結果（`validate_sss_reconstruction.py`）：重新推論存下的 PNG，
預測與攻擊當下記錄的結果 **4/4 完全一致**，確認 ΔASR 數值可信。
