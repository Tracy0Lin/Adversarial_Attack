# VLAttack 與 TC-VAD 兩階段防禦：實驗一 + 實驗二合併報告

**版本**：2026-09-19  
**環境**：RTX 5060 (sm_120)、Python 3.10、`vlattack_gpu`（PyTorch 2.11.0+cu128）  
**詳細分報**：  
- 實驗一（混合流量）：`../BLIP_attack/output/VQA/steps100_alpha0.002_run1000/EXPERIMENT_REPORT.md`  
- 實驗二（僅攻擊成功）：`../BLIP_attack/output/VQA/exp2_success1000/EXPERIMENT_REPORT.md`

---

## 摘要

本報告合併兩組互補實驗，評估 VLAttack 對 BLIP-VQA 的攻擊，以及以 TC-VAD（caption 語意離散度）
為 Stage 1、高斯去雜訊 SSS 為緩解、答案翻轉為 Stage 2 偵測的完整流程。

| 實驗 | 目錄 | 對抗側定義 | 要回答的問題 |
|------|------|------------|--------------|
| **實驗一** | `steps100_alpha0.002_run1000/` | 1000 筆被攻擊樣本（462 成功 + 538 失敗） | **混合流量下**防禦實際表現 |
| **實驗二** | `exp2_success1000/` | 1000 筆攻擊成功 + 對應 1000 張乾淨原圖 | 攻擊**已奏效**時防禦能抓到多少（**條件上限**） |

**共同攻擊設定**：PGD steps=100、alpha=0.002、L∞=0.125；實驗一 ASR **46.20%**（462/1000），mean SSIM **0.8003**。

**五項合併結論**：

1. **原 Stage 2（模糊後 caption-SVM 重評）在實驗一有害**（Youden J 0.232→0.217）；機制上模糊會消除 Stage 1 所依賴的類間差異。實驗二略升（0.290→0.314），仍遠不如答案翻轉。
2. **答案翻轉 Stage 2** 將端到端 Youden J 提升至實驗一 **0.370**（σ=1.5）／實驗二 **0.538**；多 σ 翻轉計數在實驗一 J=**0.387**、AUROC **0.7035**。
3. **緩解 θ\* 與偵測 θ_det 必須分開**：兩實驗皆為 θ\*=5×5、σ**0.8**，θ_det=5×5、σ**1.5**。
4. **Stage 1 AUROC**：混合流量 **0.6655**；僅成功子集 **0.7212**。實驗一的 ~0.67 部分來自「有擾動但未改答案」樣本的稀釋，而非單純資料量不足。
5. **解讀紀律**：論文或部署敘事應以**實驗一**為主表；**實驗二**用於解釋 flip 機制上限（中風險 TPR_fooled ≈ **83%** vs **80%**），不可直接當上線 TPR。

---

## 1. 問題設定與記號

### 1.1 攻擊

| 項目 | 設定 |
|------|------|
| 方法 | VLAttack（NeurIPS 2023），cleverhans PGD，`method='BSA'` |
| 白盒梯度 | BLIP pretrain `model_base.pth` |
| 黑盒目標 | BLIP VQA `model_vqa.pth`，rank 推論，`k_test=128`，3129 候選答案 |
| 影像 | 480×480 |
| 本報告主 run | steps=100，alpha=0.002 |

**攻擊成功**：對抗圖上 VQA 預測 ≠ 原圖預測（且原圖原本答對）。

### 1.2 防禦管線（兩實驗相同）

```
輸入 x + question
    → Stage 1：rotate_5 × k=10 → BLIP caption → MiniLM → 4-D 角距離
              → GroupKFold OOF SVM → risk r ∈ [0,1]
              → low (r<0.3) 放行 | high (r≥0.7) 攔下 | mid 交 Stage 2
    → Stage 2（偵測，擇一）：
         (A) caption-SVM 重評（原設計，θ* 模糊後重抽特徵）
         (B) 答案翻轉：VQA(blur(x,θ_det)) ≠ VQA(x)
         (C) 多 σ 翻轉計數：5 個 σ 中有幾次翻轉（0–5）
    → 緩解（SSS，與偵測分線）：在中風險路徑搜 θ* 最大化 SSS = (ΔASR·IQP·SP)^(1/3)
```

**端到端指標**：在 2000 列（1000 乾淨 label=0 + 1000 對抗 label=1）上，
low 不攔、high 攔、mid 由 Stage 2 決定，報 TPR、FPR、Youden J = TPR−FPR。

### 1.3 兩實驗的樣本關係

- 實驗一的 1000 筆攻擊 qid 與實驗二**不完全相同**：實驗二從擴充後的攻擊 log（最多 2,298 筆）
  中依 `attack_success` 取前 1000 筆成功樣本，再配對其 `_1_ori.png` 作為乾淨列。
- 因此實驗二的「乾淨 1000」=「成功攻擊那 1000 題的原圖」，與實驗一的「1000 題原圖」
  **題目集合不同**；比較 FPR 時不宜解讀為同一分布下的單純升降。

---

## 2. 實驗一：混合流量（部署相關）

**目錄**：`BLIP_attack/output/VQA/steps100_alpha0.002_run1000/`  
**完成**：2026-09-13

### 2.1 攻擊與單一防禦殘存 ASR

| 指標 | 值 |
|------|-----|
| 攻擊成功 | 462 / 1000（**46.20%**） |
| Mean SSIM | 0.8003 |

在 462 筆成功子集上，攻擊 log 內建防禦的殘存 ASR：

| 防禦 | 殘存 ASR |
|------|----------|
| mild resize | 74.46% |
| strong resize | 44.59% |
| gaussian 3×3 | 57.14% |
| **gaussian 5×5** | **37.01%** |
| resize + gaussian | 41.34% |

→ 後續 SSS / θ 掃描固定 **5×5 kernel、掃 σ** 的依據。

### 2.2 Stage 1（rotate_5，P1 SVM OOF）

| 指標 | 300 對 | 1000 對 |
|------|--------|---------|
| AUROC | 0.6695 | **0.6655** |
| F1 | 0.5845 | 0.5723 |
| 中風險率 | 43.50% | 39.30% |

風險分段（1000 對、2000 列）：low **868**、mid **786**、high **346**。

**解讀**：樣本量 ×3.3 後 AUROC 幾乎不變 → 在**混合標籤**（含攻擊失敗）下，
rotate_5 + caption 離散度約 **0.67 為方法上限**；Precision 上升代表邊界更穩，天花板未破。

### 2.3 SSS 緩解（中風險 786 列）

| σ | ΔASR | SSS | 備註 |
|---|------|-----|------|
| 0.3 | 0.018 | 0.253 | |
| 0.5 | 0.073 | 0.372 | |
| **0.8** | **0.479** | **0.583** | **θ\*** |
| 1.0 | 0.570 | 0.573 | |
| 1.5 | 0.697 | 0.558 | 偵測向 σ，非 SSS 最優 |

### 2.4 θ_det 與中風險翻轉（786 列：391 對抗含 165 成功 + 395 乾淨）

| σ | TPR | FPR | J | TPR（僅被騙子集） |
|---|-----|-----|---|-------------------|
| 0.8 | 0.325 | 0.028 | 0.297 | 0.612 |
| 1.0 | 0.366 | 0.041 | 0.325 | 0.739 |
| **1.5** | **0.407** | 0.053 | **0.353** | **0.830** |

多 σ 翻轉計數：AUROC **0.7035**，best ≥1 次翻轉 → J **0.3972**。

### 2.5 端到端（實驗一，n=2000）

| Pipeline | TPR | FPR | **J** |
|----------|-----|-----|-------|
| Stage 1 僅 high | 0.289 | 0.057 | **0.232** |
| + caption-SVM Stage 2 | 0.328 | 0.111 | 0.217 |
| + 答案翻轉 σ=1.5 | 0.448 | 0.078 | **0.370** |
| + flip_count ≥ 1 | 0.474 | 0.087 | **0.387** |

**相對 Stage 1 單獨**：最佳 flip 方案 J **+67%**，TPR **+18.5 pp**，FPR **+3.0 pp**。

---

## 3. 實驗二：僅攻擊成功（機制上限）

**目錄**：`BLIP_attack/output/VQA/exp2_success1000/`  
**完成**：2026-09-15  
**資料**：1000×`attack_success` + 1000 對應 ori；Stage 1 共 **2000 列**；mean SSIM **0.8016**。

### 3.1 Stage 1

| 指標 | 實驗一 | 實驗二 |
|------|--------|--------|
| P1 AUROC | 0.6655 | **0.7212** |
| P1 F1 | 0.5723 | **0.6864** |
| 中風險率 | 39.30% | 33.80% |

風險分段（實驗二）：

| | low | mid | high |
|--|-----|-----|------|
| attack | 219 | 350 | 431 |
| clean | 533 | 326 | 141 |

Stage 1 僅 high：TPR **0.431**、FPR **0.141**、J **0.290**。

### 3.2 SSS（中風險 676 列；對抗側 ASR_before = 1.0）

**θ\* 仍為 σ=0.8**，SSS **0.577**（實驗一 0.583）。ΔASR 在成功子集上定義更「飽和」，
但最優 σ 不變。

### 3.3 θ_det（中風險 676 列：350 對抗皆成功 + 326 乾淨）

| σ | TPR | FPR | J |
|---|-----|-----|---|
| 0.8 | 0.617 | 0.055 | 0.562 |
| 1.0 | 0.729 | 0.080 | 0.649 |
| **1.5** | **0.797** | 0.095 | **0.702** |

多 σ 翻轉 AUROC **0.8689**（實驗一 0.7035）。

### 3.4 端到端（實驗二，n=2000）

| Pipeline | TPR | FPR | **J** |
|----------|-----|-----|-------|
| Stage 1 僅 high | 0.431 | 0.141 | 0.290 |
| + caption-SVM | 0.504 | 0.190 | 0.314 |
| + 答案翻轉 σ=1.5 | **0.710** | 0.172 | **0.538** |
| + flip_count ≥ 1 | 0.723 | 0.186 | 0.537 |

---

## 4. 合併對照表（論文主表建議）

### 4.1 端到端 Youden J

| Pipeline | 實驗一 J | 實驗二 J | ΔJ |
|----------|----------|----------|-----|
| Stage 1 僅 high | 0.232 | 0.290 | +0.058 |
| + caption-SVM | 0.217 | 0.314 | +0.097 |
| + 答案翻轉 σ=1.5 | **0.370** | **0.538** | **+0.168** |
| + flip_count ≥ 1 | **0.387** | 0.537 | +0.150 |

### 4.2 中風險池內答案翻轉（θ_det=σ1.5）

| | 實驗一 | 實驗二 |
|--|--------|--------|
| 中風險 TPR（全對抗列） | 0.407 | 0.797 |
| TPR（僅 attack_success） | **0.830** | **0.797** |
| 多 σ AUROC | 0.7035 | 0.8689 |

**關鍵對齊**：實驗一在「被騙子集」上 mid 翻轉 TPR 已達 **83%**；實驗二端到端 TPR **71%**
低於 mid 池 **80%**，因 Stage 1 將部分成功攻擊判為 **high 直接攔** 或 **low 放行**，
整體 TPR 是路由後的混合量，不是 mid 池 TPR 本身。

### 4.3 Stage 1 與 SSS 參數穩定性

| 項目 | 實驗一 | 實驗二 | 是否一致 |
|------|--------|--------|----------|
| P1 AUROC | 0.6655 | 0.7212 | 升高（子集更「可分」） |
| θ\* (SSS) | σ 0.8 | σ 0.8 | ✓ |
| θ_det | σ 1.5 | σ 1.5 | ✓ |
| caption-SVM Stage 2 是否有幫助 | 否（J↓） | 略（J↑） | 機制結論仍：不如 flip |

---

## 5. 機制說明（合併敘事）

### 5.1 為何 caption-SVM Stage 2 在實驗一失敗

Stage 1 探針（旋轉）**放大** caption 離散度差異；高斯模糊**縮小**差異。
模糊後重跑 SVM：對抗圖風險下降、乾淨圖風險上升，AUROC 可低於 0.5。
這是目標函數與物理效果矛盾，不是調 σ 能修復的。

### 5.2 答案翻轉為何有效

錯誤答案若由擾動支撐，blur 後擾動消失 → 答案回到內容支撐的正解 → **翻轉**。
乾淨圖答案由內容支撐 → 輕度 blur 仍不翻轉。
同一 blur 現象，從「破壞偵測特徵」改為「偵測特徵本身」。

### 5.3 為何實驗一 TPR 低於「被騙子集 TPR」

混合 1000 對抗中 **538 筆 attack_success=false**：擾動存在但答案未變，
flip 在機制上**無法**將其判為攻擊（blur 前後皆同一正確答）。
這解釋了實驗一 mid 池 TPR 0.407 與 TPR_fooled 0.830 的巨大落差，
也是實驗二存在的理由——但實驗二不能替代混合流量評估。

### 5.4 偵測 vs 緩解雙參數

| 角色 | 參數 | σ | 原因 |
|------|------|---|------|
| 緩解 θ\* | 5×5 | **0.8** | 須交付可視影像；IQP 隨 σ 快速下降 |
| 偵測 θ_det | 5×5 | **1.5** | blur 圖僅探針；可換更高 TPR |

流程：**先 θ_det 判斷 → 若需緩解再用 θ\* 產出淨化圖**。

---

## 6. 方法學結論（可寫入論文 Discussion）

1. **直接行為證據（VQA 翻轉）優於間接語意特徵（caption 離散度）**作為 Stage 2；
   且 flip 的 mid 池 AUROC 在實驗一已超過 Stage 1 全管線 AUROC。
2. **Stage 1 在混合流量下仍有價值**（路由、縮小 mid 池），但 AUROC ~0.67 為當前探針上限；
   大量 low 放行是端到端 TPR 瓶頸。
3. **SSS 與 θ_det 分線**是必要工程決策，非實作細節。
4. **實驗一 + 實驗二並報**可避免過度樂觀（實驗二）或過度悲觀（忽略 flip 在被騙子集上近 83% TPR）兩種誤讀。

---

## 7. 限制與後續工作

| # | 限制 | 建議下一步 |
|---|------|------------|
| 1 | 538 筆「攻擊未成功」無 flip 信號 | 分開報告「偵測擾動」vs「偵測錯答」兩任務 |
| 2 | Stage 1 low 放行過多 | 門檻消融；或 low 也做輕量 flip |
| 3 | 僅 VLAttack、僅 rotate_5 | 多攻擊 / 多探針泛化 |
| 4 | 實驗二乾淨集 = 成功題原圖 | 實驗三：與實驗一**同 qid** 的 462+538+1000 乾淨混合 |

---

## 8. 重現與資料索引

### 8.1 一鍵管線

```powershell
cd BLIP_attack\output
$root = "VQA\steps100_alpha0.002_run1000"   # 或 exp2_success1000 的絕對路徑
..\..\vlattack_gpu\Scripts\python.exe -u run_full_pipeline.py `
  --experiment_root $root --device cuda `
  --stage1_methods rotate_5 --stage2_method rotate_5
```

### 8.2 關鍵產物對照

| 內容 | 實驗一 | 實驗二 |
|------|--------|--------|
| 攻擊 log | `…/attack_dual_defense_log.json` | 同左（篩選後 1000 筆） |
| ASR 摘要 | `attack_rate_summary.json` | —（全為成功） |
| Stage 1 特徵 | `tcvad_stage1_4_academic_dataset.csv` | 同檔名 |
| 端到端 | `detector_comparison.csv` | 同檔名 |
| θ\* / θ_det | `sss_best_theta.json`, `theta_det.json` | 同檔名 |
| 合併本報告 | **`成果報告/2026-09-19_COMBINED_EXPERIMENT_REPORT.md`** | （本檔） |
| **數據補齊包（表+缺口清單）** | **`BLIP_attack/output/VQA/EXPERIMENT_DATA_SUPPLEMENT.md`** | `BLIP_attack/output/VQA/data_supplement/*.csv` |

### 8.3 程式入口（repo）

| 腳本 | 用途 |
|------|------|
| `attack_vqa.py` | VLAttack 攻擊 |
| `output/run_full_pipeline.py` | Stage1 → 分析 → SSS → Stage2 → flip |
| `output/evaluate_best_stage1_new.py` | TC-VAD Stage 1 |
| `output/sss_denoise.py` | SSS / θ\* |
| `output/answer_flip_detector.py` | θ_det / 端到端對照 |
| `exp2_success1000/build_success_dataset.py` | 篩選成功子集 |

---

## 附錄 A：論文章節建議標題

可作為學位論文或期刊的一章結構：

1. **Experimental Setup** — §1 攻擊與管線  
2. **Attack Results and Baseline Defenses** — §2.1  
3. **Experiment I: Mixed Adversarial Traffic** — §2  
4. **Experiment II: Successful Attacks Only** — §3  
5. **Unified Comparison and Mechanism Analysis** — §4–§5  
6. **Discussion and Limitations** — §6–§7  

---

## 附錄 B：實作備註（合併）

| 議題 | 處理 |
|------|------|
| RTX 50 / sm_120 | `vlattack_gpu`，torch 2.11+cu128 |
| Stage 2 資料洩漏 | GroupKFold OOF SVM |
| `chdir(BLIP_ROOT)` | 所有 `experiment_root` 先 `.resolve()` |
| OneDrive + hardlink PNG | 實驗二改實體 copy；Stage 1 加 `_open_rgb` 回退 |
| 攻擊存圖 vs dataloader 正規化 | SSS 以 ImageNet 逆正規化還原（見實驗一報告附錄） |

---

*本檔為實驗一、實驗二之權威合併摘要；數值以各目錄下 CSV/JSON 為準。*
