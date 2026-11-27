# 實驗二：1000 乾淨 vs 1000「攻擊成功」樣本

目錄：`BLIP_attack/output/VQA/exp2_success1000/`
狀態：**全部完成**（2026-09-15 06:10，`===== PIPELINE COMPLETE =====`）

| Phase | 狀態 |
|-------|------|
| 0 保存實驗一結果 | ✅ `baseline_exp1/` |
| 1 補跑攻擊 | ✅ 2,298 筆、成功 1,044（ASR 45.43%） |
| 2 篩選建資料集 | ✅ 1,000 成功 + 1,000 乾淨 |
| 3 五段 pipeline | ✅ Stage 1 AUROC 0.7212；答案翻轉端到端 J **0.538** |
| 4 撰寫報告 | ✅ `EXPERIMENT_REPORT.md` |

---

## 1. 這個實驗要解決什麼

實驗一（`steps100_alpha0.002_run1000`）的對抗樣本是「所有被攻擊過的 1000 張」，
但其中只有 **462 張真的騙到模型**，另外 538 張雖然帶有擾動、模型仍答對。

這造成兩個問題：

1. **TPR 被稀釋。** 答案翻轉偵測器的原理是「擾動被洗掉 → 答案塌回正確解」，
   對於本來就沒改變答案的樣本，機制上無從偵測。實驗一報告第 10 節列為首要限制。
2. **問題定義模糊。** 「偵測有擾動的圖」和「偵測被攻擊成功的圖」是兩個不同任務。
   後者才是實務上真正有價值的（答案已經錯了才需要攔）。

實驗二把對抗側限定為 **1000 張確實攻擊成功**的樣本，直接量測
「當攻擊真的奏效時，兩階段防禦能抓到多少」。

---

## 2. 資料來源盤點（已完成）

| 來源 | 樣本數 | 成功數 | 有存圖？ |
|------|--------|--------|---------|
| `steps100_alpha0.002_run1000` | 1000 | **462** | ✅ 1000 對 |
| `attack_dual_defense_log_0401.json` | 10000 | 6281 | ❌ 無圖 |
| `attack_dual_defense_log_0304.json` | 4741 | 2887 | ❌ 無圖 |
| 其他舊 run（05_08、result_test 等） | — | — | ⚠️ 僅 18–317 對，且攻擊參數不同 |

全部目錄的 ori/atk 圖對取聯集只有 **1,010 對**。
上萬筆的舊 log 當年只記錄指標、沒存影像，**無法用於 Stage 1**（需要影像跑 caption）。

其他舊 run 雖有圖，但 steps/alpha 與本實驗不同，混用會混淆變因，**排除**。

**結論：必須補跑攻擊。**

---

## 3. 需要補跑多少

以實驗一實測 ASR **46.20%** 推算：

```
目標成功數 1000 ÷ 0.462 ≈ 2165 筆需被攻擊
已攻擊 1000 筆（得 462 成功）
還需攻擊 ≈ 1165 筆（可得 ≈ 538 成功）
```

| 項目 | 估計 |
|------|------|
| 補跑樣本數 | ~1,200（留餘裕） |
| 速率 | ~23 秒/筆 |
| **GPU 時間** | **約 7.5 小時** |
| 新增影像 | 1,200 × 12 = 14,400 張，約 4.3 GB |
| 可用樣本池 | `right_vqa_list.txt` 共 167,147 筆，充足 |
| 磁碟餘裕 | 449.8 GB 可用，無虞 |

---

## 4. 執行流程

### Phase 0 · 保存實驗一結果

實驗一的 ASR 46.20% 是對「1000 筆被攻擊樣本」算的。補跑會讓該資料夾樣本數變成 2200，
原數字會失效，因此先把實驗一的 log 與摘要快照到本資料夾 `baseline_exp1/`。

（實驗一的完整報告已 commit 到 git 的 `original` 分支，另有保障。）

### Phase 1 · 補跑攻擊至 2200 筆

沿用同一資料夾 resume，才能重用已有的 462 個成功樣本與 1000 對影像：

```powershell
cd BLIP_attack
python -u attack_vqa.py --method VLAttack --steps 100 --alphas 0.002 `
  --run_suffix _run1000 --num_samples 2200
```

`--num_samples 2200` 會讀取既有 log（1000 筆）並從第 1001 筆續跑，不重算。
中斷可用同一條指令接續。

### Phase 2 · 篩選並建立實驗二資料集

用 `build_success_dataset.py`（待建）做：

1. 讀 `steps100_alpha0.002_run1000/attack_dual_defense_log.json`
2. 篩出 `attack_success == true` 且 `_1_ori.png`、`_2_atk.png` 都存在的樣本
3. 取前 1000 筆，寫出 `attack_dual_defense_log.json`（過濾後）到本資料夾
4. 用 **hardlink**（同磁碟不佔額外空間）把這 1000 筆的 ori/atk 共 2000 張連到本資料夾 `visual_results/`
5. 輸出 `dataset_manifest.csv`（qid、question、ground_truth、原始預測、對抗預測）

這樣本資料夾就符合 pipeline 預期的結構，可直接餵給既有腳本。

### Phase 3 · 跑完整 pipeline

```powershell
cd BLIP_attack/output
python -u run_full_pipeline.py --experiment_root "VQA\exp2_success1000" `
  --device cuda --stage1_methods rotate_5 --stage2_method rotate_5
```

五段依序自動執行：

| 段 | 內容 | 預估時間 |
|----|------|---------|
| Stage 1 | rotate_5、k=10、2000 列（1000 乾淨 + 1000 對抗） | ~3.5 h |
| analysis | 風險分段交叉分析 | ~1 min |
| SSS | 中風險池 × 5 個 σ，選 θ\* | ~2.5 h |
| Stage 2（對照組） | 高斯 + caption-SVM 重評 | ~1 h |
| 答案翻轉偵測器 | θ_det 掃描 + 端到端對照 | ~1 min |

**Phase 3 總計約 7 小時。**

---

## 5. 預期產出與對照方式

最終要能填出這張表，和實驗一並列：

| Pipeline | 實驗一（混合） | 實驗二（僅成功） |
|----------|---------------|-----------------|
| Stage 1 單獨 | J = 0.232 | ? |
| + caption-SVM Stage 2 | J = 0.217 | ? |
| + 答案翻轉（θ_det） | J = 0.370 | ? |
| + 多 σ 翻轉計數 | J = 0.387 | ? |

以及：

| 指標 | 實驗一 | 實驗二 |
|------|--------|--------|
| Stage 1 AUROC | 0.6655 | ? |
| 中風險率 | 39.30% | ? |
| θ\*（緩解） | 5×5, σ=0.8 | ? |
| θ_det（偵測） | 5×5, σ=1.5 | ? |
| SSS 最佳值 | 0.5834 | ? |
| 多 σ 翻轉 AUROC | 0.7035 | ? |

---

## 6. 預期結果與判讀

### 幾乎確定會發生

**答案翻轉的 TPR 大幅上升。** 實驗一中風險池裡「攻擊成功」的樣本，
單看它們的翻轉率在 σ=1.5 時是 **83.0%**；實驗二全體都是成功樣本，
TPR 應該會從 0.407 往 0.7–0.8 靠。

### 需要觀察的

**Stage 1 的 AUROC 會不會上升。** 有兩種可能，都有意義：

- **上升** → 說明「攻擊成功」的樣本擾動較強/較結構化，caption 語意離散度確實更明顯，
  代表 Stage 1 的 0.67 上限部分來自弱擾動樣本的干擾。
- **不變** → 進一步鞏固「0.67 是方法上限」的結論，因為連最容易分的子集都突破不了。

**FPR 會不會跟著上升。** 乾淨側不變（仍是 1000 張原圖），
所以 FPR 理論上應該和實驗一相近；若明顯變動，就要回頭檢查 OOF 切分是否受樣本組成影響。

### 這個實驗不能回答什麼

實驗二把「攻擊失敗但有擾動」的樣本排除了，所以**它的 TPR 不能直接當作部署效能**——
實務上防禦器面對的是混合流量。實驗二測的是「上限」（攻擊奏效時抓得到嗎），
實驗一測的是「混合流量下的實際表現」。兩者互補，報告中須並列且說明差異。

---

## 7. 檔案結構（預期）

```
exp2_success1000/
├── PLAN.md                          # 本文件
├── baseline_exp1/                   # 實驗一快照（Phase 0）
├── build_success_dataset.py         # 篩選腳本（Phase 2）
├── dataset_manifest.csv             # 1000 筆成功樣本清單
├── attack_dual_defense_log.json     # 過濾後的 log
├── visual_results/                  # 2000 張 hardlink（ori + atk）
├── phase1_attack.log                # 補跑攻擊的完整過程
├── phase3_pipeline.log              # pipeline 的完整過程
├── tcvad_stage1_4_academic_dataset.csv
├── tcvad_stage1_metrics.csv
├── academic_plots/ risk_tables/ teacher_examples/
├── sss_grid_search.csv  sss_best_theta.json  sss_per_theta_raw.csv
├── stage2_gaussian_results.csv  stage2_gaussian_summary.csv
├── theta_det_sweep.csv  theta_det.json  detector_comparison.csv
└── EXPERIMENT_REPORT.md             # 最終報告（含與實驗一對照）
```

---

## 8. 總時程

| Phase | 內容 | 時間 |
|-------|------|------|
| 0 | 保存實驗一結果 | < 1 min |
| 1 | 補跑攻擊 1000 → 2200 | **~7.5 h** |
| 2 | 篩選建資料集 | ~2 min |
| 3 | 五段 pipeline | **~7 h** |
| 4 | 撰寫報告 | ~10 min |

**合計約 15 小時 GPU**，需分兩段執行（Phase 1 完成後才能做 Phase 2）。

執行期間請避免機器睡眠或關閉 Cursor；兩個長階段都支援中斷後用同一條指令 resume。
