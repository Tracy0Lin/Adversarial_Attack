# Stage 2 高斯後：BLIP 答對率（vs 標準答案）

更新：2026-09-26 15:17 UTC  

## 怎麼做（定義）

1. 對影像做 Stage 2 高斯（本實驗 θ*：**5×5、σ 來自 `sss_best_theta.json`**，通常 0.8）。
2. 用 **同一套 BLIP VQA** 對 **煳後圖** 再問同一題。
3. 將輸出字串與 **`correct_ans`（題目標準答案）** 比對：
   - **答對**：`pred_denoised == correct_ans`
   - **仍攻擊成功／答錯**：`pred_denoised != correct_ans`（即 `fooled_after=True`）

資料已在 `sss_per_theta_raw.csv` 的 **`pred_denoised`** 欄（當初跑 `sss_denoise.py` 時就算好）。

## 注意範圍

| 範圍 | 意義 |
|------|------|
| **mid_pool_*** | 只有 **Mid 風險** 列有跑煳圖+VQA（實驗一約 391 對抗 + 395 乾淨） |
| **routed_mitigation_1000_attack** | **1000 張對抗** 依管線：Low 不煳、Mid 用煳後答案、High 視為阻擋（輸出視為 GT） |

若要对 **Low/High 也各自跑高斯+BLIP**，需擴充 `sss_denoise.py` 重跑 GPU（目前 log 未存）。

## 結果表

| 實驗 | 範圍 | n | 煳前答對 | 煳後答對 | 煳後答對率 |
|------|------|---|----------|----------|------------|
| 實驗一 | mid_pool_attack | 391 | 226（57.80%） | 279（71.36%） | 71.36% |
| 實驗一 | mid_pool_clean | 395 | 395（100.00%） | 384（97.22%） | 97.22% |
| 實驗一 | routed_mitigation_1000_attack | 1000 | 538（53.80%） | 749（74.90%） | 74.90% |
| 實驗二 | mid_pool_attack | 350 | 0（0.00%） | 160（45.71%） | 45.71% |
| 實驗二 | mid_pool_clean | 326 | 325（99.69%） | 309（94.79%） | 94.79% |
| 實驗二 | routed_mitigation_1000_attack | 1000 | 0（0.00%） | 591（59.10%） | 59.10% |

CSV：`stage2_blip_correct_after_blur.csv`

重算：`python BLIP_attack/output/VQA/stage2_blip_accuracy_after_blur.py`
