# 資料

各資料夾的檔案明細、大小與原電腦來源路徑，見 `DATASET_MANIFEST.md`。大型圖片放在 GitHub Release，見 `release_zips/SHA256SUMS.txt` 與根目錄 README。

| 資料夾 | 內容 | 報告是否引用 |
|---|---|---|
| `輸入/` | VQAv2 標註與答案表（`annotations/`）、實驗用的 300 張 COCO 良性原圖（`benign_val2014_300/`） | — |
| `BLIP/主流程_multi_captioner/` | Exp1～3 的特徵、`probe_gaussian.csv`、計分表；`sigma08_fixed/` 是固定 σ=0.8 的版本 | ✅ Exp1／Exp3 端到端 ASR、C1～C7 |
| `BLIP/舊流程/steps100_alpha0.002_run1000/` | Exp1 原始攻擊 1,000 題的統計、翻轉偵測、SSS | ✅ 翻轉偵測 σ_det=1.5 |
| `BLIP/舊流程/exp1_mixed1000/` | Exp1 混合實驗（2,000 列）的 Stage 1 特徵與風險表 | ✅ Exp1 ASR 46.2%、SSIM |
| `BLIP/舊流程/exp2_success1000/` | Exp2 全成功實驗的 Stage 1／Stage 2、SSS、緩解彙整 | ✅ SSS 0.5766、Exp2 40.9% |
| `BLIP/Stage1正式_11-22/` | 11/22 正式重跑的 cosine 與角距四維特徵、OOF 分數、Bootstrap | ✅ Stage 1 AUROC |
| `BLIP/攻擊參數搜尋/` | 攻擊隱蔽性與自然度的參數掃描摘要 | 補充 |
| `Florence/florence_vlattack/` | Florence-2 受害模型（VLAttack 特徵攻擊）Exp1～3 | ✅ Florence 數字 |
| `Florence/florence_victim/` | 另一組 Florence-2 研究（PGD 攻擊）Exp1～3 | 補充（Exp1 攻擊 100% 成功，與 Exp2 同一批） |
| `分流分析_C1到C7/` | BLIP 與 Florence_victim 的七類分流分析報告與抽樣清單 | 補充 |
| `題目難度/` | 題目難度標註（BLIP 與 Florence）與 `c1c7_by_difficulty.csv` | 補充 |
| `成對樣本圖/` | 109 對良性／對抗 PNG（smoke20、deep_dive_blip、網站 7 個案例） | 展示 |
| `實驗清單/` | 7 組圖片資料集的 `dataset_manifest.csv`、`dataset_info.txt`、攻擊紀錄 | 溯源 |
| `補充資料表/` | 為論文／報告彙整的表格、檢核表與案例 | 補充 |
| `release_zips/` | Release 圖片 zip 的 SHA-256 | 溯源 |

資料夾 `steps100_alpha0.002_run1000`、`exp1_mixed1000`、`exp2_success1000` 保留原電腦的輸出資料夾名稱，方便對照原始紀錄。
