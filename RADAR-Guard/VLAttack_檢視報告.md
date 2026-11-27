# VLAttack 檢視報告

檢視日期：2026-11-27（依 commit `64ac36d` 第二批補入的資料更新）
檢視範圍：`RADAR-Guard/VLAttack/` 全部文件、結果檔與主要程式；與 `RADAR-Guard/AttackVLM/` 比較；書面報告參考文獻 [17]～[21]。
驗證方式：用 repo 內現有的 CSV／JSON／模型檔，以原程式的算法重新計算，再與文件和結果檔逐項比對（CPU，不重跑攻擊）。腳本與完整紀錄在 `VLAttack/重算工具/reproduction_audit/`。

---

## 一、總結

- **主要實驗結論正確。** 攻擊成功率、Stage 1 偵測成績、Stage 2 修復參數（含 IQP、SP、SSS）、三個模型檔、BLIP 與 Florence-2（`florence_vlattack`）的完整流程，都能從 repo 內的資料重算並吻合；除了第二節列出的差異，其餘全部一致。完整逐項紀錄見 `VLAttack/重算工具/reproduction_audit/`。
- **需要修改的數據有 2 處：**
  1. 端到端 ASR 表混用了兩套流程和兩種算法；
  2. 初始 ASR 在不同文件裡有 46.1%、46.2%、46.3% 三種寫法。
- Florence-2 的數字一律以 `florence_vlattack`（VLAttack 特徵攻擊）為準，本報告不討論其他 Florence 研究。
- **架構大致合適。** 有幾處重複檔案和命名容易混淆，另外少了幾份能證明數字可重現的資料。

---

## 二、需要修改的數據

### 2.1 端到端 ASR 表混用兩套流程（必改）

**出現位置**（三處都要一起改）：
- `VLAttack/README.md` 第 71 行
- `文件/實驗操作手冊.md` 第 99～101 行
- `系統/1_系統完整程式/templates/results.html` 第 115～118、149～156、199～206 行

**問題**：同一張表的數字來自不同流程和不同算法。

| 欄位 | 目前寫的（Exp1／Exp2／Exp3） | 實際來源 |
|---|---|---|
| 初始 ASR | 46.2%／100%／100% | 攻擊當下的紀錄 |
| Stage 1 後 | 29.8%／55.8%／50.9% | Exp2、Exp3 為「以攻擊紀錄為母體」的算法；Exp1 的 29.8% 找不到可對應的結果檔 |
| 兩階段後 | 24.1%／**40.9%**／36.0% | Exp1、Exp3 來自 `實驗/資料/BLIP/主流程_multi_captioner`；**Exp2 來自較舊的 `exp2_success1000` 流程** |

Exp2 的 40.9% 由另一套流程算出：依題號分組三折，中風險攻擊圖 350 張。同一套 `實驗/資料/BLIP/主流程_multi_captioner` 流程算出來應為 **39.5%**（中風險攻擊圖 338 張）。`文件/成果報告/2026-11-09_難度與C1到C7.md` 已提到這個差異，但三份文件都沒有註明。

**建議改法**：統一用 `實驗/資料/BLIP/主流程_multi_captioner/<exp>_blip/sigma08_fixed/pipeline_variants.csv`，一種算法、一套流程，分母都是 1,000 張攻擊圖：

| 實驗 | 初始 ASR | Stage 1 後 | 兩階段後 | 總下降 |
|---|---:|---:|---:|---:|
| Exp1 | 46.1% | 29.7% | **24.0%** | 22.1 個百分點 |
| Exp2 | 99.8% | 55.8% | **39.5%** | 60.3 個百分點 |
| Exp3 | 99.6% | 50.7% | **35.9%** | 63.7 個百分點 |

如果想保留「攻擊當下全部成功＝100%」的寫法，就在表下加註：「初始 ASR 以攻擊紀錄計；之後各欄以存成 PNG 後重新測試計，Exp2、Exp3 各有 2、4 張攻擊圖存檔後失效。」

### 2.2 初始 ASR 有三種寫法（建議統一）

| 寫法 | 出處 | 意思 |
|---|---|---|
| 46.2%（462/1000） | README、實驗手冊、`attack_rate_summary.txt` | 攻擊當下的紀錄 |
| 46.1%（461/1000） | `實驗/資料/BLIP/主流程_multi_captioner` 各計分表 | 存成 PNG 後重新測試 |
| 46.3% | `實驗/資料/BLIP/主流程_multi_captioner/three_exp_report_metrics.json` | 找不到對應算法 |

**建議**：正文寫 46.2%，並註明它是攻擊紀錄的數字；表格裡的流程數字用 46.1%。46.3% 不要引用。

### 2.3 部署模型的驗證成績（可不改，建議加註）

`實驗/模型/MODEL_MANIFEST.md` 與 `model_metadata.json` 記錄的 OOF AUROC 0.6456，是訓練腳本以題號推算照片編號後算出來的。推算有誤，207 張照片被切成 333 組，同一張照片的不同題目可能分在不同折。以正確的照片分組重算為 0.6599。部署模型本身用全部資料訓練，不受影響。

### 2.4 不需要修改、但要知道的

- `exp2_success1000` 這套流程（README 的 SSS 0.5766、ΔASR 0.4571、Exp2 40.9%）以題號分三折。改用照片分組估算時，Exp2 端到端 ASR 約變為 41.5%，差距不到 1 個百分點。
- `實驗/資料/BLIP/主流程_multi_captioner/exp1_blip`、`exp2_blip`（預設 θ 版）的計分表中，有 5 張乾淨圖被誤標 `fooled_before=True`。這些統計只計攻擊圖，修正前後計算結果完全相同。

---

## 三、已驗證正確的數據

| 項目 | 數字 |
|---|---|
| 攻擊 | Exp1 ASR 46.2%（攻擊紀錄）、SSIM 0.8003 |
| Stage 1 正式 11/22 | Exp1 Cosine 0.6640／Angular 0.6653；Exp2 0.7217／0.7233；所有 Recall、Precision、F1；Bootstrap CI [−0.0204, 0.0214]、[−0.0090, 0.0130] |
| Stage 2 | SSS 0.5766、ΔASR 0.4571、IQP 0.4596、SP 0.9126；翻轉偵測 TPR 40.66%、FPR 5.32%、Youden J 0.3535、被騙樣本 TPR 83.03% |
| 模型檔 | 三個 `.joblib` 與重新訓練的輸出完全相同；網站與 `實驗/模型` 的部署模型輸出相同 |
| Florence-2（`florence_vlattack`） | Exp1～3 的風險分數、分流、七種流程變體、SSS 格點（IQP、SP、SSS）全部吻合 |
| BLIP 預設 θ／sigma08_fixed | Exp1～3 全部可從計分表、`probe_gaussian.csv`、`caption_features.csv` 重現 |
| C1～C7（BLIP） | Exp1～3 可從計分表完全重現 |
| `c1c7_by_difficulty.csv` | 還原版 `regen_c1c7_by_difficulty.py` 重算全部 72 列，0 格不同 |
| 資料集描述 | 300 張良性圖；109 對成對樣本（20＋82＋7）；題目難度 2,538 題（easy 1,288／medium 598／hard 652）；Exp3 照片與 Exp1／Exp2 零重疊 |

---

## 四、整體架構評估

### 2026-11-27 已全面整理資料夾架構

| 舊架構 | 新架構 |
|---|---|
| `實驗設計_VLAttack/1_資料前處理`（實際上主要是攻擊程式） | `實驗/程式/1_攻擊與資料建立/`；Florence、題目難度、備份各自獨立成資料夾 |
| `2_訓練腳本`（5 支只有 1 支是訓練） | `實驗/程式/2_訓練/` 只留 `train_stage1_svm.py`，刪除 4 個與評估腳本完全相同的副本 |
| `3_評估腳本`（28 支平鋪） | `實驗/程式/3_評估/`（互相 import 的評估腳本）＋`4_報告產生/`（7 支報告腳本、C1～C7、實驗二圖表） |
| `5_資料集/tabular_features_and_logs/`（看不出哪個是報告用） | `實驗/資料/BLIP/{主流程_multi_captioner, 舊流程, Stage1正式_11-22, 攻擊參數搜尋}`、`實驗/資料/Florence/` 等 |
| `6_模型參數`、手冊、SDS、`成果報告` 分散各處 | `實驗/模型/`；手冊、SDS、成果報告集中到 `文件/` |
| 溯源檔散在根目錄 | `溯源/`，另新增 `新舊路徑對照.tsv` |

**整理後確認：**
- 重現驗證結果不變：387 項吻合、5 項不一致、3 項略過。
- 溯源清單 565 個路徑都存在；只有 3 個改了路徑設定的檔案雜湊值改變。
- 訓練腳本、網站模型路徑、重跑工具都能正常運作。

新增導覽：`實驗/README.md`（含**數字來源對照表**）、`實驗/資料/README.md`、`實驗/程式/3_評估/README.md`、`文件/成果報告/README.md`、`溯源/README.md`。

**沒有改的：**
- **原始實驗腳本裡的路徑：** 都是原電腦的路徑，保持原檔。
- **歷史報告：** `文件/成果報告/` 與兩份 `EXPERIMENT_REPORT.md` 的內文不動，只修了 3 個斷掉的相對連結。
- **原電腦輸出資料夾的名稱：** `exp2_success1000` 等保留原名，方便追溯。

### 仍待處理

| # | 問題 | 建議 |
|---|---|---|
| 1 | ~~兩套評估流程沒說明~~ | ✅ 已在 `實驗/README.md` 加入數字來源對照表 |
| 2 | `common.py`、`captioners.py`、`evaluate_best_stage1_new.py` 在 `1_攻擊與資料建立/` 和 `3_評估/` 各有一份 | 這兩份是各自資料夾 import 需要的，刪了會跑不起來，所以保留（`2_訓練/` 的副本已刪除） |
| 3 | `cleverhans/` 整包 511 個檔案，實際只用到 `cleverhans/torch/attacks/BLIP/` | 可精簡；至少在 README 註明 |
| 4 | 三份環境檔用途沒寫清楚：`requirements.txt` 給網站、`requirements_experiment.txt` 給實驗、`environment.yaml` 是原專案的舊環境 | README 各加一句用途 |
| 5 | `實驗/程式/1_攻擊與資料建立/requirements-gpu.txt` 有一行少了換行（`transformers==4.26.1tokenizers==0.13.3`），pip 安裝會失敗 | 補換行 |
| 6 | 程式寫死原作者電腦路徑（`vlattack_engine.py` 8 處，Florence 等腳本也有） | 改用環境變數或相對路徑；至少保留目前 README 的說明 |
| 7 | 網站安全：`job_id` 未檢查格式、對外開放 `0.0.0.0`、錯誤訊息含本機路徑 | 只在本機展示就改綁 `127.0.0.1`；`job_id` 加格式檢查 |

---

## 五、需要補上的

### 第二批（commit `64ac36d`）已補上

| 原本缺少 | 狀態 |
|---|---|
| BLIP 預設 θ 的 Exp3 計分表 | ✅ 已補，C1～C7 Exp3 已驗證 |
| `probe_gaussian.csv`（BLIP 與 Florence） | ✅ 已補，IQP、SP、SSS 格點已驗證 |
| `c1c7_by_difficulty.csv` 的產生腳本 | ⚠️ 原檔遺失，已補還原版 `regen_c1c7_by_difficulty.py`，可 100% 重現 72 列 |
| 腳本來源 SHA-256 清單 | ✅ `溯源/SHA256SUMS_scripts.txt`、`溯源/SCRIPT_ORIGIN_MAP.tsv` |
| Exp1 翻轉偵測逐張紀錄 | ✅ 已確認在 `steps100_alpha0.002_run1000/sss_per_theta_raw.csv`（README 記錄重算一致，本驗證未另外重算） |
| VQAv2 官方標註 | ✅ README 已附官方下載連結與 SHA-256 |

### 仍然缺少

| # | 缺少 | 說明 |
|---|---|---|
| 1 | 五折模型檔與每折測試 ID | 當時沒有存，但演算法是確定性的，可以重新產生（不需 GPU） |
| 2 | Release 圖片 zip（BLIP Exp3、`florence_vlattack` 三組、題目難度圖、C1～C7 PNG 等） | 已在原電腦打包並列出雜湊，**尚未上傳**（需要 `gh auth login`） |
| 3 | `florence_vlattack` 預設 θ 的格點檔（`sss_grid_mid.csv`、`theta_det_grid_mid.csv`） | 只影響 3 項驗證，報告用的 sigma08_fixed 版本有格點檔 |
| 4 | 8 個腳本在原電腦找不到原檔 | 包括產生正式數字的 `train_stage1_svm.py`、`run_rotation_cos_vs_ang.py`、`compute_rpf1.py`；它們的輸出已驗證可重現，但無法證明與當時執行的版本完全相同 |

`steps100_alpha0.002_run1000` 的 16 GB 圖片、COCO 完整影像、BLIP 權重，README 已說明不放進 repo，不算缺漏。

### 文件

- ~~README 加「數字來源對照表」~~：✅ 已加在 `實驗/README.md`。
- README 註明「部署模型沒有獨立測試成績，不能引用五折數字」（AttackVLM 的 `MODELS.md` 已有同樣說明，可參考）。

---

## 六、參考文獻 [17]～[21] 核對

| 編號 | 文獻 | 內容核對 | VLAttack 是否用到 |
|---|---|---|---|
| [17] | Rombach et al., Latent Diffusion Models, CVPR 2022, pp. 10684–10695 | 正確 | 否（AttackVLM 用 SD 1.4 產生目標圖） |
| [18] | Wang et al., SSIM, IEEE TIP 13(4), pp. 600–612, 2004 | 正確 | 是（對抗圖 SSIM、Stage 2 品質指標） |
| [19] | Zhang et al., LPIPS, CVPR 2018, pp. 586–595 | 正確 | 是（`sss_per_theta_raw.csv` 的 `lpips`） |
| [20] | Mittal et al., BRISQUE, IEEE TIP 21(12), pp. 4695–4708, 2012 | 正確 | 是（`sss_per_theta_raw.csv` 的 `brisque`） |
| [21] | Vedantam et al., CIDEr, CVPR 2015, pp. 4566–4575 | 正確 | 否（AttackVLM 使用） |

**格式小建議**：
- 日期寫法不一致，[21] 寫「June 7-12, 2015」，[17]、[19] 只寫月份。建議統一。
- [20] 的 URL 與 doi 重複，可只留 doi。

---

## 七、建議處理順序

1. **修正端到端 ASR 表（2.1）：** 改 README、實驗手冊、`results.html` 三處，統一數字來源。
2. **統一初始 ASR 寫法（2.2）：** 對照表已在 `實驗/README.md`，README 正文仍需改寫。
3. **上傳 Release 圖片 zip（五、仍缺 #2）。**
4. ~~架構整理~~：✅ 2026-11-27 已全面整理；四、仍待處理 #3～#7 依時間決定。
