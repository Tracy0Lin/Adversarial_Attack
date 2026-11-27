# 重現驗證紀錄

`verify_reported_numbers.py` 只讀 repo 內既有的 CSV／JSON 與 `.joblib`，用 CPU 重新計算以下結果，並把每一項比對寫進 `reproduction_audit.json`：

- 根目錄 README「核心實驗驗證摘要」裡的所有數字
- BLIP 與 Florence-2 兩個受害模型的完整流程結果
- C1～C7 分流深入分析
- 題目難度標註

腳本不修改任何實驗檔，也不需要 GPU、BLIP 權重或圖片。

## 執行方式

需要 Python 3.10 或 3.11，套件版本與網站環境相同：`scikit-learn==1.6.1`、`numpy==1.26.4`、`pandas==2.3.3`、`joblib==1.6.0`。

```bash
python 重算工具/reproduction_audit/verify_reported_numbers.py
```

缺少某個輸入檔時，該項會記為 `SKIPPED`，其他項照常執行。

## 結果（2026-11-27，資料版本 commit `64ac36d`，scikit-learn 1.6.1）

共 395 項：**388 項 PASS、4 項不一致、3 項略過**，0 項錯誤。

| 類別 | 檢查內容 | PASS | 不一致 | 略過 |
|---|---|---:|---:|---:|
| 1 攻擊 | Exp1 ASR、SSIM | 2 | | |
| 2 Stage 1 正式 11/22 | Exp1／Exp2 cosine 與角距四維的 AUROC、Recall、Precision、F1 | 26 | | |
| 3 Stage 1 分流 | `tcvad_stage1_4_academic_dataset.csv` 逐列風險分數與分類 | 4 | | |
| 4 Stage 2 | SSS σ*=0.8、翻轉偵測 σ_det=1.5 | 14 | | |
| 5 端到端 ASR | Exp1／Exp2／Exp3 防禦後 ASR | 3 | 2 | |
| 6 模型檔 | 三個 `.joblib` 與 `model_metadata.json` | 16 | | |
| 7 BLIP sigma08_fixed | Exp1～3，見下方說明 | 50 | | |
| 7 BLIP 預設 θ | Exp1～3（C1～C7 的來源） | 46 | 2 | |
| 7 Florence-2 sigma08_fixed（`florence_vlattack`，報告用） | Exp1～3 | 50 | | |
| 7 Florence-2 預設 θ（`florence_vlattack`） | Exp1～3 | 41 | | 3 |
| 7 Florence_victim sigma08_fixed | Exp1～3 | 50 | | |
| 7 Florence_victim 預設 θ | Exp1～3（C1～C7 的來源） | 48 | | |
| 8 C1～C7 BLIP_victim | 從計分表重算分類、筆數、照片數、百分比 | 9 | | |
| 8 C1～C7 Florence_victim | 同上 | 10 | | |
| 9 題目難度 | `labels.csv` 合併、二元難度、還原版腳本重算 `c1c7_by_difficulty.csv` 全部 72 列 | 19 | | |

**第 7 類每一個 Exp 檢查的項目：**
- Stage 1 逐列風險分數（Exp1／Exp2 依照片五折；Exp3 以 Exp2 全部資料訓練後套用）
- 低／中／高分類、`fooled_before`
- 完整流程的攔截與回答
- 7 種流程變體各 12 個指標
- `summary.json` 的計數、AUROC、中風險比例
- 翻轉偵測格點
- SSS 格點：ΔASR、乾淨圖改答率，以及由 `probe_gaussian` 和 `caption_features` 重算的 IQP、SP、SSS

**3 項略過：** `florence_vlattack` 的預設 θ 資料夾只有計分表、流程變體與 summary，沒有格點檔。

## 不一致項目（第 1～4 項仍存在；第 5、6 項為已釐清或已修正的紀錄）

| # | 項目 | 狀況 | 影響 |
|---|---|---|---|
| 1 | README Exp1 防禦後 ASR | 寫 24.1%，計分表為 **24.0%**（240/1000） | 差 0.1 個百分點 |
| 2 | README Exp3 防禦後 ASR | 寫 36.0%，計分表為 **35.9%**（359/1000） | 差 0.1 個百分點 |
| 3 | BLIP 預設 θ Exp1 計分表 | 3 張乾淨圖被標成 `fooled_before=True`（題號 393266001、262200002、524333005） | 現行 `analyze.py` 只會對攻擊圖標 True，推測此檔由較早版本產生；修正前後所有統計完全相同 |
| 4 | BLIP 預設 θ Exp2 計分表 | 同上，2 張（262200002、263382003） | 同上 |
| 5 | `實驗/資料/分流分析_C1到C7/Florence_victim` Exp1（**已釐清，不是錯誤**） | Exp1 與 Exp2 的資料完全相同 | Florence_victim 用白盒 PGD 攻擊，Exp1 的 1,000 題全部攻擊成功；Exp2 由 `build_exp2.py` 從 Exp1 挑出前 1,000 筆成功題，所以是同一批。驗證腳本已改為檢查這個原因，結果 PASS |
| 6 | `實驗/資料/分流分析_C1到C7/Florence_victim` Exp3（**已修正**） | 原報告 C1 為 540 筆、攻擊成功 999 張；計分表為 541 筆、1,000 張，多出題號 132288000 | 報告產生後 Exp3 又替換了樣本（`replace_exp3_one.py`）。2026-11-27 已用 `重算工具/rerun_florence_routing.py` 重新產生，現在吻合 |

## 需要知道的事

- **README 的三個端到端 ASR 來自兩套流程。**
  - Exp1、Exp3 出自 `實驗/資料/BLIP/主流程_multi_captioner`（`analyze.py`，依照片分組五折）。
  - Exp2 出自 `exp2_success1000`（`evaluate_best_stage1_new.py`，依題號分組三折）。
  - 同一套 `實驗/資料/BLIP/主流程_multi_captioner` 流程的 Exp2 為 39.5%。
- **依題號分組時，同一張照片的不同題目可能分在不同折。** 第 3、4 類與 Exp2 端到端 ASR 都屬於這一套。改用照片分組估算時，約 8～13% 的樣本換了風險等級，Exp2 端到端 ASR 約由 40.9% 變為 41.5%。
- **部署模型的 OOF 成績（AUROC 0.6456）** 是以訓練腳本原本的分組方式算出：207 張照片被切成 333 組。部署模型本身用全部資料訓練，不受分組影響。
- **網站與 `實驗/模型` 的部署模型檔 SHA-256 不同**，因為複製時重新序列化，但模型內容與輸出完全相同。
- **`c1c7_by_difficulty.csv` 的原始產生程式已遺失。** `實驗/程式/4_報告產生/regen_c1c7_by_difficulty.py` 是還原版，本驗證確認它能 100% 重現全部 72 列。
- **Windows 路徑長度：** repo 放在很深的資料夾時（完整路徑超過 260 字元），Python 會讀不到部分檔案。請把 repo 放在較短的路徑，或開啟 Windows 長路徑支援。
