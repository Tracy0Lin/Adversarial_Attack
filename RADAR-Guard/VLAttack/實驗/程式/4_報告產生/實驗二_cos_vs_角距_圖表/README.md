# 實驗二 cos vs 角距：真實數據 Excel 與圖（2026-11-26，Asia/Taipei）

本資料夾由 11/22 正式重跑的逐張特徵與折外分數重算，沒有合成資料，也沒有改動原始檔。

## 指標是否完全重現

是。由 2,000 張折外決策值重算，四位小數與發表表一致：

| 設定 | AUROC | Recall | Precision | F1-Score | TP | FP | TN | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 未做角距離（cos） | 0.7217 | 0.6050 | 0.7043 | 0.6509 | 605 | 254 | 746 | 395 |
| 有做角距離（角距） | 0.7233 | 0.6790 | 0.6893 | 0.6841 | 679 | 306 | 694 | 321 |

完整精度：cosine AUROC = 0.7216720000000001，角距 AUROC = 0.7232600000000001。

AUROC 差（角距 − cos）照片分組 bootstrap 1000 次（seed=42）：mean = 0.001945，95% CI = [-0.0090, 0.0130]，與發表區間 [−0.0090, 0.0130] 一致。區間跨過 0。

## 檔案

- `實驗二_cos_vs_角距_真實數據.xlsx`：工作表「指標表」「逐張特徵」「畫圖用_分組特徵」「ROC」「說明」。
- `build_exp2_excel_and_figures.py`：本資料夾的重現腳本。
- `figures/組合圖_2x2_iqr與tail.png`：對應參考圖構圖（左未做角距離、右有做角距離；上 iqr、下 tail）。
- `figures/組合圖_4x2_四特徵.png`：四項特徵，左 cos（center／q95 為 1−cos）、右角距。
- `figures/圖三_…` 至 `圖六_…`：iqr、tail 單張主圖（橫軸是正式特徵）。
- `figures/圖一_…`、`圖七_…`：center、q95 主圖（cosine 側為 1−cos，軸標有寫）。
- `figures/補充_原始cosine_center與q95/`：正式 SVM 使用的原始 mean(cos)、Q95(cos)。
- `figures/指標長條圖_AUROC_Recall_Precision_F1.png`
- `figures/ROC曲線_cos_vs_角距.png`

## 來源（唯讀）

- `/workspace/radar_dossier/cos_vs_angular_20261123_archive/05_formal_1122_rerun/outputs/exp2_features.csv`
- `/workspace/radar_dossier/cos_vs_angular_20261123_archive/05_formal_1122_rerun/outputs/oof_scores.csv`
- `/workspace/radar_dossier/cos_vs_angular_20261123_archive/05_formal_1122_rerun/formal_1122_rpf1.csv`
- `/workspace/radar_dossier/cos_vs_angular_20261123_archive/05_formal_1122_rerun/outputs/bootstrap_delta.csv`

正式對照組用的是原始 cosine 四維，不是 1−cos。細節見 Excel「說明」頁。
