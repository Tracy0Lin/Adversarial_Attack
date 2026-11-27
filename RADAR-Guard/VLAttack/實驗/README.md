# 實驗

VLAttack 攻擊與 RADAR-Guard 防禦的實驗程式、資料與模型。

| 資料夾 | 內容 |
|---|---|
| `程式/` | 攻擊、訓練、評估、報告產生的腳本。各腳本屬於哪套流程，見 `程式/3_評估/README.md` |
| `資料/` | 實驗輸入、各流程的計分表與結果。哪個資料夾是報告用的，見 `資料/README.md` |
| `模型/` | 已訓練的 Stage 1 SVM（`.joblib`）、超參數 JSON，說明見 `模型/MODEL_MANIFEST.md` |

操作步驟見 `文件/實驗操作手冊.md`。

## 兩套評估流程

報告的數字來自兩套流程。兩者的 Stage 1 分組方式不同，數字不能直接混用。

| 流程 | 資料位置 | 主要腳本 | Stage 1 交叉驗證 |
|---|---|---|---|
| **主流程**（multi_captioner） | `資料/BLIP/主流程_multi_captioner/`、`資料/Florence/` | `程式/3_評估/analyze.py`、`run_analyze_sigma08.py` | 依照片分 5 折 |
| **舊流程**（dual-defense） | `資料/BLIP/舊流程/` | `程式/3_評估/evaluate_best_stage1_new.py`、`sss_denoise.py`、`answer_flip_detector.py`、`mitigation_summary.py` | 依題號分 3 折（同一照片的不同題目可能跨折） |

## 數字來源對照表

根目錄 README「核心實驗驗證摘要」的每個數字都可以用 `重算工具/reproduction_audit/verify_reported_numbers.py` 重算。

| 數字 | 來源檔 | 產生腳本 | 流程 |
|---|---|---|---|
| Exp1 ASR 46.2%（462/1000）、SSIM 0.8003 | `資料/BLIP/舊流程/exp1_mixed1000/dataset_manifest.csv` | `程式/1_攻擊與資料建立/attack_vqa.py`（攻擊紀錄） | — |
| Stage 1 AUROC 0.6640／0.6653（Exp1）、0.7217／0.7233（Exp2）與 Recall、Precision、F1 | `資料/BLIP/Stage1正式_11-22/svm_summary.csv`、`formal_1122_rpf1.csv` | `程式/3_評估/run_rotation_cos_vs_ang.py`、`compute_rpf1.py` | 依照片 5 折 |
| SSS 0.5766、ΔASR 0.4571、IQP 0.4596、SP 0.9126（σ*=0.8） | `資料/BLIP/舊流程/exp2_success1000/sss_best_theta.json` | `程式/3_評估/sss_denoise.py` | 舊流程 |
| 翻轉偵測 TPR 40.66%、FPR 5.32%、Youden J 0.3535、被騙樣本 TPR 83.03%（σ_det=1.5） | `資料/BLIP/舊流程/exp2_success1000/baseline_exp1/theta_det.json` | `程式/3_評估/answer_flip_detector.py` | 舊流程 |
| Exp2 防禦後 ASR 40.9% | `資料/BLIP/舊流程/exp2_success1000/mitigation_summary.csv` | `程式/3_評估/mitigation_summary.py` | 舊流程 |
| Exp1 防禦後 ASR 24.0%、Exp3 35.9%（README 寫 24.1%、36.0%） | `資料/BLIP/主流程_multi_captioner/exp{1,3}_blip/sigma08_fixed/pipeline_variants.csv`（`A3_full_mitigate_only`） | `程式/3_評估/run_analyze_sigma08.py` → `analyze.py` | 主流程 |
| 主流程 Exp2 防禦後 ASR 39.5% | `資料/BLIP/主流程_multi_captioner/exp2_blip/sigma08_fixed/pipeline_variants.csv` | 同上 | 主流程 |
| 部署模型與 OOF AUROC 0.6456 | `模型/stage1_rbf_svm_rotate5_exp1.joblib`、`模型/model_metadata.json` | `程式/2_訓練/train_stage1_svm.py` | — |
| Florence-2 三實驗 | `資料/Florence/florence_vlattack/exp{1,2,3}/florence2/vqa_florence2_sigma08_fixed/` | `程式/3_評估/analyze.py`（`--vqa-backend florence2`） | 主流程 |
| C1～C7 分流分析 | `資料/分流分析_C1到C7/` | `程式/4_報告產生/stage_routing_deep_dive/build_report.py` | 主流程（預設 θ） |
| 題目難度 × C1～C7 | `資料/題目難度/c1c7_by_difficulty.csv` | `程式/4_報告產生/regen_c1c7_by_difficulty.py`（還原版，原檔遺失） | 主流程（預設 θ） |

## 已知問題

- **`資料/Florence/florence_victim/` 的 Exp1 與 Exp2 是同一批資料。** 這組用白盒 PGD 直接攻擊 Florence-2，Exp1 的 1,000 題全部攻擊成功；Exp2 是從 Exp1 挑出攻擊成功的 1,000 題（`程式/Florence/攻擊/florence_victim/build_exp2.py`），所以兩者的資料與數字相同，這是正常結果。報告用的 Florence 數字來自 `florence_vlattack`。
- **初始 ASR 有三種算法。** 攻擊當下的紀錄：Exp1 46.2%（462/1000）、Exp2／Exp3 100%。存成 PNG 後重測、與紀錄的原答案比：46.1%、99.8%、99.6%（計分表的 `fooled_before`）。存成 PNG 後重測、與乾淨圖重測的答案比：Exp1 46.3%（`資料/BLIP/主流程_multi_captioner/three_exp_report_metrics.json`，`程式/4_報告產生/export_three_exp_report_metrics.py`）。
