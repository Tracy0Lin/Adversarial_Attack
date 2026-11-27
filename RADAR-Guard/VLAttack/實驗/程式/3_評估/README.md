# 評估腳本

這些腳本彼此會 import（例如主流程的 `captioners.py`、`run_probe.py`、`vqa_backends.py` 會用到舊流程的 `evaluate_best_stage1_new.py` 與 `sss_denoise.py`），所以放在同一層，不分子資料夾。

腳本裡寫死的路徑是原作者電腦的路徑，換機器執行前需要修改。只想驗證數字的話，用 `重算工具/reproduction_audit/verify_reported_numbers.py`，不需要改路徑。

## 主流程（multi_captioner，Stage 1 依照片分 5 折）

| 腳本 | 用途 |
|---|---|
| `common.py` | 共用路徑、實驗定義與影像工具 |
| `captioners.py` | 圖片描述後端（BLIP／ViT-GPT2／GIT／gemma3），含快取 |
| `run_probe.py` | 高斯模糊探針：模糊前後的 VQA 答案與 IQP／CLIP，輸出 `probe_gaussian.csv` |
| `vqa_backends.py` | VQA 答案後端（BLIP、Florence-2 等） |
| `analyze.py` | Stage 1 計分、分流、Stage 2、所有指標，輸出 `rows_scored.csv`、`pipeline_variants.csv`、`summary.json` |
| `run_analyze_sigma08.py` | 固定 σ=0.8 重跑 `analyze.py`，輸出到 `sigma08_fixed/`（報告用） |
| `make_tables.py` | 彙整各 `summary.json` 成 `RESULTS_tables.md` |
| `eval_vqa_victim.py` | 先導測試：非 BLIP 受害模型在乾淨圖與攻擊圖的答案差異 |

## 舊流程（dual-defense，Stage 1 依題號分 3 折）

| 腳本 | 用途 |
|---|---|
| `evaluate_best_stage1_new.py` | Stage 1：旋轉 Caption、四維特徵、OOF 風險分數，輸出 `tcvad_stage1_4_academic_dataset.csv` |
| `evaluate_stage2_gaussian.py` | Stage 2：只對中風險樣本做高斯模糊，再用 SVM 重新計分 |
| `sss_denoise.py` | SSS 甜蜜點搜尋（σ*=0.8 的出處） |
| `answer_flip_detector.py` | 答案翻轉偵測（σ_det=1.5 的出處） |
| `mitigation_summary.py` | 緩解後 ASR 彙整（Exp2 40.9% 的出處） |
| `analyze_risk_results.py` | 風險分級、乾淨圖低／中風險一致性、教學範例 |
| `run_full_pipeline.py` | 依序執行以上各步 |
| `validate_sss_reconstruction.py` | 檢查：重新推論存檔 PNG 能否重現紀錄的答案 |
| `stage2_blip_accuracy_after_blur.py` | 模糊後有多少答案等於標準答案 |

## Stage 1 正式 11/22 對照（依照片分 5 折）

| 腳本 | 用途 |
|---|---|
| `run_rotation_cos_vs_ang.py` | cosine 四維與角距四維對照（README Stage 1 AUROC 的出處） |
| `compute_rpf1.py` | 從折外預測算 AUROC、Recall、Precision、F1 |

## 補充實驗

| 腳本 | 用途 |
|---|---|
| `run_gaussian_caption_shift.py` | Exp1／Exp2 單次高斯模糊後的描述變化 |
| `run_gaussian_seq10_4d.py` | 連續 10 次高斯模糊後的四項描述一致性統計 |

產生報告與彙整表的腳本在 `實驗/程式/4_報告產生/`。
