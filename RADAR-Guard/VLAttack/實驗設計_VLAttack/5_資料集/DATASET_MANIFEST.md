# 實驗設計資料集清單（`5_資料集/`）

本目錄整理 VLAttack 實驗所使用之標註檔、良性影像、成對對抗影像及實驗特徵與評測表格（合計約 206 MB）。

## 1. 目錄內容明細

| 子目錄 / 檔案 | 內容與筆數 | 大小 | 原始來源 |
|---|---|---|---|
| `annotations/vqa_val.json` | VQAv2 驗證集問題與 ground truth | 51.7 MB | `VLAttack\BLIP_attack\annotation\vqa_val.json` |
| `annotations/answer_list.json` | VQA 候選答案字典（BLIP rank 推論用） | 34 KB | `VLAttack\BLIP_attack\annotation\answer_list.json` |
| `annotations/right_vqa_*.txt` | `prepare_vqa.py` 篩出 BLIP 原本答對的題目編號與答案表（含 200 題與完整表） | ~1.5 MB | `VLAttack\BLIP_attack\right_vqa_*.txt` |
| `benign_val2014_300/` | 300 張 COCO `val2014` 良性原圖（`.jpg`） | 46.6 MB | `VLAttack\BLIP_attack\data\VQAv2\val2014\` |
| `paired_samples/smoke20/` | 20 組完整 BSA PGD 煙霧測試良性與對抗成對影像 | ~8 MB | `VLAttack\BLIP_attack\runs\IOPGD-s100-a0002-smoke20\visual_results\` |
| `paired_samples/deep_dive_blip/` | 82 組依風險分流（C1/C2/C4/C5/C6）抽樣之 `_1_ori.png` 與 `_2_atk.png` 成對圖（181 檔） | 60.5 MB | `VLAttack_final\BLIP_attack\output\VQA\stage_routing_deep_dive\BLIP_victim\` |
| `paired_samples/web_demo_7cases/` | 系統 7 個代表案例（各含 `ori.png`, `atk.png`, `gaussian_mild.png`, `gaussian_strong.png` + `catalog.json`） | ~15 MB | `Web\static\vlattack_demo\` + `catalog.json` |
| `tabular_features_and_logs/exp1_mixed1000/` | Exp1（1,000 題自然混合 ASR 46.2%，2,000 列）`tcvad_stage1_4_academic_dataset.csv`、`dataset_manifest.csv`、`risk_tables/` | ~1 MB | `VLAttack_final\BLIP_attack\output\VQA\exp1_mixed1000\` |
| `tabular_features_and_logs/exp2_success1000/` | Exp2（1,000 對純成功攻擊 + 1,000 良性，2,000 列）特徵表、SSS 網格搜尋、Flip 偵測、緩解彙整 | ~2 MB | `VLAttack_final\BLIP_attack\output\VQA\exp2_success1000\` |
| `tabular_features_and_logs/steps100_alpha0.002_run1000/` | 1,000 題原始攻擊統計（`attack_rate_summary.json`、`EXPERIMENT_REPORT.md`、`theta_det.json` 等） | ~2 MB | `VLAttack_final\BLIP_attack\output\VQA\steps100_alpha0.002_run1000\` |
| `tabular_features_and_logs/multi_captioner_blip/` | Exp1/Exp2/Exp3（BLIP Caption 基準與 `attack_log_snapshot_2298.json` 2,298 題攻擊紀錄） | ~16 MB | `VLAttack_final\BLIP_attack\output\VQA\multi_captioner\` |
| `tabular_features_and_logs/rotation_cos_vs_ang/` | 11/22 正式 Cosine 4D vs Angular 4D 重跑特徵表、`oof_scores.csv`、`svm_summary.csv`、`bootstrap_delta.csv`、`formal_1122_rpf1.csv` | ~1.6 MB | `VLAttack_final\BLIP_attack\output\VQA\rotation_cos_vs_ang\` + box 歸檔 |

## 2. 未重複收錄之大型檔案說明

1. **`val2014.zip`（6.64 GB）**：為原始 COCO val2014 全量壓縮包，因 `C:` 剩餘空間有限，未重複複製；實際實驗用之 300 張原圖已完整複製於 `benign_val2014_300/`。
2. **全量攻擊輸出 PNG 目錄**：`exp1_mixed1000/visual_results`、`exp2_success1000/visual_results` 各 1,000 對（2,000 張）已上傳為 GitHub Release `vlattack-data-v1` 的兩個 zip（見 §3）；`steps100_alpha0.002_run1000/visual_results`（約 16 GB）未上傳。本目錄保留其結構化日誌（`attack_log_snapshot_2298.json`、`dataset_manifest.csv`）與 109 對代表性成對樣本影像。

## 3. 後續補充檔案（2026-11-27）

| 路徑 | 內容 | 大小 |
|---|---|---|
| `question_difficulty/labels.csv` | 2,538 題困難度標註（easy 1,288 / medium 598 / hard 652；欄位 `question_id, question, n_question_texts, experiments, image_path, image_ok, difficulty_3, reason, difficulty_binary`） | 661,269 bytes |
| `question_difficulty/samples.csv` | 2,538 題取樣清單 | 約 0.49 MB |
| `question_difficulty/c1c7_by_difficulty.csv` | 各研究／實驗／C1–C7 類別依困難度的題數彙整 | 2,516 bytes |
| `experiment_manifests/exp1_mixed1000_dataset_manifest.csv`、`exp2_success1000_dataset_manifest.csv` | 實驗一／二每題的原始預測、對抗預測、攻擊是否成功、SSIM、LPIPS、CLIP | 各約 0.11 MB |
| `data_supplement/` | 實驗資料補充（42 個 CSV、20 張 PNG、7 個 MD） | 5,765,740 bytes |
| `release_zips/SHA256SUMS.txt` | Release 兩個 zip 的 SHA-256 | 202 bytes |

**Release 圖片（不在 repo 內）**：`exp1_mixed1000_visual_results.zip`（733,719,871 bytes）、`exp2_success1000_visual_results.zip`（738,985,508 bytes），下載與校驗方式見上層 `README.md` 的「資料與權重下載」。