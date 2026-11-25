# 模型參數與權重說明（`6_模型參數/`）

本目錄收錄 RADAR-Guard 兩階段防禦之所有已訓練參數、超參數設定，以及大型預訓練 BLIP 權重之路徑參照表。

## 1. 本目錄實體檔案清單

| 檔案名稱 | 類型 | 說明 |
|---|---|---|
| `stage1_rbf_svm_rotate5_exp1.joblib` | Scikit-learn Pipeline (`StandardScaler` + `SVC(kernel='rbf', C=1.0, class_weight='balanced')`) | **系統部署用 Stage 1 風險分類器**，由 `2_訓練腳本/train_stage1_svm.py` 於 Exp1 `rotate_5`（2,000 列 4D 角距離特徵）訓練匯出；同步置於 `系統/1_系統完整程式/models_params/`。5-fold GroupKFold OOF AUROC = **0.6456**、F1 = **0.5762**。 |
| `stage1_rbf_svm_formal1122_exp1.joblib` | Scikit-learn Pipeline | 由 11/22 正式重跑 `rotation_cos_vs_ang/exp1_features.csv`（Angular 4D，2,000 列）訓練匯出。 |
| `stage1_rbf_svm_formal1122_exp2.joblib` | Scikit-learn Pipeline | 由 11/22 正式重跑 `rotation_cos_vs_ang/exp2_features.csv`（Angular 4D，2,000 列）訓練匯出。 |
| `model_metadata.json` | JSON | 上述三個 `.joblib` 模型之訓練環境版本（`scikit-learn 1.6.1`、`joblib 1.6.0`）、特徵欄位、StandardScaler mean/scale、Support Vectors 數量與 5-fold GroupKFold 指標。 |
| `radar_guard_params.json` | JSON | 系統與實驗完整超參數（BSA PGD `ε=0.125`, `α=0.002`, `steps=100`；Stage 1 `k=10`, `±5°`, `0.30/0.70` 分流；Stage 2 `5×5` Gaussian `σ*=0.8`, `σ_det=1.5`）。 |
| `sss_best_theta.json` | JSON | Stage 2 SSS 網格搜尋最佳參數輸出（來自 `exp2_success1000`：`kernel=5`, `sigma=0.8`, `sss=0.5766`）。 |
| `theta_det.json` | JSON | Stage 2 答案翻轉偵測最佳單一探針輸出（來自 `steps100_alpha0.002_run1000`：`kernel=5`, `sigma=1.5`, `TPR=0.4066`, `FPR=0.0532`, `TPR_fooled_only=0.8303`）。 |
| `checkpoints_manifest.json` | JSON | 外部大型 BLIP 權重（4.46 GB）與 HuggingFace 模型快取之路徑及容量對照。 |

## 2. 外部大型預訓練權重（不重複複製，合計 4.46 GB）

系統執行 `vlattack_engine.py` 時會自動依序搜尋 `VLATTACK_CHECKPOINTS_DIR`、`VLATTACK_ROOT` 及本機既有目錄 `C:\Users\tracy\github\VLAttack\BLIP_attack\checkpoints\`：

- `model_base_14M.pth`（2,115,792,294 bytes / 2.12 GB）：BLIP 預訓練模型（VLAttack BSA PGD 白盒代理梯度來源）
- `model_vqa.pth`（1,446,244,375 bytes / 1.45 GB）：BLIP-VQA 受害模型（`inference="rank", k_test=128`）
- `model_base_caption_capfilt_large.pth`（896,081,425 bytes / 896 MB）：BLIP Caption 解碼器（Stage 1 產生 11 句影像描述）
- HuggingFace 快取（`C:\Users\tracy\.cache\huggingface\hub\`）：`sentence-transformers/all-MiniLM-L6-v2`、`bert-base-uncased`

> **版本相容性提醒**：上述 `.joblib` 係以系統執行環境 `VLAttack_Web\.venv`（`scikit-learn==1.6.1`）匯出；若改於實驗環境 `VLAttack\vlattack_env`（`scikit-learn==1.7.2`）載入，建議執行 `python 2_訓練腳本/train_stage1_svm.py` 重新匯出以消除版號警告，或依賴 `vlattack_engine.py` 內建的 CSV 自動擬合備援機制。
