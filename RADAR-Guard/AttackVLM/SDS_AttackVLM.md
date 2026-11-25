# SDS：AttackVLM 分支設計規格

## 範圍與輸入輸出

本文件描述 AttackVLM 的攻擊實驗與 BLIP 防禦介面，是小組系統 SDS 的 AttackVLM 補充。輸入為來源影像、配對目標影像及目標描述；攻擊輸出包含 MF-ii 中間圖、MF-tt 對抗圖和紀錄。防禦輸入為一張影像及 SVM 模型路徑；輸出包含 21 句描述、四維特徵、decision margin、風險分數、分流動作，以及中風險時的修復圖與最終描述。

## 模組流程

1. `runtime/blip/tools/run_attack_cli.py` 以正式 MF-ii、描述產生、MF-tt 腳本依序產生單張攻擊；正式批次來源另保存在 `scripts_original/`。
2. `runtime/blip/integration/formal_stage1.py` 對輸入建固定旋轉 10 組、縮放 10 組及原圖；`integration/service.py` 用 BLIP base_coco 產生 21 句描述。
3. `integration/formal_features.py` 透過 CLIP ViT-B/32 文字向量計算 `d_center, d_iqr, d_tail, d_q95`。`integration/detector.py` 核對模型檔與特徵順序，使用 SVM `decision_function`、sigmoid 風險及 0.30／0.80 門檻分流。
4. Low 保留原描述；Medium 對原輸入圖套用 `bilateral_strong`（直徑 7、`sigma_color=50`、`sigma_space=7`）再產生描述；High 回傳阻擋。旋轉縮放圖用於偵測特徵，不是第二階段修復輸入。

## 模型與部署邊界

各模型的 `fold_*.joblib` 僅供對應正式五折評估；`models/{blip,git,vitgpt2}/{natural,stress}/deployment.joblib` 是各情境全資料另訓的 SVM。`scripts/score_detector_features.py` 提供四維特徵的通用計分入口；BLIP 另有單張影像的完整防禦服務。BLIP 影像服務要求相容的 Python、scikit-learn、LAVIS、CLIP、torch 及 GPU 推論環境；`runtime/blip/environment/requirements.reference.txt` 是原環境清單，不是保證可直接安裝的完整配方。

既有網站的 `/workspace/Web/attackvlm_bridge.py` 以子程序呼叫 BLIP handoff 的 `tools/defend_cli.py` 或 `defend_pair_cli.py`，並從 `ATTACKVLM_DEPLOYMENT_MODEL` 讀取新圖片用模型路徑。網站屬小組系統目錄；這份交付不宣稱已更新其執行中環境變數或完成 GPU 網站端到端驗證。GIT／ViT-GPT2 的正式腳本、評估模型與新匯出的特徵級部署 SVM 均未接到現有 BLIP 影像上傳服務。

## 溯源與限制

正式原始腳本、結果與來源 SHA-256 放在 `scripts_original/`、`scripts/formal_runs/`、`results/`；部分腳本寫有當時主機的絕對路徑，重新執行須設定環境、路徑與上游模型權重。研究指標由各情境正式結果提供，不以新部署模型的訓練集預測冒充獨立測試結果。
