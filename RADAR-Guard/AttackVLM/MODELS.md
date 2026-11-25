# 模型參數與用途

## BLIP 防禦分類器

`models/blip/natural/`、`models/blip/stress/` 各含 `fold_1.joblib` 至 `fold_5.joblib`、`fold_test_ids.json`、`audit.json` 和 `deployment.joblib`。五折模型用途是 `oof_replay`，只接受該折未參與訓練的圖片 ID；`deployment.joblib` 的用途是 `deployment_full_training`，以該情境的全部 2,000 筆四維特徵訓練，可交給單張防禦服務處理新圖片。

自然／壓力情境的五折重建各自與原始 OOF 結果完全一致（margin、risk 最大誤差 0；預測、分折及風險分流不一致數均為 0）。部署模型是另外訓練的檔案，**不可把五折 OOF 指標當成其獨立測試成績**。交付時已用相同環境的 `load_detector` 及一筆正式特徵檢查兩個部署檔能載入與計分；這不是 GPU 網站端到端驗證，也不表示網站目前正在載入這兩個新檔案。

分類器使用 `StandardScaler`＋RBF `SVC`（`C=1`、`gamma=scale`、`class_weight=balanced`）；四維順序為 `d_center, d_iqr, d_tail, d_q95`。SVM margin 經 sigmoid 轉成風險分數，這個分數**不是校準過的攻擊機率**。`risk<0.30` 通過，`0.30<=risk<0.80` 進雙邊濾波修復，`risk>=0.80` 阻擋。BLIP joblib 會檢查特徵順序、Clean=0／MF-tt=1 與 scikit-learn 版本；本次匯出版本為 1.6.1。只載入自己信任的 joblib 檔案。

## GIT 與 ViT-GPT2

`models/git/{natural_oof,stress_oof}/` 和 `models/vitgpt2/{natural_oof,stress_oof}/` 各有五個正式五折檔，用來重現相應實驗。另有 `models/{git,vitgpt2}/{natural,stress}/deployment.joblib` 四個全資料模型，分別使用該情境 2,000 筆正式四維特徵訓練，與五折模型分開。`scripts/train_deployment_svm.py` 會核對正式特徵 SHA-256、評估設定、1,000 組 Clean／MF-tt 配對和模型重新載入的 margin；各目錄的 `manifest.json` 記錄來源和檔案雜湊。這些全資料模型沒有獨立測試指標，不能冒用 OOF 數字。

`scripts/score_detector_features.py` 可在相容的 scikit-learn 環境，用四維特徵載入三種模型的部署 SVM 並回傳風險與分流。GIT／ViT-GPT2 的新圖片仍需先用各自的描述模型和特徵擷取程式產生同順序的四維特徵；目前網頁影像上傳流程只接 BLIP，不能僅更換 joblib 就改成 GIT／ViT-GPT2 網頁服務。

BLIP base_coco、CLIP ViT-B/32、microsoft/git-large-coco、nlpconnect/vit-gpt2-image-captioning 的預訓練權重屬第三方依賴，不在此交付的 SVM 模型參數內。重跑影像模型須依環境設定取得相容的權重與版本。
