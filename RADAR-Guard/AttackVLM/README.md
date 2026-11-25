# AttackVLM 交付資料

本目錄是 RADAR-Guard 畢業專題中的 AttackVLM 分支，涵蓋 BLIP、ViT-GPT2 與 GIT-large-COCO 的攻擊、偵測、防禦和三種情境的正式紀錄。VLAttack 與網頁平台由小組其他目錄交付。

## 從哪裡開始

- [MU.md](MU.md)：資料驗證、環境、BLIP 單張攻擊／防禦與模型重建的操作步驟。
- [SDS_AttackVLM.md](SDS_AttackVLM.md)：本分支的模組、資料流與網頁介面。
- [DATASETS.md](DATASETS.md)：四個資料 ZIP 的組成與情境定義。
- [MODELS.md](MODELS.md)：SVM 評估模型、部署模型及外部預訓練權重的區別。

| 位置 | 內容 |
| --- | --- |
| `runtime/blip/` | BLIP 單張攻擊、防禦與分類器匯出工具；包含正式參考特徵及原腳本。 |
| `scripts_original/` | BLIP、ViT-GPT2、GIT 的正式來源腳本快照及來源雜湊清單。 |
| `scripts/formal_runs/` | GIT／ViT-GPT2 正式輸出中保存的攻擊、描述、特徵與評估腳本。 |
| `models/blip/` | 自然、壓力情境各五個五折模型及一個另行訓練的部署模型。 |
| `models/git/`、`models/vitgpt2/` | 各情境五折評估模型，以及自然、壓力情境各一個全資料部署 SVM。 |
| `scripts/train_deployment_svm.py`、`scripts/score_detector_features.py` | GIT／ViT-GPT2 部署 SVM 匯出及四維特徵計分工具。 |
| `results/` | BLIP、GIT、ViT-GPT2 的正式特徵、清單與主要結果。 |
| `data/` | 正常圖、攻擊圖、來源圖與目標圖的 ZIP，以及 ZIP SHA-256。 |

此交付保留了原實驗腳本及結果；部分原始 JSON 和 GIT／ViT-GPT2 正式腳本含當時遠端主機的絕對路徑。移到別台電腦重跑時須按 [MU.md](MU.md) 調整路徑和依賴。第三方 BLIP、CLIP、GIT、ViT-GPT2 預訓練大模型權重不在此目錄；交付的模型參數是本專題訓練的 SVM。報告中的五折數值只對應五折評估模型，不代表另行訓練的部署模型成效。
