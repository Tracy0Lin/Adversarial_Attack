# MU：AttackVLM 操作手冊

以下命令以 Linux 終端機為例；在本目錄執行。`$PY` 應指向與模型相容的實驗環境。既有老師主機使用 `/workspace/projects/couser/environments/attackvlm/bin/python`（Python 3.9、scikit-learn 1.6.1）。`runtime/blip/environment/requirements.reference.txt` 記錄原環境版本，但不是新機器的一鍵安裝腳本。BLIP／CLIP 等預訓練權重須由相容上游來源取得；本交付只含專題訓練的 SVM 權重。

## 1. 驗證並解壓資料

```bash
cd /path/to/Adversarial_Attack/RADAR-Guard/AttackVLM
(cd data && sha256sum -c SHA256SUMS.txt)
mkdir -p extracted/shared extracted/blip extracted/vitgpt2 extracted/git
unzip data/shared_inputs.zip -d extracted/shared
unzip data/blip_dataset.zip -d extracted/blip
unzip data/vitgpt2_dataset.zip -d extracted/vitgpt2
unzip data/git_dataset.zip -d extracted/git
```

資料夾與情境對應見 [DATASETS.md](DATASETS.md)。原始 manifest 和某些 JSON 是實驗主機的絕對路徑，移機後請改用解壓後相對路徑。

## 2. BLIP 模型重建與 CPU 檢查

```bash
export PY=/workspace/projects/couser/environments/attackvlm/bin/python
$PY runtime/blip/tools/export_detector_models.py \
  --data-root runtime/blip/reference_data/project \
  --output /tmp/blip_model_reexport \
  --scenario both --include-deployment-model
```

輸出目錄必須不存在。工具以 image ID 配對重建自然／壓力情境五折，與保存的 OOF 預測、margin、risk、分流比對通過後才匯出；另產生兩個全資料部署模型。原始模型已放在 `models/blip/`，一般使用無須重新訓練。部署模型沒有獨立測試成績，不得引用五折 OOF 數字作為其成績。`runtime/blip/integration/detector.py` 可在 CPU 上載入可信的 BLIP joblib；模型與特徵順序、版本不符時會拒絕執行。

## 3. 單張 BLIP 攻擊（需要 GPU）

```bash
$PY runtime/blip/tools/run_attack_cli.py \
  --source-image /path/to/source.png \
  --target-image /path/to/paired_target.png \
  --target-caption '目標英文描述' \
  --output /tmp/attackvlm_one_case
```

來源圖、目標圖與描述應依 `shared_inputs.zip` 的原始配對清單選取。輸出目錄必須不存在。工具依序執行 MF-ii、描述產生、MF-tt，並在輸出目錄保存 `commands.json`、每段日誌和圖片；不是重新訓練 BLIP。預設正式攻擊參數見 `runtime/blip/README_給網頁開發者.md`。

## 4. 單張 BLIP 防禦（需要 GPU）

```bash
$PY runtime/blip/tools/defend_cli.py \
  --image /path/to/input.png \
  --model models/blip/natural/deployment.joblib \
  --repaired-output /tmp/attackvlm_repaired
```

使用壓力情境部署模型時改傳 `models/blip/stress/deployment.joblib`。回傳 JSON 含四維特徵、風險、分流、原描述與最終描述。正式案例的五折重播須使用同情境、同一張圖的 `--scenario natural|stress --image-id ID`；工具按 `runtime/blip/artifacts/` 內對應測試 ID 選折。不能把五折重播模型拿去處理任意新圖。

## 5. 批次實驗腳本與結果

BLIP 的資料前處理、21 句描述與 CLIP 特徵腳本在 `scripts_original/formal_tests/paper_coco1000_v1/tcvad/experiment1_blip/scripts/`；五折訓練／評估在同目錄的 `evaluate_blip_rbf_svm_group5fold.py`，以及 `runtime/blip/tools/export_detector_models.py`。MF-ii／MF-tt 的正式來源另在 `scripts_original/work_code/`。

GIT、ViT-GPT2 的攻擊與 21 句描述、特徵、五折評估腳本保存在 `scripts/formal_runs/{git,vitgpt2}/`，原始啟動腳本另在 `scripts_original/work_code/`，正式結果在 `results/{git,vitgpt2}/`。這些是原主機執行快照，仍有絕對路徑；在新機器批次重跑前須先依資料 ZIP、環境和 checkpoint 修正路徑。五折模型是評估產物，另訓的部署 SVM 位於 `models/{git,vitgpt2}/{natural,stress}/`。

## 6. GIT／ViT-GPT2 模型訓練與四維特徵計分（CPU）

交付目錄已含四個部署模型。若需重新匯出，指定新的輸出根目錄，以免覆蓋交付檔：

```bash
$PY scripts/train_deployment_svm.py --release-root . \
  --output-root /tmp/attackvlm_reexport_models
```

腳本核對四份正式特徵與評估設定後，分別訓練 2,000 筆全資料 SVM，並重新載入驗證。`scripts/score_detector_features.py` 使用各 VLM 產生的 21 句描述所擷取的四維特徵；下例的 `0.1` 只是介面測試值，不是實驗數據：

```bash
$PY scripts/score_detector_features.py \
  --model models/git/natural/deployment.joblib \
  --d-center 0.1 --d-iqr 0.1 --d-tail 0.1 --d-q95 0.1
```

ViT-GPT2 改傳 `models/vitgpt2/natural/deployment.joblib`。這是特徵級模型接口；從新圖片產生 GIT／ViT-GPT2 描述與特徵的批次腳本仍須依新環境調整路徑。現有網頁只提供 BLIP 的影像級單張防禦流程。

## 7. 網頁交接

現有網頁的 AttackVLM bridge 以 `ATTACKVLM_BUNDLE` 指向 BLIP runtime 根目錄，以 `ATTACKVLM_PYTHON` 指向 GPU Python，以 `ATTACKVLM_DEPLOYMENT_MODEL` 指向經確認的 BLIP 部署 joblib，並以 `ATTACKVLM_RESULTS` 指定輸出位置。設定或更換網站模型後仍需在實際 GPU 網頁上測試攻擊、防禦、圖片與 JSON 回傳。本目錄的 CPU 檢查只驗證模型檔可載入與計分，未代替端到端測試。
