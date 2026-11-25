# RADAR-Guard 系統操作手冊（Maintenance & User Manual, MU）

**適用對象**：指導教授、專案成員、系統維運與測試人員  
**程式目錄**：`系統/1_系統完整程式/`

---

## 1. 系統環境需求

- **作業系統**：Windows 10/11（本機實測環境）或 Linux
- **Python 版本**：`Python 3.10.11`
- **套件相依檔**：根目錄 `requirements.txt`（複製自實測環境 `VLAttack_Web\.venv`）
  - 核心套件：`Flask==3.1.3`、`torch==2.2.2`、`torchvision==0.17.2`、`transformers==4.26.1`、`timm==0.4.12`、`fairscale==0.4.4`、`scikit-learn==1.6.1`、`scikit-image==0.22.0`、`joblib==1.6.0`、`pandas==2.3.3`、`numpy==1.26.4`、`lpips==0.1.4`
- **硬體建議**：
  - 靜態頁面與展示案例重播（`/results`、`/vlattack-cases`）：純 CPU 即可流暢運行，無須載入大型模型。
  - 即時 BSA PGD 攻擊與 BLIP 推論（`/vlattack-live`）：建議配備 NVIDIA GPU（CUDA）；若無 GPU，可設定環境變數 `VLATTACK_DEVICE=cpu` 以 CPU 模式運行（單張 100 步 PGD 耗時較長）。

---

## 2. 安裝與外部權重準備

### 2.1 若使用本機既有虛擬環境（推薦，免重新下載套件）
本機 `LAPTOP-3AKL7OUA` 已備妥相容虛擬環境：
- 系統 Web 環境：`C:\Users\tracy\github\VLAttack_Web\.venv\Scripts\python.exe`

### 2.2 若於全新機器安裝環境
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r ..\..\requirements.txt
```

### 2.3 大型預訓練權重路徑設定（4.46 GB）
`vlattack_engine.py` 會自動偵測本機預設目錄 `C:\Users\tracy\github\VLAttack\BLIP_attack\checkpoints\` 內的三個權重檔：
- `model_base_14M.pth`（2.12 GB）
- `model_vqa.pth`（1.45 GB）
- `model_base_caption_capfilt_large.pth`（896 MB）

若權重存放於其他路徑，請於啟動前設定環境變數：
```powershell
$env:VLATTACK_CHECKPOINTS_DIR = "D:\your_path\checkpoints"
```
若無 NVIDIA GPU 欲於本機以 CPU 測試完整模型載入與推論，請加上：
```powershell
$env:VLATTACK_DEVICE = "cpu"
```

---

## 3. 啟動與操作流程

### 3.1 啟動 Flask Web 平台
切換至 `系統/1_系統完整程式/` 目錄後啟動伺服器：

```powershell
cd C:\Users\tracy\github\RADAR-Guard_專題成果_20261125\系統\1_系統完整程式
C:\Users\tracy\github\VLAttack_Web\.venv\Scripts\python.exe app.py
```

啟動後請開啟瀏覽器前往：**`http://127.0.0.1:5005/`**

### 3.2 Web 平台功能操作說明

1. **首頁（`/`）**：瀏覽 VLAttack 威脅情境與 RADAR-Guard（TC-VAD + SSS 高斯去雜訊）整體流程圖，點擊上方導覽列或下方按鈕進入各功能模組。
2. **案例示範（`/vlattack-cases`）**：
   - 從左側清單挑選 7 個代表案例（例如 `G_attack_mid`、`A_clean_low`、`D_attack_high` 等）。
   - 檢視「原圖 / 對抗圖 / 高斯濾波修復圖」、4D 角距離統計、SVM 風險分數 `r` 與實際 BLIP-VQA 回答變化。
3. **實驗結果（`/results`）**：
   - 可切換 **BLIP**、**Florence** 與 **綜合比較** 分頁，查看三個獨立實驗情境（Exp1/Exp2/Exp3）之初始 ASR、兩階段後 ASR、下降百分點與良性誤判率。
4. **即時流程（`/vlattack-live`）**：
   - **步驟 1（上傳與攻擊）**：上傳圖片（或選用系統預設圖）、輸入 VQA 問題（例如 `"what color is the cat?"`），設定 `eps=0.125`、`alpha=0.002`、`steps=100`（測試可酌降 `steps`），點擊執行攻擊；畫面將顯示原圖、對抗圖、放大殘差熱圖、SSIM 與前後回答。
   - **步驟 2（Stage 1 偵測）**：點擊風險偵測，後端對影像執行 10 次 ±5° 旋轉 + 原圖共 11 句 BLIP Caption，計算 4D 角距離與風險分數 `r`，顯示 `Low (≤0.30)`、`Mid (0.30~0.70)` 或 `High (≥0.70)` 分流結果。
   - **步驟 3（Stage 2 修復與翻轉驗證）**：點擊防禦後，系統輸出 `σ*=0.8` 修復影像、`kept_answer`、`σ_det=1.5` 探針翻轉結果與最終端到端放行／攔截決策。

---

## 4. 命令列工具（CLI）操作說明

`radar_guard_cli.py` 提供無須啟動瀏覽器的快速檢查與推論介面：

```powershell
# 1. 檢查裝置、權重與 .joblib 路徑狀態
C:\Users\tracy\github\VLAttack_Web\.venv\Scripts\python.exe radar_guard_cli.py --status

# 2. 列出 7 個展示案例 ID
C:\Users\tracy\github\VLAttack_Web\.venv\Scripts\python.exe radar_guard_cli.py --list-cases

# 3. 讀取指定案例詳細 JSON
C:\Users\tracy\github\VLAttack_Web\.venv\Scripts\python.exe radar_guard_cli.py --demo-case G_attack_mid

# 4. 對單張圖片執行 Stage 1 偵測 + Stage 2 修復（若無 GPU 請加設 $env:VLATTACK_DEVICE="cpu"）
C:\Users\tracy\github\VLAttack_Web\.venv\Scripts\python.exe radar_guard_cli.py --image .\static\images\cat.png --question "what animal is this?" --mode defend
```

---

## 5. 快速煙霧測試（Smoke Test）與常見問題排除

### 5.1 無 GPU 環境下驗證網站與模型參數是否正常
```powershell
C:\Users\tracy\github\VLAttack_Web\.venv\Scripts\python.exe -c "import app; c=app.app.test_client(); print([(r, c.get(r).status_code) for r in ['/','/vlattack-live','/vlattack-cases','/results','/api/vlattack/catalog']])"
```
預期所有路由皆回傳 HTTP `200`。

### 5.2 常見問題排除
| 狀況 | 原因與排除方式 |
|---|---|
| 執行 `/api/vlattack/live/attack` 回傳「這個 Python 沒有 CUDA」 | 本機無可用 GPU 或使用 CPU 版 PyTorch；請於啟動前設定 `$env:VLATTACK_DEVICE="cpu"`，或指定具備 CUDA 之環境路徑。 |
| 找不到 `model_base_14M.pth` 等權重 | 請確認 `C:\Users\tracy\github\VLAttack\BLIP_attack\checkpoints\` 存在，或設定 `VLATTACK_CHECKPOINTS_DIR` 指向權重所在資料夾。 |
| 載入 `.joblib` 出現 `InconsistentVersionWarning` | 請使用與 `.joblib` 相同之 `scikit-learn 1.6.1` 環境（`VLAttack_Web\.venv`），或執行 `實驗設計_VLAttack\2_訓練腳本\train_stage1_svm.py` 重新匯出目前環境相容版本；`vlattack_engine.py` 亦會自動退回由 CSV 重新擬合，不影響執行。 |
| HuggingFace 下載 `all-MiniLM-L6-v2` 失敗（離線環境） | 請確認 `C:\Users\tracy\.cache\huggingface\hub\models--sentence-transformers--all-MiniLM-L6-v2` 存在，並可設定 `$env:HF_HUB_OFFLINE="1"`。 |
