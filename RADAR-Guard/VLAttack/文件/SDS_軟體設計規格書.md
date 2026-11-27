# RADAR-Guard 系統軟體設計規格書（Software Design Specification, SDS）

**系統名稱**：RADAR-Guard 多模態視覺語言對抗攻防與雙階段防禦平台（VLAttack 模組）  
**版本**：v1.0（對應 `系統/1_系統完整程式/`）  
**最後更新**：2026-11-25

---

## 1. 系統概述與設計目標

### 1.1 系統定位
RADAR-Guard 是一套針對視覺語言模型（Vision-Language Models, VLM）視覺問答任務（VQA）所設計之**即時對抗攻防驗證與雙階段防禦系統**。本系統聚焦於 **VLAttack（Block-wise Similarity Attack, BSA）** 純視覺像素擾動威脅，提供網頁視覺化平台（Flask）與命令列工具（CLI）兩種操作介面，實現以下核心能力：

1. **即時對抗攻擊重現**：保持提問文字不變，對輸入圖像施加 L∞ 有界白盒 PGD 擾動，並以黑盒 BLIP-VQA 驗證回答是否遭誤導。
2. **Stage 1 幾何轉換風險分流（TC-VAD）**：利用對抗擾動在微小空間旋轉（±5°）下的語意描述不穩定性，以四維角距離特徵與 RBF-SVM 輸出風險分數 `r ∈ (0, 1)`，將請求分流至「低風險放行 / 中風險修復 / 高風險攔截」。
3. **Stage 2 甜蜜點高斯修復與翻轉偵測（SSS + Answer-Flip）**：針對中風險樣本施加 `5×5` 二維高斯模糊（修復參數 `σ* = 0.8`），同時搭配強探針（`σ_det = 1.5`）比對回答翻轉行為。
4. **實驗案例重播與跨模型結果展示**：提供 7 組預計算案例（含低／中／高風險與乾淨圖對照）及 BLIP / Florence 特徵距離對照報表。

### 1.2 系統邊界與排除項目
- 本系統僅涵蓋 **VLAttack + RADAR-Guard** 完整管線；專案另一攻擊模組 AttackVLM 由團隊其他成員負責，已自本交付系統程式中完整抽離。
- 系統可在 **GPU（CUDA）** 或 **CPU（`VLATTACK_DEVICE=cpu`）** 模式下運作，亦支援透過 `VLATTACK_PYTHON` 呼叫外部推論子程序。

---

## 2. 系統架構與目錄模組說明

```text
系統/1_系統完整程式/
├── app.py                     # Flask Web 伺服器與 REST API 路由層（port 5005）
├── vlattack_live.py           # 即時三階段任務協調層（Attack -> Detect -> Defend）與 Job 持久化
├── vlattack_engine.py         # PyTorch / BLIP / MiniLM / SVM 模型載入與推論核心引擎
├── vlattack_gpu_worker.py     # 無 GPU Web 環境下呼叫外部 CUDA Python 的 stdin/stdout worker
├── vlattack_bridge.py         # 7 個預先計算展示案例與 METHOD_EFFECTS 載入橋接層
├── radar_guard_cli.py         # 單機 CLI 介面（狀態查詢、展示案例查詢、單圖端到端推論）
├── models_params/
│   └── stage1_rbf_svm_rotate5_exp1.joblib   # 預訓練 Stage 1 StandardScaler + RBF-SVM 模型
├── BLIP_attack/               # BLIP-pretrain / BLIP-VQA / BLIP-caption 模型與前處理定義
├── cleverhans/                # BSA PGD 對抗攻擊實作（projected_gradient_descent）
├── data/vlattack_demo/        # 7 個展示案例之中繼資料 catalog.json
├── static/                    # CSS 樣式、流程圖、7 個案例對照圖與即時結果輸出區（static/results/live/）
└── templates/                 # Jinja2 HTML 頁面（index, vlattack_live, vlattack_cases, vlattack, results）
```

---

## 3. 核心流程與資料流設計

### 3.1 即時攻防三階段資料流（`vlattack_live.py` + `vlattack_engine.py`）

```text
[輸入影像 + 問題字串]
   │
   ▼  Step 1: 攻擊生成 (run_attack / engine.attack)
   ├─ 影像縮放至 480×480、Normalize(mean=0.5, std=0.5) 轉為 [-1, 1] 張量
   ├─ 白盒代理：BLIP-pretrain (model_base_14M.pth) 提取多層特徵
   ├─ CleverHans BSA PGD (ε=0.125, α=0.002, steps=100, L∞, clip=[-1, 1])
   └─ 黑盒受害者：BLIP-VQA (model_vqa.pth, inference="rank", k_test=128)
      計算 original_answer、adversarial_answer、attack_success、SSIM 與殘差熱圖
   │
   ▼  Step 2: Stage 1 幾何一致性風險偵測 (run_detect / engine.score)
   ├─ 幾何轉換：保留原圖 1 張 + 以固定隨機種子對 ±5° 進行 k=10 次旋轉（共 11 張視角）
   ├─ 影像描述：BLIP-caption (model_base_caption_capfilt_large.pth, num_beams=3) 產出 11 句描述
   ├─ 語意嵌入：all-MiniLM-L6-v2 (384 維, mean pooling + L2 normalize)
   ├─ 角距離計算：θ_i = arccos(clip(e_i · c, -1, 1)) / π，其中 c 為質心單位向量
   ├─ 4D 統計特徵：[d_center (mean), d_iqr (Q75-Q25), d_tail (Q90-Q50), d_q95 (Q95)]
   └─ SVM 風險分數：由預訓練 .joblib（StandardScaler + RBF SVC, C=1.0, balanced）
      計算決策值 margin，並以 r = 0.5 * tanh(margin / 2.0) + 0.5 映射至 (0, 1)
   │
   ▼  Step 3: 風險分流與 Stage 2 高斯修復/偵測 (run_defend)
   ├─ r ≤ 0.30  ──► Low Risk  ──► 直接放行，輸出原問答結果
   ├─ r ≥ 0.70  ──► High Risk ──► 直接攔截，拒絕推論
   └─ 0.30 < r < 0.70 ──► Mid Risk
        ├─ 修復分支：5×5 Gaussian Blur (σ* = 0.8) → 送入 BLIP-VQA 得修復後答案 kept_answer
        └─ 探針分支：5×5 Gaussian Blur (σ_det = 1.5) → 比對 probe_answer != adv_answer（翻轉偵測）
```

### 3.2 預訓練模型與權重解析優先順序（`vlattack_engine.py`）

1. **Stage 1 RBF-SVM 分類器**：
   - 依序嘗試載入環境變數 `VLATTACK_SVM_MODEL`、`系統/1_系統完整程式/models_params/stage1_rbf_svm_rotate5_exp1.joblib`、`實驗/模型/stage1_rbf_svm_rotate5_exp1.joblib`。
   - 若 `.joblib` 不存在或因 scikit-learn 版號差異無法載入，系統自動退回讀取 `tcvad_stage1_4_academic_dataset.csv` 並以 `rotate_5`（2,000 列）即時擬合相同超參數之 `Pipeline(StandardScaler(), SVC(...))`。
2. **BLIP 大型預訓練權重（4.46 GB）**：
   - 依序搜尋 `VLATTACK_CHECKPOINTS_DIR`、`VLATTACK_ROOT/BLIP_attack/checkpoints`、本機預設路徑 `C:\Users\tracy\github\VLAttack\BLIP_attack\checkpoints\`，自動解析 `model_base_14M.pth`、`model_vqa.pth`、`model_base_caption_capfilt_large.pth`。

---

## 4. 介面與 API 規格（`app.py`）

### 4.1 網頁頁面路由（HTTP GET）

| 路由 | 模板 | 功能說明 |
|---|---|---|
| `/`、`/index.html` | `index.html` | 平台導覽首頁：說明 VLAttack 攻擊模式與 RADAR-Guard 雙階段防禦架構 |
| `/vlattack-live`、`/vlattack-live.html` | `vlattack_live.html` | 即時攻防流程頁：支援上傳圖片或使用預設範例，逐步執行 Attack / Stage 1 / Stage 2 |
| `/vlattack-cases`、`/vlattack-cases.html` | `vlattack_cases.html` | 7 組預計算案例重播頁：展示乾淨圖與不同風險區間（C1～C6）的真實 BLIP 問答與修復結果 |
| `/results`、`/results.html` | `results.html` | 實驗數據對照頁：比較 BLIP 與 Florence 特徵距離在三個實驗情境下之 ASR 與誤判率 |
| `/vlattack` | *(Redirect)* | 重新導向至 `/vlattack-live` |

### 4.2 REST API 端點

| 端點 | 方法 | 輸入格式 | 回傳關鍵欄位（JSON） |
|---|---|---|---|
| `/api/vlattack/catalog` | GET | 無 | `attack`, `pipeline`, `metrics`, `samples`（7 筆）, `method_effects` |
| `/api/vlattack/sample/<sample_id>` | GET | Path 參數 | 該案例的 `ori_url`, `atk_url`, `gaussian_mild_url`, `gaussian_strong_url`, 問答與四維特徵 |
| `/api/vlattack/live/status` | GET | 無 | `cuda`, `gpu_name`, `device`, `loaded` |
| `/api/vlattack/live/attack` | POST | `multipart/form-data` (`image`, `question`, `eps`, `alpha`, `steps`) | `job_id`, `original_url`, `adv_url`, `diff_url`, `ssim`, `history`, `original_answer`, `adversarial_answer`, `attack_success` |
| `/api/vlattack/live/detect` | POST | JSON `{"job_id": "..."}` | `job_id`, `attack` (`risk_score`, `risk_bin`, `action`, `d_center`, `d_iqr`, `d_tail`, `d_q95`), `clean` |
| `/api/vlattack/live/defend` | POST | JSON `{"job_id": "..."}` | `job_id`, `purified_url`, `ssim_to_clean`, `flip_probe`, `probe_answer`, `kept_answer`, `end_to_end_flagged`, `decision` |

### 4.3 單機命令列介面（`radar_guard_cli.py`）
- `--status`：檢查裝置（CUDA/CPU）與各項模型權重、`.joblib`、CSV 解析狀態。
- `--list-cases` / `--demo-case <id>`：查詢 7 個展示案例資料。
- `--image <path> --question "<text>" --mode [detect|defend|attack_and_defend]`：於命令列直接執行偵測、修復或完整攻防流程並輸出 JSON。

---

## 5. 系統狀態持久化與執行緒安全
- **執行緒安全**：`vlattack_engine.py` 使用全域 `threading.Lock()`（`_LOCK`）保護單例 `ExperimentEngine`，避免多請求並發載入模型或執行 PGD 梯度計算時發生顯存衝突或亂數種子汙染（固定 `seed=42`）。
- **任務快取持久化**：`vlattack_live.py` 以 `uuid4().hex[:12]` 產生 `job_id`，將中間影像（`_ori.png`, `_atk.png`, `_diff.png`, `_sss.png`）與中繼結果寫入 `static/results/live/<job_id>.json`，確保即使在 Flask reloader 或子程序呼叫模式下，後續 `/detect` 與 `/defend` 請求皆可正確讀取狀態。
