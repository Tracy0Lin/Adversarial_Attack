# RADAR-Guard 多模態對抗攻防與雙階段防禦系統 — 專題成果資料夾（VLAttack 模組）

本資料夾為依 **複製（Copy-Only）** 原則整理之專題成果交付包，涵蓋 Tracy Lin 負責之 **VLAttack 視覺模態對抗攻擊與 RADAR-Guard（TC-VAD 幾何一致性偵測 + SSS 高斯去雜訊修復）雙階段防禦** 完整系統程式、軟體設計規格書（SDS）、操作手冊（MU）、實驗前處理／訓練／評估腳本、良性與對抗資料集，以及實際訓練匯出之模型參數。

> **團隊分工說明**：本專案雙線攻擊架構中的 **AttackVLM（白盒語意扭曲）** 部分由團隊其他成員負責，故本成果資料夾僅收錄與說明 **VLAttack** 實驗設計及對應之 RADAR-Guard 系統，不包含 AttackVLM 之程式、資料與文件。

---

## 目錄結構總覽

```text
RADAR-Guard_專題成果_20261125/
├── README.md                              # 本總覽文件（結構、來源對照、缺漏與外部權重說明）
├── requirements.txt                       # 系統網站與推論環境套件清單（來自 VLAttack_Web\.venv，Python 3.10.11 / PyTorch 2.2.2 / sklearn 1.6.1）
├── requirements_experiment.txt            # 實驗評估環境套件清單（來自 VLAttack\vlattack_env，Python 3.10.11 / PyTorch 1.13.1 / sklearn 1.7.2）
├── environment.yaml                       # Conda 實驗環境定義（複製自 VLAttack_final\environment.yaml）
├── 系統/
│   ├── 1_系統完整程式/                     # 可獨立執行之 Flask Web 平台 + CLI（已移除 AttackVLM 模組與頁面）
│   │   ├── app.py                         # Flask 平台入口（port 5005，含即時攻防 / 案例示範 / 實驗結果 API）
│   │   ├── vlattack_engine.py             # 核心推論引擎（BSA PGD、BLIP-VQA、11句旋轉 Caption、MiniLM 4D 角距離、優先載入 .joblib SVM、5×5 高斯濾波）
│   │   ├── vlattack_live.py               # 即時三階段執行與狀態持久化
│   │   ├── vlattack_bridge.py             # 7 個案例示範資料橋接
│   │   ├── vlattack_gpu_worker.py         # 跨環境 GPU 子程序呼叫器
│   │   ├── radar_guard_cli.py             # 單機 CLI 入口（支援 --status / --list-cases / --demo-case / --image --question）
│   │   ├── models_params/                 # 系統預設載入之已訓練 Stage 1 RBF-SVM（stage1_rbf_svm_rotate5_exp1.joblib）
│   │   ├── BLIP_attack/                   # BLIP 模型架構、configs、answer_list.json 與 Stage 1/2 評測函式
│   │   ├── cleverhans/                    # BSA PGD 攻擊演算法實作
│   │   ├── data/vlattack_demo/            # 7 個預先計算之系統展示案例（catalog.json）
│   │   ├── static/                        # 樣式表、流程示意圖、7 組案例對照圖片
│   │   ├── templates/                     # index.html, vlattack_live.html, vlattack_cases.html, vlattack.html, results.html
│   │   └── scripts/                       # build_vlattack_demo.py, setup-env.ps1, run-gpu.ps1
│   ├── 2_SDS_軟體設計規格書.md             # 系統架構、模組設計、資料流、API 與規格書
│   └── 3_MU_系統操作手冊.md               # 系統建置、權重設定、Web 與 CLI 操作手冊
└── 實驗設計_VLAttack/
    ├── 1_資料前處理/                       # VQAv2 篩題、候選答案建表、BSA PGD 對抗樣本生成、Exp2/Exp3 建置、11句旋轉 Caption 與 4D 特徵萃取
    ├── 2_訓練腳本/                         # Stage 1 StandardScaler + RBF-SVM 訓練與 .joblib 匯出腳本（含 5-fold GroupKFold OOF 驗證）
    ├── 3_評估腳本/                         # 風險分級、SSS 高斯搜尋（σ*=0.8）、答案翻轉偵測（σ_det=1.5）、緩解後 ASR、11/22 Cosine vs Angular 正式對照
    ├── 4_MU_實驗操作手冊.md               # 實驗重現完整流程與指令手冊
    ├── 5_資料集/                           # 良性原圖 (300張)、成對良性/對抗影像 (109對)、VQAv2 標註與對照表、Exp1/2/3 與 11/22 特徵 CSV/JSON
    │   └── DATASET_MANIFEST.md            # 資料集明細、來源路徑與未重複複製之大型檔案說明
    └── 6_模型參數/                         # 已訓練 .joblib 模型、超參數 JSON、SSS/Flip 最佳參數與外部 BLIP 權重清單
        └── MODEL_MANIFEST.md              # 模型與參數檔案說明
```

---

## 原始專案來源對照（均以唯讀複製，未修改原檔）

| 本資料夾項目 | 原始來源路徑（LAPTOP-3AKL7OUA） | 備註 |
|---|---|---|
| `系統/1_系統完整程式/` | `C:\Users\tracy\github\Web`（及 `VLAttack_Web` 之 Poetry/PowerShell 腳本） | 使用最新（2026-11-16 23:09）`Web` 版本；僅在複本中移除 AttackVLM 頁面／路由並加入 `.joblib` 載入與 `radar_guard_cli.py` |
| `實驗設計_VLAttack/1_資料前處理`～`3_評估腳本` | `C:\Users\tracy\github\VLAttack_final\BLIP_attack\`、`VLAttack_final\cleverhans\`、`VLAttack\BLIP_attack\` | 另含整理之 `train_stage1_svm.py` 與 `compute_rpf1.py` |
| `實驗設計_VLAttack/5_資料集/` | `C:\Users\tracy\github\VLAttack\BLIP_attack\data\VQAv2\val2014`、`runs\IOPGD-s100-a0002-smoke20`、`VLAttack_final\BLIP_attack\output\VQA\` | 彙整良性、對抗成對影像與三組完整實驗特徵／統計表 |
| `實驗設計_VLAttack/6_模型參數/` | 由 `2_訓練腳本/train_stage1_svm.py` 實測訓練匯出 + `exp2_success1000/sss_best_theta.json` + `steps100_alpha0.002_run1000/theta_det.json` | 產生三個 `.joblib` 部署／研究模型檔及 `model_metadata.json` |
| `系統/2_SDS_軟體設計規格書.md`、`3_MU_系統操作手冊.md`、`實驗設計_VLAttack/4_MU_實驗操作手冊.md` | 依實際程式碼與實驗紀錄撰寫 | 原專案無既有 SDS/MU 檔案，本次依實測程式補齊 |

---

## 核心實驗驗證摘要（均出自本資料夾實測紀錄）

1. **VLAttack 攻擊成效（BLIP-VQA, Exp1 1,000 題, ε=0.125, α=0.002, steps=100）**：
   - 初始攻擊成功率（ASR）：**46.20%**（462 / 1,000），平均影像相似度 SSIM = **0.8003**。
2. **Stage 1 TC-VAD（±5° 旋轉 10 次 + 原圖共 11 句 BLIP Caption → MiniLM 4D 角距離 → RBF-SVM）**：
   - **11/22 正式 5-fold GroupKFold（依 `photo_id` 分組）重跑對照**：
     - Exp1（混合 2,000 筆）：Cosine 4D AUROC **0.6640** vs Angular 4D AUROC **0.6653**（Recall 0.4920, Precision 0.6694, F1 0.5671；95% Bootstrap CI `[-0.0204, 0.0214]`）。
     - Exp2（成功攻擊 2,000 筆）：Cosine 4D AUROC **0.7217** vs Angular 4D AUROC **0.7233**（Recall 0.6790, Precision 0.6893, F1 0.6841；95% Bootstrap CI `[-0.0090, 0.0130]`）。
   - 風險分流門檻：低風險放行 `r ≤ 0.30`、中風險送 Stage 2 `0.30 < r < 0.70`、高風險直接攔截 `r ≥ 0.70`。
3. **Stage 2 修復與翻轉偵測（5×5 2D Gaussian Blur）**：
   - **SSS 甜蜜點修復參數**：kernel = `5×5`，**`σ* = 0.8`**（Exp2 中風險子集 `SSS = 0.5766`、`ΔASR = 0.4571`、`IQP = 0.4596`、`SP = 0.9126`）。
   - **答案翻轉偵測探針**：kernel = `5×5`，**`σ_det = 1.5`**（Exp1 單一 σ=1.5 翻轉偵測 `TPR = 40.66%`、`FPR = 5.32%`、`Youden J = 0.3535`；針對被成功欺騙樣本 `TPR = 83.03%`）。
   - **端到端 ASR 下降（BLIP 基準）**：Exp1 ASR 由 **46.2% → 24.1%**（-22.1 個百分點）；Exp2 由 **100% → 40.9%**（-59.1 個百分點）；Exp3 由 **100% → 36.0%**（-64.0 個百分點）。

---

## 磁碟空間控管與外部大型檔案說明（請指導教授／開發者留意）

因本機 `C:` 槽剩餘空間約 **17.48 GB**，依空間保護原則，以下 **超過 5 GB 之原始資料或大型預訓練權重未重複複製** 至本資料夾內，而是保留既有路徑參照並於 manifest 中完整紀錄：

1. **BLIP 預訓練權重（共 4.46 GB）**：
   - `model_base_14M.pth`（2.12 GB）、`model_vqa.pth`（1.45 GB）、`model_base_caption_capfilt_large.pth`（896 MB）
   - 實際位於本機 `C:\Users\tracy\github\VLAttack\BLIP_attack\checkpoints\`；`vlattack_engine.py` 已設定自動偵測該路徑，亦可透過環境變數 `VLATTACK_CHECKPOINTS_DIR` 或 `VLATTACK_ROOT` 指定自訂目錄（詳見 `實驗設計_VLAttack/6_模型參數/checkpoints_manifest.json`）。
2. **COCO val2014 完整壓縮檔（6.64 GB）**：
   - `C:\Users\tracy\github\VLAttack\BLIP_attack\data\VQAv2\val2014.zip` 未重複複製；本資料夾 `5_資料集/benign_val2014_300/` 已完整收錄實驗實際使用之 **300 張**良性原圖（46.6 MB），並收錄 **109 對**（smoke20 + deep_dive_blip + web_demo_7cases）良性與對抗成對 PNG，以及 **1,000 / 2,000 列完整實驗表格特徵 CSV/JSON**。
3. **1,000 對全量 PNG 輸出目錄**：
   - `steps100_alpha0.002_run1000/visual_results` 與 `exp2_success1000/visual_results` 的全量 PNG 影像檔未存於本機 `LAPTOP-3AKL7OUA`（推測存於離線研究機台 `TL`），本機已完整保留其對應之特徵 CSV、VQA 攻擊紀錄 JSON 與評估結果。
4. **HuggingFace 快取模型**：
   - `sentence-transformers/all-MiniLM-L6-v2` 與 `bert-base-uncased` 位於 `C:\Users\tracy\.cache\huggingface\hub\`，可供離線／快取載入。
