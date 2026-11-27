# RADAR-Guard 多模態對抗攻防與雙階段防禦系統 — 專題成果資料夾（VLAttack 模組）

本資料夾為依 **複製（Copy-Only）** 原則整理之專題成果交付包，涵蓋 Tracy Lin 負責之 **VLAttack 視覺模態對抗攻擊與 RADAR-Guard（TC-VAD 幾何一致性偵測 + SSS 高斯去雜訊修復）雙階段防禦** 完整系統程式、軟體設計規格書（SDS）、操作手冊（MU）、實驗前處理／訓練／評估腳本、良性與對抗資料集，以及實際訓練匯出之模型參數。

> **團隊分工說明**：本專案雙線攻擊架構中的 **AttackVLM（白盒語意扭曲）** 部分由團隊其他成員負責，故本成果資料夾僅收錄與說明 **VLAttack** 實驗設計及對應之 RADAR-Guard 系統，不包含 AttackVLM 之程式、資料與文件。

---

## 目錄結構總覽

2026-11-27 全面整理過資料夾架構。舊路徑對應到哪個新路徑，請查 `溯源/新舊路徑對照.tsv`。`文件/成果報告/` 是歷史紀錄，內文仍使用舊路徑名稱。

```text
VLAttack/
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
├── 實驗/                                  # 實驗程式、資料與模型（導覽見 實驗/README.md）
│   ├── 程式/
│   │   ├── 1_攻擊與資料建立/               # BLIP 攻擊整套：VQAv2 篩題、候選答案、BSA PGD、Exp2/Exp3 建置、旋轉 Caption 與 4D 特徵
│   │   ├── 2_訓練/                         # train_stage1_svm.py：Stage 1 SVM 訓練與 .joblib 匯出
│   │   ├── 3_評估/                         # 主流程與舊流程評估腳本（互相 import，放同一層；分類見該資料夾 README）
│   │   ├── 4_報告產生/                     # 結果表、C1～C7 分流報告、c1c7_by_difficulty、實驗二圖表
│   │   ├── 5_題目難度/                     # 題目難度建表與標註合併
│   │   ├── Florence/{攻擊,評估}/           # Florence-2 受害模型的攻擊與評估
│   │   └── _備份_勿執行/                   # 會自動 git push／覆寫輸出的流程備份
│   ├── 資料/                               # 導覽見 實驗/資料/README.md，明細見 DATASET_MANIFEST.md
│   │   ├── 輸入/                           # VQAv2 標註、300 張 COCO 良性原圖
│   │   ├── BLIP/主流程_multi_captioner/    # 報告主流程（Exp1～3 端到端 ASR、C1～C7 來源）
│   │   ├── BLIP/舊流程/                    # steps100_alpha0.002_run1000、exp1_mixed1000、exp2_success1000（SSS、翻轉偵測、Exp2 40.9%）
│   │   ├── BLIP/Stage1正式_11-22/          # Stage 1 正式 cos vs 角距（README AUROC 出處）
│   │   ├── BLIP/攻擊參數搜尋/
│   │   ├── Florence/florence_vlattack/     # 報告用的 Florence-2 數字
│   │   ├── Florence/florence_victim/       # 另一組 Florence 研究（白盒 PGD，Exp1 全部攻擊成功，與 Exp2 同一批）
│   │   ├── 分流分析_C1到C7/  題目難度/  成對樣本圖/  實驗清單/  補充資料表/  release_zips/
│   │   └── DATASET_MANIFEST.md
│   └── 模型/                               # 已訓練 .joblib、超參數 JSON、MODEL_MANIFEST.md
├── 文件/
│   ├── 實驗操作手冊.md                     # 實驗重現流程與指令
│   ├── 系統操作手冊.md                     # 系統建置、權重設定、Web 與 CLI 操作
│   ├── SDS_軟體設計規格書.md               # 系統架構、模組設計、資料流、API
│   └── 成果報告/                           # 2026-09～11 的書面／週報（歷史紀錄）
├── 溯源/                                  # SCRIPT_ORIGIN_MAP.tsv、SHA256SUMS_scripts.txt、新舊路徑對照.tsv
└── 重算工具/                              # 驗證與重算用腳本（含 reproduction_audit/ 重現驗證紀錄）
```

---

## 原始專案來源對照（均以唯讀複製，未修改原檔）

| 本資料夾項目 | 原始來源路徑（LAPTOP-3AKL7OUA） | 備註 |
|---|---|---|
| `系統/1_系統完整程式/` | `C:\Users\tracy\github\Web`（及 `VLAttack_Web` 之 Poetry/PowerShell 腳本） | 使用最新（2026-11-16 23:09）`Web` 版本；僅在複本中移除 AttackVLM 頁面／路由並加入 `.joblib` 載入與 `radar_guard_cli.py` |
| `實驗/程式/1_攻擊與資料建立`～`實驗/程式/3_評估` | `C:\Users\tracy\github\VLAttack_final\BLIP_attack\`、`VLAttack_final\cleverhans\`、`VLAttack\BLIP_attack\` | 另含整理之 `train_stage1_svm.py` 與 `compute_rpf1.py` |
| `實驗/資料/` | `C:\Users\tracy\github\VLAttack\BLIP_attack\data\VQAv2\val2014`、`runs\IOPGD-s100-a0002-smoke20`、`VLAttack_final\BLIP_attack\output\VQA\` | 彙整良性、對抗成對影像與三組完整實驗特徵／統計表 |
| `實驗/模型/` | 由 `實驗/程式/2_訓練/train_stage1_svm.py` 實測訓練匯出 + `exp2_success1000/sss_best_theta.json` + `steps100_alpha0.002_run1000/theta_det.json` | 產生三個 `.joblib` 部署／研究模型檔及 `model_metadata.json` |
| `文件/SDS_軟體設計規格書.md`、`文件/系統操作手冊.md`、`文件/實驗操作手冊.md` | 依實際程式碼與實驗紀錄撰寫 | 原專案無既有 SDS/MU 檔案，本次依實測程式補齊 |

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
   - 實際位於本機 `C:\Users\tracy\github\VLAttack\BLIP_attack\checkpoints\`；`vlattack_engine.py` 已設定自動偵測該路徑，亦可透過環境變數 `VLATTACK_CHECKPOINTS_DIR` 或 `VLATTACK_ROOT` 指定自訂目錄（詳見 `實驗/模型/checkpoints_manifest.json`）。
2. **COCO val2014 完整壓縮檔（6.64 GB）**：
   - `C:\Users\tracy\github\VLAttack\BLIP_attack\data\VQAv2\val2014.zip` 未重複複製；本資料夾 `實驗/資料/輸入/benign_val2014_300/` 已完整收錄實驗實際使用之 **300 張**良性原圖（46.6 MB），並收錄 **109 對**（smoke20 + deep_dive_blip + web_demo_7cases）良性與對抗成對 PNG，以及 **1,000 / 2,000 列完整實驗表格特徵 CSV/JSON**。
3. **1,000 對全量 PNG 輸出目錄**：
   - `exp1_mixed1000/visual_results` 與 `exp2_success1000/visual_results` 已壓縮為兩個 zip，上傳到 GitHub Release `vlattack-data-v1`（見下方「資料與權重下載」）；`steps100_alpha0.002_run1000/visual_results`（約 16 GB）未上傳，本 repo 保留其特徵 CSV、VQA 攻擊紀錄 JSON 與評估結果。
4. **HuggingFace 快取模型**：
   - `sentence-transformers/all-MiniLM-L6-v2` 與 `bert-base-uncased` 位於 `C:\Users\tracy\.cache\huggingface\hub\`，可供離線／快取載入。

---

## 資料與權重下載（Release 圖片與官方 BLIP 權重）

本 repo 只放小檔（程式、CSV、JSON、文件與三個 SVM `.joblib`），**不使用 Git LFS**。大型檔案分成兩類：Release 上的實驗圖片壓縮檔，以及 Salesforce 官方的 BLIP 權重。

### 1. 實驗圖片（GitHub Release `vlattack-data-v1`）

| 檔案 | 內容 | 大小 | SHA-256 |
|---|---|---:|---|
| `exp1_mixed1000_visual_results.zip` | 實驗一 1,000 對（`_1_ori.png` 乾淨圖 + `_2_atk.png` 攻擊圖，共 2,000 張）及 `dataset_manifest.csv` | 733,719,871 bytes（0.68 GiB） | `2740f7b0e036272b3b33df1f5df71c822a40626d2c1c8046cbd096e07862e7ea` |
| `exp2_success1000_visual_results.zip` | 實驗二 1,000 對成功攻擊（2,000 張）及 `dataset_manifest.csv` | 738,985,508 bytes（0.69 GiB） | `f56dbd2fd82341ce9f6a34c37090c87baa7a066e527f1f99231b00b90512e0ac` |

- Release 頁面：https://github.com/Tracy0Lin/Adversarial_Attack/releases/tag/vlattack-data-v1
- 直接下載：
  - https://github.com/Tracy0Lin/Adversarial_Attack/releases/download/vlattack-data-v1/exp1_mixed1000_visual_results.zip
  - https://github.com/Tracy0Lin/Adversarial_Attack/releases/download/vlattack-data-v1/exp2_success1000_visual_results.zip
- **本 repo 為私有**，下載需要有 repo 讀取權限並已登入 GitHub；未登入時直接以瀏覽器或 curl 開連結會得到 404。建議使用 GitHub CLI：

```bash
gh release download vlattack-data-v1 --repo Tracy0Lin/Adversarial_Attack --dir release_zips_download
```

- 校驗方式：把兩個 zip 與 `實驗/資料/release_zips/SHA256SUMS.txt` 放在同一資料夾後執行：

```bash
sha256sum -c SHA256SUMS.txt          # Linux / macOS / Git Bash
```

```powershell
# Windows PowerShell：逐一比對，輸出需與 SHA256SUMS.txt 相同
Get-FileHash .\exp1_mixed1000_visual_results.zip -Algorithm SHA256
Get-FileHash .\exp2_success1000_visual_results.zip -Algorithm SHA256
```

- 解壓後會得到 `exp1_mixed1000/visual_results/*.png` 與 `exp2_success1000/visual_results/*.png`，檔名格式為 `<question_id>_1_ori.png`、`<question_id>_2_atk.png`。
- 小檔 CSV 另放在 repo：`實驗/資料/題目難度/`（`labels.csv`、`samples.csv`、`c1c7_by_difficulty.csv`）、`實驗/資料/實驗清單/`、`實驗/資料/補充資料表/`，以及 `文件/成果報告/2026-11-09_難度與C1到C7.md`。其中 `labels.csv`、`samples.csv` 的 `image_path` 欄為原實驗機（TL）的絕對路徑，僅供追溯，移機後請改用 zip 內的相對路徑。

### 2. BLIP 預訓練權重（官方下載，不在 repo 內）

三個檔案皆來自 Salesforce BLIP 官方 README（https://github.com/salesforce/BLIP），下載連結已以 HEAD 請求確認為 HTTP 200，且 Content-Length 與本專案實驗使用的檔案完全相同。

| 檔案 | 用途 | 大小（bytes） | 官方下載 | SHA-256（實驗所用檔案） |
|---|---|---:|---|---|
| `model_base_14M.pth` | BLIP 預訓練模型（BSA PGD 白盒代理梯度來源） | 2,115,792,294 | https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_14M.pth | `894d4946d521a258dfb3e069c7cf4556e9da1865b7356d71e75226d384f99a46` |
| `model_base_caption_capfilt_large.pth` | BLIP Caption（Stage 1 的 11 句描述） | 896,081,425 | https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_caption_capfilt_large.pth | `96ac8749bd0a568c274ebe302b3a3748ab9be614c737f3d8c529697139174086` |
| `model_vqa.pth` | BLIP-VQA 受害模型 | 1,446,244,375 | https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_vqa.pth | `91b817b95d4f9f38de6d1ab10b993ab7c5816b8e9d4bc40e7cc2c10d1d02f729` |

```bash
mkdir -p checkpoints && cd checkpoints
curl -L -O https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_14M.pth
curl -L -O https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_caption_capfilt_large.pth
curl -L -O https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_vqa.pth
```

**放置位置：** 預設放在 `BLIP_attack/checkpoints/`。放在其他位置時，設定環境變數 `VLATTACK_CHECKPOINTS_DIR` 指向存放三個 `.pth` 的資料夾（或設 `VLATTACK_ROOT`），`vlattack_engine.py` 會依序搜尋。

```powershell
$env:VLATTACK_CHECKPOINTS_DIR = 'D:\models\blip'     # Windows PowerShell
```

```bash
export VLATTACK_CHECKPOINTS_DIR=/data/models/blip    # Linux / macOS
```

另需 HuggingFace 的 `sentence-transformers/all-MiniLM-L6-v2` 與 `bert-base-uncased`，首次執行時會自動下載。

### 3. 未包含的資料

- `steps100_alpha0.002_run1000/visual_results`：4,423 組 × 12 種變化共 53,076 張 PNG、約 16 GB，體積過大，未上傳；其統計與特徵表已在 `實驗/資料/BLIP/舊流程/steps100_alpha0.002_run1000/`。
- COCO val2014 完整影像（40,504 張，約 6.18 GB）：請自行從 COCO 官方網站下載（https://cocodataset.org/#download）；實驗實際用到的 300 張已收錄在 `實驗/資料/輸入/benign_val2014_300/`。
- Florence 的大型輸出（探針快取 `probe_cache_*.csv`、`screening_log.jsonl`、題目池 JSON、題目難度的裁切圖與標註圖）未上傳；Florence 的腳本與 `sigma08_fixed` 結果表已放進本 repo，見下方「補充：Florence、題目難度、分流分析與重算工具」。

---

## 補充：Florence、題目難度、分流分析與重算工具

2026-11-27 補入。以下檔案都是原電腦上的唯讀複本（已用 SHA256 比對），沒有修改原檔；不含圖片、模型權重或 `.pyc`。腳本內寫死的路徑是原作者電腦的路徑，換機器執行前需自行改路徑。

| 內容 | 腳本位置 | 結果／資料位置 |
|---|---|---|
| Florence-2 受害模型：攻擊與資料建立 | `實驗/程式/Florence/攻擊/`（含 README） | — |
| Florence-2 評估與分流分析 | `實驗/程式/Florence/評估/`（含 README） | `實驗/資料/Florence/florence_vlattack/exp{1,2,3}/florence2/vqa_florence2_sigma08_fixed/`（`rows_scored.csv`、`summary.json`、`pipeline_variants.csv`、`sss_grid_mid.csv`、`theta_det_grid_mid.csv`） |
| BLIP 實驗三 `sigma08_fixed` 結果 | `實驗/程式/3_評估/analyze.py`、`run_analyze_sigma08.py` | `實驗/資料/BLIP/主流程_multi_captioner/exp3_blip/sigma08_fixed/` |
| 題目難度（BLIP / Florence）建表與合併 | `實驗/程式/5_題目難度/`（`build_samples.py`、`merge_labels.py`、`build_florence_samples.py`、`merge_florence_labels.py`） | `實驗/資料/題目難度/`：`batches/*.json`、`labels/*.jsonl`、`florence/{samples.csv,labels.csv,batches,labels}` |
| 檢核表與補充資料表（舊版管線，2026-09-26） | `實驗/程式/4_報告產生/build_checklist_analytics.py`、`build_experiment_data_supplement.py` | `實驗/資料/補充資料表/` |
| 分流深入分析（C1–C7）與答案距離 | `實驗/程式/4_報告產生/stage_routing_deep_dive/`（`build_report.py`、`answer_distance/build_answer_distance.py`、`build_clean_answer_distance.py`） | `實驗/資料/分流分析_C1到C7/`（各模型 `REPORT.md`、csv；圖片 PNG 未上傳） |
| 攻擊甜蜜點分析 | `實驗/程式/1_攻擊與資料建立/analyze_attack_sweet_spot.py` | `實驗/資料/BLIP/攻擊參數搜尋/attack_stealth_natural_summary.csv` |
| 數字重算工具（為驗證而寫，見其 README） | `重算工具/recompute_c1c4.py`、`重算工具/asr_redefine/compute.py` | — |
| 流程備份（**請勿執行**，見其 README） | `實驗/程式/_備份_勿執行/`：`run_end_to_end.py`（會自動 git commit/push）、`attack_supervisor.py`（會覆寫攻擊輸出） | — |

---

## 補充（第二批補入）：驗證用計分表、探針快取、腳本來源清單與已知缺漏

2026-11-27 第二批補入，全部是原電腦檔案的唯讀複本（只放小檔，不含圖片、zip、權重）。

### 新增檔案

| 內容 | 位置 |
|---|---|
| Florence_victim 計分表（Exp1–3，`vqa_florence2` 與 `vqa_florence2_sigma08_fixed`）、`probe_gaussian_florence2.csv`、`caption_features.csv`、題目池與篩題紀錄（`eligible_pool.json`、`screening_log.jsonl`、`florence_correct_ans_table.json`、`probe_cache_florence2.csv`） | `實驗/資料/Florence/florence_victim/` |
| Florence（VLAttack 特徵攻擊）預設 θ 計分表（`vqa_florence2`）、`probe_gaussian_florence2.csv`、探針快取、caption／VQA cache | `實驗/資料/Florence/florence_vlattack/` |
| BLIP 預設 θ 的 Exp3 計分表、`probe_gaussian.csv`（Exp1–3）、探針快取、各項 log、caption cache | `實驗/資料/BLIP/主流程_multi_captioner/` |
| 7 組圖片資料集的 `dataset_manifest.csv`、`dataset_info.txt`、`attack_dual_defense_log.json` | `實驗/資料/實驗清單/` |
| 書面／週報等 25 份成果報告 | `文件/成果報告/` |
| 腳本來源 SHA-256 清單（`溯源/SHA256SUMS_scripts.txt`，內容為 repo 內檔案的 LF 版雜湊）與逐檔來源對照（`溯源/SCRIPT_ORIGIN_MAP.tsv`） | 本資料夾根目錄 |
| 還原版腳本 `regen_c1c7_by_difficulty.py` | `實驗/程式/3_評估/` |

`溯源/SCRIPT_ORIGIN_MAP.tsv` 的狀態：28 個與原檔逐位元相同、518 個只差換行（CRLF／LF）、14 個內容不同、8 個在原電腦找不到原檔，另有 1 個還原版腳本（`RECONSTRUCTED_NO_ORIGINAL`）。內容不同的 14 個主要是系統網站（移除 AttackVLM 後修改過）與 `requirements.txt`。

### 還原版腳本（不是原檔）

`regen_c1c7_by_difficulty.py` 是 **2026-11-27 還原的版本，不是原始產生程式**。產生 `question_difficulty/c1c7_by_difficulty.csv` 的原始腳本已遺失（原電腦所有 `.py`／`.ipynb`／`.md` 都找不到）。這支腳本依 `文件/成果報告/2026-11-09_難度與C1到C7.md` 的實驗主表規則重建：高風險擋下、低風險放行、中風險只糊 σ = 0.8；BLIP 的答案跟 `original_prediction` 比，Florence 跟 `ground_truth` 比。它用 `rows_scored.csv` 與難度標註重算，**72 列結果與原檔完全一致（0 列不同）**。檔案開頭有「RECONSTRUCTED」標記。

### 尚未上傳的 Release 圖片（計畫放進 `vlattack-data-v1`）

下列 9 個 zip 已在原電腦打包並驗證，**尚未上傳**（等 `gh auth login`），雜湊先列在 `實驗/資料/release_zips/SHA256SUMS.txt` 的註解區。

| 檔案 | 內容 | 大小（bytes） |
|---|---|---:|
| `exp3_success1000_visual_results.zip` | BLIP 實驗三 1,000 對 | 743,521,497 |
| `florence_vlattack_exp{1,2,3}_*_visual_results.zip` | Florence（VLAttack 特徵攻擊）三組，各 1,000 對 | 1,960,564,830／1,952,161,486／2,016,374,669 |
| `florence_victim_exp{1,2,3}_*_visual_results.zip` | Florence_victim 三組（Exp1 的資料夾含 2,803 張） | 1,833,287,129／1,458,855,573／1,470,432,821 |
| `question_difficulty_images.zip` | 題目難度的裁切圖與標註圖（64 張） | 10,728,102 |
| `stage_routing_deep_dive_pngs.zip` | C1–C7 分類 PNG（440 張） | 267,284,813 |

### 無法復原或需要注意的項目

- **五折模型檔與每折測試 ID 當時沒有存檔，找不到。** 可以重新產生：`run_rotation_cos_vs_ang.py`／`analyze.py` 用 `GroupKFold(n_splits=5)`（依 `photo_id` 分組、不洗牌）加 RBF SVC（C=1、`class_weight=balanced`），結果是確定性的。本次**沒有重新訓練**。
- **8 個腳本在原電腦找不到原檔**：`requirements_experiment.txt`、`實驗/程式/2_訓練/train_stage1_svm.py`、`實驗/程式/3_評估/compute_rpf1.py`、`實驗/程式/3_評估/run_rotation_cos_vs_ang.py`、`實驗/程式/4_報告產生/實驗二_cos_vs_角距_圖表/build_exp2_excel_and_figures.py`、`系統/1_系統完整程式/radar_guard_cli.py`、`vlattack_gpu_worker.py`、`templates/vlattack.html`。
- **Florence_victim 的 Exp1 與 Exp2 計分表逐位元相同**（`vqa_florence2` 與 `sigma08_fixed` 都是），Exp3 不同。
- **Florence_victim `exp1_mixed1000/visual_results` 有 2,803 張 PNG、涵蓋 1,803 個 question id**（不是 1,000 對），且沒有 `dataset_manifest.csv`。
- **Exp1 翻轉偵測**不缺資料：`steps100_alpha0.002_run1000/sss_per_theta_raw.csv` 加 `answer_flip_detector.py` 重算後，σ = 1.5 的 TPR 40.66%、FPR 5.32%、`theta_det_sweep.csv`、`flip_count_per_sample.csv`、`theta_det.json` 都與原檔一致。
- **VQAv2 官方標註**（`重算工具/asr_redefine` 需要）：`https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip`（10,518,930 bytes，SHA-256 `0caae7c8d1dafd852727f5ac046bc1efca9b72026bd6ffa34fc489f3a7b3291e`），解壓後得 `v2_mscoco_val2014_annotations.json`（171,762,489 bytes）。檔案太大，不放 repo。
- `steps100_alpha0.002_run1000/visual_results`（53,076 張，約 17 GB）確認存在，但不上傳。
