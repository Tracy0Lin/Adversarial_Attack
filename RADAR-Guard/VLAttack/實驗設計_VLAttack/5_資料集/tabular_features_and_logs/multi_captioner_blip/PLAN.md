# 多 caption 模型 × 三實驗：兩階段防禦評估計畫

建立日：2026-10-15  
目錄：`BLIP_attack/output/VQA/multi_captioner/`  
程式：`BLIP_attack/output/multi_captioner/`

---

## 需求對照（與你列的由上往下一致）

| # | 你的要求 | 計畫與程式 | 狀態 |
|---|----------|------------|------|
| 1 | **實驗一**：1000 張圖走 VLAttack，自然得到 1000 乾淨 + 1000 攻擊 | 資料：`steps100_alpha0.002_run1000` 前 1000 題（2000 列） | 資料已有；流程待 **4 種 caption × 統計** |
| 2 | **實驗二**：1000 乾淨 + 1000 攻擊，攻擊皆成功 | 資料：`exp2_success1000` | 資料已有；同上 |
| 3 | **實驗三**：僅 1000 張攻擊成功圖（獨立、不重疊照片） | `build_exp3.py` + 補攻擊 `attack_supervisor.py` | **進行中**（目前約 53/1000 張合格，需補攻擊） |
| 4 | 實驗一、二、三皆跑 **整體流程**：第一階段 ±5° + 第二階段高斯 | `run_captions.py` + `run_probe.py` + `analyze.py` | 一、二探針已完成；caption 與統計進行中；三待資料 |
| 5 | 舊指標 + 更能展現成果的指標 | §6 + `analyze.py` / `make_tables.py` | 已寫入程式，待全部跑完出表 |
| 6 | **實驗二消融**：第一階段、第二階段分開看 | `analyze.py` 變體 A0–A3（見 §5） | 程式已有；待各 caption 跑完 |
| 7 | **LLM-API**（`C:\Users\tracy_aboikg6\github\LLM-API`）當 caption | `gemma3:4b` 經 `http://140.128.36.108:8025/chat` 帶圖 | 已接好；實驗一、二 caption 進行中 |
| 8 | **ViT-GPT2**、**GIT-large-COCO** | `captioners.py` 本機 GPU | 已接好；排在本機 BLIP 之後跑 |
| 9 | 上述模型 **各跑實驗一、二、三** 並統計 | 矩陣：3 實驗 × 4 caption（見下） | 一、二進行中；三待補攻擊 |

**Caption 模型（你要的三種 + 一項對照）**

| 你要的 | 代號 | 說明 |
|--------|------|------|
| LLM-API 視覺大模型 | `gemma3` | 走 LLM-API 代理，不是本機再載一套 |
| ViT-GPT2 | `vitgpt2` | `nlpconnect/vit-gpt2-image-captioning` |
| GIT-large-COCO | `git` | `microsoft/git-large-coco` |
| （未要求，為對照舊論文） | `blip` | 與先前 TC-VAD 相同 BLIP caption，方便比新模型 |

若你只想報 **LLM + ViT-GPT2 + GIT** 三條主線，最終表可以只列這三個；BLIP 仍會跑完當基準，不影響你的三條。

**整體流程（三個實驗都同一套）**

1. 第一階段：原圖 + 10 次 ±5° → 11 句 caption → 四維特徵 → SVM 風險分 → Low / Mid / High。  
2. 第二階段（Mid）：5×5 高斯掃 σ → θ\* 緩解（SSS）+ θ_det 答案翻轉偵測 → 端到端路由。  
3. 黑盒 VQA 仍是 BLIP VQA（與先前攻擊設定一致）；換的是 **caption 探針**（及 SSS 裡的 caption 相似度）。

---

## 1. 要回答的問題

同一條兩階段防禦流程（第一階段 ±5° 旋轉 caption 探測，第二階段高斯濾波），把 caption 模型從 BLIP 換成其他三種之後：

1. 在三種資料情境（混合流量、平衡的成功攻擊、純攻擊）下，攔截與緩解的效果各是多少。
2. 第一階段、第二階段各自貢獻多少（實驗二消融）。
3. 哪一種 caption 模型最適合當第一階段的探針。

## 2. 三個實驗（彼此獨立）

| | 實驗一 | 實驗二 | 實驗三 |
|--|--------|--------|--------|
| 內容 | 1000 張圖照常跑 VLAttack，得到 1000 乾淨 + 1000 攻擊（成功、失敗都有） | 1000 乾淨 + 1000 攻擊，攻擊全部成功 | 1000 張攻擊成功圖，沒有乾淨圖 |
| 來源 | `steps100_alpha0.002_run1000` 攻擊 log 前 1000 筆（= 現有 `tcvad_stage1_4_academic_dataset.csv` 的題號） | `exp2_success1000`（已建好） | **新攻擊**：接續同一個攻擊目錄往下跑 |
| 規模 | 2000 列、207 張照片 | 2000 列、388 張照片 | 1000 列 |
| 回答 | 實際部署的混合流量下表現如何 | 攻擊已經奏效時的上限 | 只有攻擊時攔得到多少、ASR 剩多少 |

### 實驗三的獨立性

- 只收 `attack_success == true`、圖檔齊全的題目。
- 題號不在實驗一、實驗二裡；**照片**（`question_id // 1000`）也不在實驗一、實驗二裡，避免同一張照片的不同問題跨實驗洩漏。
- 舊 log 裡有 44 張符合條件；其餘約 956 張要新攻擊。以 ASR 約 45% 估算，要再攻擊約 2200 筆，每筆約 26 秒，**約 16 小時 GPU**。
- 攻擊參數與實驗一、二完全相同：VLAttack、PGD、steps=100、alpha=0.002、L∞=0.125。
- 攻擊程式曾因記憶體不足中斷，所以用監督程式 `attack_supervisor.py`：當掉自動重啟（攻擊本身支援 resume），湊滿 1000 張就停。
- 開跑前先把現有 2298 筆的 log 快照到 `attack_log_snapshot_2298.json`。

### 實驗三怎麼打分

實驗三只有攻擊圖，SVM 無法在上面訓練（只有一類）。做法：

- 第一階段的 SVM 用**實驗二的全部 2000 列**訓練好（同一個 caption 模型），再對實驗三打分。實驗二與實驗三沒有共同照片。
- 第二階段的 θ\*（緩解）與 θ_det（偵測）也沿用該 caption 模型在實驗二選出的值，不在實驗三上調參。
- 因為沒有乾淨圖，實驗三沒有 FPR、AUROC，只報攔截率、各風險區間人數、緩解後 ASR。

## 3. Caption 模型

| 代號 | 模型 | 執行方式 |
|------|------|----------|
| `blip` | BLIP caption（`model_base_caption_capfilt_large.pth`，beam 3） | 本機 GPU；作為對照基準 |
| `vitgpt2` | `nlpconnect/vit-gpt2-image-captioning`（beam 3） | 本機 GPU |
| `git` | `microsoft/git-large-coco`（beam 3） | 本機 GPU |
| `gemma3` | `gemma3:4b`，經 `C:\Users\tracy_aboikg6\github\LLM-API` 代理（`http://140.128.36.108:8025/chat` 帶圖片），temperature 0，提示「用一句短句描述這張圖」 | 遠端 Ollama |

句向量一律用 `sentence-transformers/all-MiniLM-L6-v2`，與舊流程相同。

## 4. 整體流程

### 第一階段：±5° 旋轉 caption 探測

1. 圖縮到 384×384。
2. 原圖寫 1 句 caption，再做 10 次旋轉（每次隨機 −5° 或 +5°，種子與舊程式相同），各寫 1 句，共 11 句。
3. 11 句轉成 MiniLM 向量，算每句與平均方向的角距，得到 `d_center`、`d_iqr`、`d_tail`、`d_q95`。
4. RBF SVM（C=1、`class_weight=balanced`）轉成風險分。實驗一、二用 **5 折 GroupKFold，按照片分組**做折外分數；實驗三用實驗二訓練好的模型。
5. 風險 ≤ 0.3 為 Low（放行），≥ 0.7 為 High（攔下），中間為 Mid（進第二階段）。

旋轉只有兩種角度，所以每張圖只有 3 種不同輸入（原圖、−5°、+5°）。caption 以「像素雜湊 + 角度」快取，11 句的組成與舊程式完全一樣，只是不重算相同輸入。

### 第二階段：高斯濾波（只處理 Mid）

先對**所有列**跑一次與 caption 模型無關的高斯探針：5×5 高斯，σ ∈ {0.3, 0.5, 0.8, 1.0, 1.5}，記錄 BLIP VQA 模糊前後的答案、IQP（SSIM、1−LPIPS、銳度比、BRISQUE）與 CLIP 相似度。之後每個 caption 模型只要再算模糊圖的 caption（給 SP 用），就能各自組出 Mid 的結果。

- **緩解（θ\*）**：SSS = (ΔASR · IQP · SP)^(1/3)，SP = 幾何平均(CLIP 相似度, caption 相似度)。在 Mid 上選 SSS 最大的 σ；Mid 圖以 θ\* 模糊後再交給 VQA。
- **偵測（θ_det）**：模糊前後 VQA 答案不同就判為攻擊。在 Mid 上選 Youden J 最大的 σ。

端到端規則：Low 放行；High 攔下；Mid 依答案翻轉 @θ_det 判定是否攔下，未攔下者以 θ\* 模糊後回答。

## 5. 實驗二消融（第一階段 / 第二階段分開）

**只看第一階段**

| 代號 | 設定 |
|------|------|
| A1 | 只攔 **High**（≥0.7），Low / Mid 放行 |
| A1′ | 攔 **Mid + High**（風險 > 0.3），等同「不用第二階段、把中風險當攻擊」 |

**只看第二階段**（2000 列全走 Mid 路徑，不經第一階段路由）

| 代號 | 設定 |
|------|------|
| A2 | 全列：答案翻轉偵測 @θ_det + 未攔者以 θ\* 緩解（θ 在**全 2000 列**上選） |
| A2′ | 全列：只做 θ\* 緩解，**不**依翻轉攔截 |

**對照與完整**

| 代號 | 設定 |
|------|------|
| A0 | 不防禦 |
| A3′ | 完整路由，但 Mid **只緩解、不**用翻轉攔截 |
| A3 | **完整流程**：Low 放 / High 攔 / Mid 翻轉偵測 + θ\* 緩解 |

每個 caption 模型各做一次。翻轉用的 VQA 答案與 caption 模型無關；θ\* 的 SP 會隨 caption 模型改變。

## 6. 指標

沿用舊指標：AUROC、TPR、FPR、Youden J、中風險率、Low/Mid/High 分布、θ\*、θ_det、SSS、路由後 ASR、乾淨答案改變率。

新增：

| 指標 | 用途 |
|------|------|
| Precision、F1、Balanced Accuracy | 攔截決策的整體品質 |
| AUPRC | 正負類比例改變時比 AUROC 穩 |
| TPR @ FPR ≤ 5% / 1% | 低誤攔時的實際攔截力 |
| 有效 ASR（被攔下視為防住） | 端到端還剩多少攻擊得逞 |
| 乾淨服務率 | 乾淨題沒被攔、答案也沒被模糊改掉的比例（可用性） |
| 每張圖平均 VQA／caption 呼叫次數 | 成本 |
| 照片層 bootstrap 95% 信賴區間（1000 次） | 誤差範圍 |
| caption 改寫率（11 句中與原句不同的比例） | 解釋各模型的穩定度差異 |

## 7. 執行順序

| 步驟 | 內容 | 預估 |
|------|------|------|
| 0 | 快照攻擊 log；寫程式；小量試跑 | 1 h |
| 1 | 背景啟動實驗三補攻擊（監督程式） | **約 16 h**（與下列步驟並行） |
| 2 | 實驗一、二高斯探針（所有列 × 5 σ） | 約 1 h |
| 3 | 實驗一、二 × 4 caption 模型：第一階段 + 模糊圖 caption | 約 3–5 h |
| 4 | 實驗一、二統計與實驗二消融 | 幾分鐘 |
| 5 | 補攻擊完成後建實驗三資料集、跑探針與 4 個 caption 模型 | 約 2 h |
| 6 | 實驗三統計、總報告 `RESULTS.md` | 30 min |

GPU 只有 8 GB：補攻擊進行中時，本機 caption 模型若擠不下就先跑 `gemma3`（遠端），其餘等攻擊空檔或結束後再跑。

## 8. 產出

```
multi_captioner/
├── PLAN.md                         本檔
├── attack_log_snapshot_2298.json   補攻擊前的 log 快照
├── attack_supervisor.log           補攻擊監督紀錄
├── exp3_success1000/               實驗三資料集（log、manifest、visual_results）
├── caption_cache/<model>/          caption 快取
├── <exp>/probe_gaussian.csv        高斯探針（與 caption 模型無關）
├── <exp>/<model>/stage1.csv        第一階段特徵與風險分
├── <exp>/<model>/stage2_mid.csv    Mid 的逐 σ 結果
├── <exp>/<model>/summary.json      該組合的全部指標
├── exp2/<model>/ablation.csv       實驗二消融
└── RESULTS.md                      總報告
```

## 9. 限制（事先說明）

- 黑盒 VQA 仍是 BLIP VQA；換的只是第一階段的 caption 探針與 SP 的 caption 部分。
- 實驗二、三的對抗側都只有成功攻擊，數字是條件上限，不能當部署數字；部署敘事以實驗一為主。
- 實驗三的乾淨對照不存在，FPR 要引用實驗二同模型的數字。
- `gemma3:4b` 是遠端共用服務，結果依賴當時的模型版本；temperature 0 仍可能有極少量非決定性。
