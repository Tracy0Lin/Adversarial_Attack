# BLIP 正式實驗交接包

本包以使用者上傳的遠端正式實驗程式、設定、四維特徵CSV、OOF預測、結果檔與實際環境為依據。網頁負責接收輸入、呼叫流程、顯示結果，不反過來決定或修改研究方法。

## 先讀的兩份交接說明

- docs/網頁需要修改的地方.md：逐一列出app.py與attackvlm.html各函式需要如何接上正式實驗。
- docs/交接範圍與尚缺項目.md：區分目前包含的核心程式、必須匯出的模型，以及尚未完整收集的離線評估/資料準備工作。

本包不宣稱包含所有實驗完整重跑所需的程式與資產。

## 唯一方法基準

| 階段 | 正式設定 |
|---|---|
| 受測模型 | BLIP blip_caption / base_coco |
| MF-ii | steps=100、epsilon=8、alpha=1、batch_size=1 |
| MF-tt | steps=8、query=100、subquery=50、sigma=8、epsilon=8、alpha=1、batch_size=1 |
| 第一階段 | 固定旋轉10組＋固定縮放10組，加base共21句描述 |
| 文字特徵 | CLIP ViT-B/32；d_center、d_iqr、d_tail=q90-q50、d_q95 |
| 分類器 | StandardScaler＋RBF SVM，C=1、gamma=scale、class_weight=balanced |
| 風險分數 | sigmoid(decision margin)，不是攻擊機率 |
| 分流 | Low <0.30；Medium 0.30至未滿0.80；High >=0.80 |
| 第二階段 | 僅Medium的輸入圖片進bilateral_strong，d=7、sigmaColor=50、sigmaSpace=7 |
| 防禦描述 | 固定Beam Search，num_beams=3、max_length=30、min_length=10 |
| 攻擊描述 | nucleus sampling；依原攻擊腳本設定保留 |

本包不含舊網頁實作與pilot5備份，避免合併時選到另一套方法或舊版。所有保留的原始檔都有SHA-256可核對。

## 程式位置與角色

- original/project/work_code/：目前確認的正式MF-ii、MF-tt與描述產生程式。
- original/project/formal_tests/paper_coco1000_v1/：自然情境正式轉換、特徵、五折、修復與結果來源。
- original/project/formal_tests/attackvlm_three_scenarios_v1/blip/：壓力測試正式程式與結果來源。
- reference_data/project/：兩種情境既有的特徵CSV、OOF預測、OOF指標；彼此分開。
- environment/：老師主機實際版本與LAVIS來源。
- integration/：此次新增的單張處理介面。6個核心函式從正式原檔原樣抽出並核對AST；這些新檔名不代表原本實驗就使用此介面。
- tools/：此次新增的分類器匯出與原腳本呼叫工具；不變更原始研究設定。

## 先重現正式五折結果，預設不新增部署模型

正式實驗是以image_id分組的五折OOF。匯出工具預設只重建並保存每個情境的5個fold模型，對照原OOF資料，全部通過才保存。沒有以全資料重訓的模型取代報告模型。

將本包解壓到新的資料夾，例如 /workspace/projects/couser/BLIP_Formal_Experiment_20261019，在遠端執行：

```bash
bash /workspace/projects/couser/BLIP_Formal_Experiment_20261019/tools/prepare_cpu_models.sh
```

這會使用原attackvlm環境的Python，僅在CPU讀取既有四維特徵，重建五折、比對、匯出模型。最後產生BLIP_Ready_Handoff_日期時間.zip。自然與壓力情境各有fold_1.joblib至fold_5.joblib、fold_test_ids.json、audit.json。它不重跑圖片攻擊、不載入BLIP/CLIP、不使用GPU、不修改舊實驗與網站。

工具拒絕覆蓋已存在的artifacts；如需重新匯出，另指定新的--output。新上傳的任意圖片需要另行設計並驗證部署分類器；--include-deployment-model是明確選用的附加工具，不是預設流程，產生的模型也不能冒充原OOF評估。

## 單張正式案例

GPU資源可用後，以對應情境的實際案例圖片和ID驗證。例如：

```bash
/workspace/projects/couser/environments/attackvlm/bin/python tools/defend_cli.py --image IMAGE_PATH --scenario natural --image-id 0001
```

IMAGE_PATH與0001必須真的指向該情境的同一案例，工具依原分組選擇未看過該ID的fold模型。自然與壓力情境的ID不能混用。重播案例仍須比對21句描述、四維特徵、風險、處理路徑及濾波後描述。

## 原正式攻擊的單張呼叫

tools/run_attack_cli.py會依序呼叫未修改的mf_ii_transfer.py、generate_captions.py、mf_tt_query.py，明確傳入正式參數，並保存每一步命令與日誌。它需要來源圖片、對應目標圖片和目標文字，會真實執行攻擊，GIT工作占用GPU期間不要執行。

## 已驗證與尚待驗證

自然與壓力情境共4,000筆OOF的分組、分類與風險分流已在CPU重建，全部一致；最大margin誤差約2.14e-14。檢查環境是Python3.12/sklearn1.8.0，因此交付模型仍須用老師主機原Python3.9/sklearn1.6.1環境匯出並再次通過核對。

沒有修改原始圖片、攻擊結果、研究統計或正在執行的GIT。單張BLIP/CLIP/GPU推論與網站端到端測試尚未完成。本包不是已驗證完成的網站，仍需在資源允許時由開發者驗證。包內沒有模型權重與案例圖片；使用原主機環境時應保留既有快取，跨主機時需再確認權重。

詳細設定與驗證限制見docs/正式設定與驗證.md；單張介面見docs/正式流程與單張接口.md。
