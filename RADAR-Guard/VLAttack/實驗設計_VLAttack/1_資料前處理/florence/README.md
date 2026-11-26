# Florence-2 受害模型實驗腳本（攻擊／資料建立）

來自原作者電腦 `BLIP_attack/output/florence_victim/` 與 `florence_vlattack/`。

- `florence_vlattack/`：Florence-2 為受害模型的 VLAttack 實驗（exp1–3 的樣本建立、攻擊、檢查、`run_study.py` 整合）。報告用的 Florence 數字來自這一組。
- `florence_victim/`：Florence-2 的 PGD 攻擊、推論引擎、eligible 題目池與 exp2/exp3 的資料建立／補替換。
- 評估端腳本在 `../../3_評估腳本/florence/`；結果檔在 `../../5_資料集/tabular_features_and_logs/florence_vlattack/`。

注意：這些是**攻擊／模型腳本，需要 GPU 與 Florence-2 權重、COCO 圖片**，且含寫死的原電腦路徑（`IMG_ROOT`、`WEIGHTS`）。圖片、探針快取（`probe_cache_*.csv`）、`screening_log.jsonl` 沒有放進 repo（太大）。