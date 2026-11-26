# Florence-2 評估腳本

- `florence_vlattack/score_florence.py`：Florence 實驗的打分。
- `florence_victim/`：`run_defense_pipeline.py`、`run_e2e.py`（防禦管線）、`vqa_match.py`、`analyze_stage_routing.py`（Florence 的分流分析）。`common.py` 與攻擊端 `1_資料前處理/florence/florence_victim/common.py` 內容相同，各放一份方便各自執行。
- 防禦與統計用的 `analyze.py` / `run_analyze_sigma08.py` 在上一層 `3_評估腳本/`（BLIP 與 Florence 共用）。
- Florence exp1–3 的 `sigma08_fixed` 結果在 `5_資料集/tabular_features_and_logs/florence_vlattack/exp{1,2,3}/florence2/vqa_florence2_sigma08_fixed/`。