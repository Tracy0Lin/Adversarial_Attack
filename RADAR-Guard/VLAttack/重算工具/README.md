# 重算工具（為驗證數字而寫的唯讀腳本）

這兩支腳本是 2026-11-23～24 為了**核對書面報告 / 簡報數字**而寫，不是原本實驗流程的一部分。它們只讀既有輸出（`rows_scored.csv`，`sigma08_fixed`），不改動任何實驗檔案。

| 檔案 | 用途 |
|---|---|
| `recompute_c1c4.py` | 由 BLIP `sigma08_fixed/rows_scored.csv`（multi_captioner exp1/2/3）重算表 4.28 的 C1–C4 數量。 |
| `asr_redefine/compute.py` | 以 VQAv2 人工答案（soft accuracy）重新定義 ASR，輸出新舊 ASR 對照。 |
| `rerun_florence_routing.py` | 2026-11-27 新增。呼叫未修改的 `實驗/程式/4_報告產生/stage_routing_deep_dive/build_report.py`，只重新產生 `Florence_victim/`（讀 repo 內計分表、不複製圖片），輸出到指定的暫存資料夾。 |
| `reproduction_audit/verify_reported_numbers.py` | 2026-11-27 新增。從 repo 內 CSV／JSON／`.joblib` 重算根目錄 README 的所有主要數字，結果寫入 `reproduction_audit.json`，說明見該資料夾 README。路徑皆相對於 repo，不需改路徑。 |

注意：

- 兩支腳本都有寫死的路徑（`/workspace/...`，是撰寫時的 Linux 工作環境），在別的電腦要先改 `OUT`、`SCORED`、`BASE`、`ANN` 等路徑。
- 需要 VQAv2 官方標註 `v2_mscoco_{train,val}2014_annotations.json`（不在本 repo）。
- `recompute_c1c4.py` 也會讀 `asr_redefine/` 產生的 `per_question.csv`，要先跑 `compute.py`。
- C1–C4 在這裡的定義（書面報告 4.2.10）與 `實驗/資料/分流分析_C1到C7` 的 C1–C7 不同，請勿混用。
- 實驗一的 C1–C4 只計 `fooled_before` 的 461 筆（不是 1000）；實驗二、三計全部 1000 張攻擊圖。