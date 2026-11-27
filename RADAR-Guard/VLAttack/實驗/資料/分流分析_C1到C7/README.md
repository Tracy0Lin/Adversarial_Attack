# 防禦路由七類深入分析（BLIP + Florence）

由 `build_report.py` 產生。每類別每實驗保留 **5–10** 個樣本（優先不同 `photo_id`），
圖片在子目錄 `樣本_qid/` 下（`_*_ori.png` / `_*_atk.png`）。

## 目錄

- [COMPARISON.md](../../../文件/成果報告/2026-10-24_stage_routing_COMPARISON.md) — BLIP vs Florence 對照摘要
- [BLIP_victim/REPORT.md](BLIP_victim/REPORT.md) — 含樣本表與圖片路徑
- [Florence_victim/REPORT.md](Florence_victim/REPORT.md) — 含樣本表與圖片路徑
- `build_report.py` — 可重跑以更新 CSV / 樣本圖

## 2026-11-27 重新產生 `Florence_victim/`

`BLIP_victim/` 仍是原電腦的唯讀複本；`Florence_victim/` 已在 2026-11-27 重新產生，不再是原檔。

- **原因：** 原報告產生後，Exp3 又替換了一個樣本（`實驗/程式/Florence/攻擊/florence_victim/replace_exp3_one.py`），報告沒有重產。原報告的 Exp3 C1 為 540 筆、攻擊成功 999 張；現在的計分表是 541 筆、1,000 張，多出題號 132288000。
- **做法：** 用 `重算工具/rerun_florence_routing.py` 呼叫本資料夾未修改的 `build_report.py`，讀取 repo 內的 `florence_victim/<exp>/florence2/vqa_florence2/rows_scored.csv`。圖片不在 repo，所以不複製圖片，只保留原本的圖片路徑格式。
- **變動：**
  - Exp3 C1 改為 541 筆、185 張照片；C1～C5 百分比分母改為 1,000。
  - `all_categorized_rows.csv` 多 1 列。
  - 所有 CSV 多出 `ground_truth` 欄（新版計分表才有）。
  - `risk` 有浮點誤差（最大 1.7e-13）。
  - Exp1、Exp2 的分類與計數完全不變。
- **Exp1 與 Exp2 相同是正常的：** Florence_victim 用白盒 PGD 攻擊，Exp1 的 1,000 題全部攻擊成功，Exp2 是從 Exp1 挑出攻擊成功的 1,000 題，所以兩者是同一批資料（見 `文件/成果報告/2026-10-26_Florence三實驗.md` 第 5 行）。

## 類別代碼

- **C1**：攻擊成功→低風險→二階後仍攻擊成功
- **C2**：攻擊成功→中風險→二階後仍攻擊成功
- **C3**：攻擊成功→低風險→二階後防禦成功
- **C4**：攻擊成功→中風險→二階後防禦成功
- **C5**：攻擊成功→高風險（一階）
- **C6**：乾淨圖→中風險→二階後變攻擊成功
- **C7**：乾淨圖→高風險→二階後變攻擊成功

## 產生摘要

```json
[
  {
    "study": "BLIP_victim",
    "n_summary_rows": 14
  },
  {
    "study": "Florence_victim",
    "n_summary_rows": 14
  }
]
```