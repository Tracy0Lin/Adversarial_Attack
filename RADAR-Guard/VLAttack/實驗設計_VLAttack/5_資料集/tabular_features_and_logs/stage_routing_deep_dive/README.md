# 防禦路由七類深入分析（BLIP + Florence）

由 `build_report.py` 產生。每類別每實驗保留 **5–10** 個樣本（優先不同 `photo_id`），
圖片在子目錄 `樣本_qid/` 下（`_*_ori.png` / `_*_atk.png`）。

## 目錄

- [COMPARISON.md](../../../../成果報告/2026-10-24_stage_routing_COMPARISON.md) — BLIP vs Florence 對照摘要
- [BLIP_victim/REPORT.md](BLIP_victim/REPORT.md) — 含樣本表與圖片路徑
- [Florence_victim/REPORT.md](Florence_victim/REPORT.md) — 含樣本表與圖片路徑
- `build_report.py` — 可重跑以更新 CSV / 樣本圖

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