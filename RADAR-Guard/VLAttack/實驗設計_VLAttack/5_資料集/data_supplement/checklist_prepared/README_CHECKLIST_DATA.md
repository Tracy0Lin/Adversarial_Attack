# 清單用數據包（不覆蓋原版）

產生時間：2026-09-26 13:15 UTC  
輸出目錄：`data_supplement/checklist_prepared/`

## ① 樣本流向表

- `exp1_sample_flow_summary.csv` / `exp2_sample_flow_summary.csv`：逐階段 ASR／殘存率／維持率
- `exp1_routing_breakdown.csv` / `exp2_routing_breakdown.csv`：S1 各風險段路由 + 正誤答
- `sample_flow_routing_combined.csv`：兩實驗 Attack/Clean 路由長表

## ② 完整流程串接驗證

- `pipeline_recompute_vs_saved.csv`：由 CSV 重算 E2E 偵測，與原 `detector_comparison.csv` 對照
- `filter_method_mid_vs_e2e.csv`：濾波方法 **Mid 池** vs **路由後 2000 列**
- `mitigation_denominator_exp1.csv`：**1000 張** vs 舊 **462 成功子集** 口徑對照

## ③ 情境對照（自然 vs 壓力）

- `scenario_natural_vs_stress.csv`：攔截率、殘存 ASR、Stage 貢獻分解

### 重點數字（偵測 flip_count≥1）

- **exp1_natural**：Stage1 TPR=28.9% → E2E TPR=47.4% (+18.5%)；FPR 5.7%→8.7%；路由緩解殘存 ASR=25.1%
- **exp2_stress**：Stage1 TPR=43.1% → E2E TPR=72.3% (+29.2%)；FPR 14.1%→18.6%；路由緩解殘存 ASR=40.9%

### 差異主要來自哪一階段？

見 `scenario_stage_attribution.csv`（High 路由差、Stage1 TPR 差、Stage2 增量差）。

## 限制（需 GPU 才補）

Mid 探針 `sss_per_theta_raw.csv` 僅含 **中風險列**；對 Low/High 的攻擊圖 **沒有** 糊後 VQA。
因此「濾波方法獨立測試」若要求 **1000 張攻擊圖皆跑 flip**，需另跑 VQA+blur；
本包已用 **1000 張口徑的緩解 ASR**（`all_1000_attack_*` / `routed_*`）與 **E2E 路由偵測** 補齊可算部分。
