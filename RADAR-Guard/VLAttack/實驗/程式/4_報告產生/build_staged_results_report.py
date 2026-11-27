"""Stage-by-stage experiment report (Exp1 / Exp2) with GT-based Stage-2 attack rate.

Stage-2 *attack rate* (mitigation): VQA answer != ground truth (correct_ans),
i.e. direct BLIP wrong-answer rate — NOT flip vs pred_ref alone.

Writes (does not overwrite other reports):
  data_supplement/checklist_prepared/EXPERIMENT_STAGED_RESULTS_ANALYSIS.md
  data_supplement/checklist_prepared/exp1_stages_gt.csv
  data_supplement/checklist_prepared/exp2_stages_gt.csv
  data_supplement/checklist_prepared/exp1_exp2_final_comparison_gt.csv

Run:
  python BLIP_attack/output/VQA/build_staged_results_report.py
"""
from __future__ import annotations

import importlib.util
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "data_supplement" / "checklist_prepared"

_spec = importlib.util.spec_from_file_location(
    "checklist", HERE / "build_checklist_analytics.py"
)
chk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(chk)

EXP1 = {
    "key": "exp1",
    "title": "實驗一（混合流量）",
    "root": HERE / "steps100_alpha0.002_run1000",
    "manifest": HERE / "exp1_mixed1000" / "dataset_manifest.csv",
    "auroc_p1": 0.6655,
}
EXP2 = {
    "key": "exp2",
    "title": "實驗二（僅攻擊成功）",
    "root": HERE / "exp2_success1000",
    "manifest": HERE / "exp2_success1000" / "dataset_manifest.csv",
    "auroc_p1": 0.7212,
}


def wrong_vs_gt(pred: pd.Series, gt: pd.Series) -> pd.Series:
    return pred.astype(str) != gt.astype(str)


def compute_stages(meta: dict) -> pd.DataFrame:
    root = meta["root"]
    manifest = chk.load_manifest(meta["manifest"])
    stage1 = chk.load_stage1(root)
    sigma_mit = chk.load_theta_star(root)
    thr = chk.load_flip_threshold(root)
    flip_flags = chk.flip_count_flags(root, thr)

    gt = manifest.set_index("image_id")["ground_truth"].astype(str)
    adv = manifest.set_index("image_id")["adversarial_prediction"].astype(str)
    n = len(manifest)

    raw = pd.read_csv(root / "sss_per_theta_raw.csv")
    raw["image_id"] = raw["image_id"].astype(str)
    mid_atk = raw[(raw["sigma"] == sigma_mit) & (raw["label"] == 1) & (raw["kind"] == "attack")]
    mid_clean = raw[(raw["sigma"] == sigma_mit) & (raw["label"] == 0) & (raw["kind"] == "clean")]
    den = mid_atk.set_index("image_id")

    s0_wrong = wrong_vs_gt(adv, gt)
    s0_n = int(s0_wrong.sum())
    s0_rate = float(s0_wrong.mean())

    ori_ok = manifest["original_prediction"].astype(str) == manifest["ground_truth"].astype(str)
    rows = [
        _row(meta["key"], "S0", "基線（對抗圖，無防禦）", "attack_wrong_vs_gt", n, s0_n, s0_rate, "VQA(adv)≠標準答案"),
        _row(
            meta["key"],
            "S0",
            "基線（原圖，無防禦）",
            "clean_correct_vs_gt",
            n,
            int(ori_ok.sum()),
            float(ori_ok.mean()),
            "VQA(ori)=標準答案",
        ),
    ]

    s1_atk = stage1[stage1["label"] == 1]
    s1_clean = stage1[stage1["label"] == 0]
    for bin_name in ("low", "mid", "high"):
        g = s1_atk[s1_atk["risk_bin"] == bin_name]
        ids = g["image_id"]
        w = wrong_vs_gt(adv.reindex(ids), gt.reindex(ids))
        rows.append(
            _row(
                meta["key"],
                "S1",
                f"TC-VAD 路由（Attack→{bin_name}）",
                f"attack_route_{bin_name}_n",
                n,
                len(g),
                len(g) / n,
                "路由後 VQA 仍為對抗圖輸出",
            )
        )
        rows.append(
            _row(
                meta["key"],
                "S1",
                f"Attack {bin_name} 內攻擊率(vs GT)",
                f"attack_wrong_in_{bin_name}",
                len(g),
                int(w.sum()),
                float(w.mean()) if len(g) else float("nan"),
                "該 bin 內 adv 答案≠GT",
            )
        )

    for bin_name in ("low", "mid", "high"):
        g = s1_clean[s1_clean["risk_bin"] == bin_name]
        rows.append(
            _row(
                meta["key"],
                "S1",
                f"TC-VAD 路由（Clean→{bin_name}）",
                f"clean_route_{bin_name}_n",
                n,
                len(g),
                len(g) / n,
                "僅分流",
            )
        )

    rows.append(
        _row(
            meta["key"],
            "S1",
            "TC-VAD 分類（P1 OOF）",
            "auroc_p1",
            2000,
            None,
            meta["auroc_p1"],
            "2000 列偵測 AUROC",
        )
    )

    wb = wrong_vs_gt(mid_atk["pred_ref"], mid_atk["correct_ans"])
    wa = wrong_vs_gt(mid_atk["pred_denoised"], mid_atk["correct_ans"])
    rows.append(
        _row(
            meta["key"],
            "S2_mit",
            f"Mid 池緩解（5×5 σ={sigma_mit:g}，煳前）",
            "mid_attack_wrong_before",
            len(mid_atk),
            int(wb.sum()),
            float(wb.mean()),
            "Mid 對抗圖 VQA≠GT",
        )
    )
    rows.append(
        _row(
            meta["key"],
            "S2_mit",
            f"Mid 池緩解（5×5 σ={sigma_mit:g}，煳後）",
            "mid_attack_wrong_after",
            len(mid_atk),
            int(wa.sum()),
            float(wa.mean()),
            "煳後 BLIP 答案≠GT（Stage2 攻擊率定義）",
        )
    )
    rows.append(
        _row(
            meta["key"],
            "S2_mit",
            f"Mid 池緩解維持率(vs Mid 煳前)",
            "mid_retention",
            len(mid_atk),
            None,
            float(wa.mean() / wb.mean()) if wb.mean() else float("nan"),
            "wrong_after / wrong_before",
        )
    )

    after_routed = []
    for r in s1_atk.merge(manifest, left_on="image_id", right_on="image_id").itertuples():
        g = str(r.ground_truth)
        if r.risk_bin == "high":
            after_routed.append(False)
        elif r.risk_bin == "low":
            after_routed.append(str(r.adversarial_prediction) != g)
        else:
            q = r.image_id
            if q in den.index:
                after_routed.append(str(den.loc[q, "pred_denoised"]) != g)
            else:
                after_routed.append(str(r.adversarial_prediction) != g)
    ar = pd.Series(after_routed)
    rows.append(
        _row(
            meta["key"],
            "S2_mit",
            "全流程緩解（Low 放行/Mid 煳/High 阻擋）",
            "routed_attack_wrong_vs_gt",
            n,
            int(ar.sum()),
            float(ar.mean()),
            "High 視為恢復正確；Mid 用煳後≠GT",
        )
    )
    rows.append(
        _row(
            meta["key"],
            "S2_mit",
            "全流程緩解維持率(vs S0)",
            "routed_retention_vs_s0",
            n,
            None,
            float(ar.mean() / s0_rate) if s0_rate else float("nan"),
            "殘存攻擊率 / S0 攻擊率",
        )
    )

    clean_harm = wrong_vs_gt(mid_clean["pred_denoised"], mid_clean["correct_ans"])
    rows.append(
        _row(
            meta["key"],
            "S2_mit",
            f"Mid 乾淨圖副作用（σ={sigma_mit:g}）",
            "mid_clean_wrong_after_blur",
            len(mid_clean),
            int(clean_harm.sum()),
            float(clean_harm.mean()),
            "原圖本對；煳後 BLIP≠GT",
        )
    )

    y = stage1["label"].values == 1
    det = chk.rates(y, chk.end_to_end_flagged(stage1, flip_flags))
    s1o = chk.rates(y, chk.end_to_end_flagged(stage1, {}))
    s1_atk_m = s1_atk.merge(manifest[["image_id", "ground_truth", "adversarial_prediction"]], on="image_id")
    s1_atk_m["blocked"] = chk.end_to_end_flagged(s1_atk_m, flip_flags)
    residual = s1_atk_m[~s1_atk_m["blocked"]]
    res_wrong = (residual["adversarial_prediction"].astype(str) != residual["ground_truth"].astype(str)).sum()

    rows.extend(
        [
            _row(meta["key"], "S2_det", "僅 Stage1 High 攔截", "stage1_TPR", n, s1o["TP"], s1o["TPR"], "偵測 TPR"),
            _row(meta["key"], "S2_det", "僅 Stage1", "stage1_FPR", n, s1o["FP"], s1o["FPR"], "偵測 FPR"),
            _row(
                meta["key"],
                "S2_det",
                f"E2E 偵測 flip_count≥{thr}",
                "e2e_TPR",
                n,
                det["TP"],
                det["TPR"],
                "Stage1 路由 + Mid 翻轉",
            ),
            _row(meta["key"], "S2_det", f"E2E flip_count≥{thr}", "e2e_FPR", n, det["FP"], det["FPR"], "乾淨誤攔"),
            _row(
                meta["key"],
                "S2_det",
                "E2E 放行後仍攻擊(vs GT)",
                "residual_wrong_after_det",
                n,
                int(res_wrong),
                float(res_wrong / n),
                "FN 子集上 adv 仍≠GT",
            ),
            _row(
                meta["key"],
                "S2_det",
                "E2E Youden J",
                "youden_j",
                2000,
                None,
                det["youden_j"],
                "TPR−FPR",
            ),
        ]
    )

    return pd.DataFrame(rows)


def _row(exp, stage, label, metric_id, denom, count, rate, note):
    return {
        "experiment": exp,
        "stage": stage,
        "label_zh": label,
        "metric_id": metric_id,
        "denominator_n": denom,
        "count": count,
        "rate": rate,
        "note": note,
    }


def pct(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    return f"{100 * float(x):.2f}%"


def pick(df: pd.DataFrame, metric_id: str):
    r = df[df["metric_id"] == metric_id].iloc[0]
    return r["count"], r["rate"]


def section_markdown(title: str, df: pd.DataFrame) -> list[str]:
    lines = [f"### {title}", "", "| 階段 | 說明 | 分子 | 分母 | 比例 |", "|------|------|------|------|------|"]
    order = [
        ("S0", "attack_wrong_vs_gt"),
        ("S0", "clean_correct_vs_gt"),
        ("S1", "attack_route_low_n"),
        ("S1", "attack_wrong_in_low"),
        ("S1", "attack_route_mid_n"),
        ("S1", "attack_wrong_in_mid"),
        ("S1", "attack_route_high_n"),
        ("S1", "attack_wrong_in_high"),
        ("S1", "auroc_p1"),
        ("S2_mit", "mid_attack_wrong_before"),
        ("S2_mit", "mid_attack_wrong_after"),
        ("S2_mit", "mid_retention"),
        ("S2_mit", "routed_attack_wrong_vs_gt"),
        ("S2_mit", "routed_retention_vs_s0"),
        ("S2_mit", "mid_clean_wrong_after_blur"),
        ("S2_det", "stage1_TPR"),
        ("S2_det", "stage1_FPR"),
        ("S2_det", "e2e_TPR"),
        ("S2_det", "e2e_FPR"),
        ("S2_det", "residual_wrong_after_det"),
        ("S2_det", "youden_j"),
    ]
    for stage, mid in order:
        sub = df[(df["stage"] == stage) & (df["metric_id"] == mid)]
        if sub.empty:
            continue
        r = sub.iloc[0]
        cnt = "—" if pd.isna(r["count"]) else str(int(r["count"]) if r["count"] == r["count"] else r["count"])
        if mid == "auroc_p1" or mid == "youden_j":
            prop = f"{float(r['rate']):.4f}"
        else:
            prop = pct(r["rate"])
        lines.append(
            f"| {r['stage']} | {r['label_zh']} | {cnt} | {int(r['denominator_n'])} | {prop} |"
        )
    lines.append("")
    return lines


def comparison_table(d1: pd.DataFrame, d2: pd.DataFrame) -> pd.DataFrame:
    ids = [
        "attack_wrong_vs_gt",
        "routed_attack_wrong_vs_gt",
        "mid_attack_wrong_after",
        "e2e_TPR",
        "e2e_FPR",
        "residual_wrong_after_det",
        "mid_clean_wrong_after_blur",
        "auroc_p1",
    ]
    rows = []
    for mid in ids:
        a = d1[d1["metric_id"] == mid].iloc[0]
        b = d2[d2["metric_id"] == mid].iloc[0]
        rows.append(
            {
                "metric_id": mid,
                "label": a["label_zh"],
                "exp1_rate": a["rate"],
                "exp2_rate": b["rate"],
                "delta_exp2_minus_exp1": float(b["rate"]) - float(a["rate"])
                if pd.notna(a["rate"]) and pd.notna(b["rate"])
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def analysis_text(cmp: pd.DataFrame) -> list[str]:
    def g(mid):
        return cmp[cmp["metric_id"] == mid].iloc[0]

    lines = [
        "## 綜合分析",
        "",
        "### 指標定義（本報告統一）",
        "",
        "- **攻擊率**：BLIP 給出的答案 **不等於題目標準答案（ground truth）**。"
        "  原圖在本資料集皆答對，故 S0 與過去 log 的「對抗≠原圖預測」數值一致。",
        "- **Stage 2 緩解攻擊率**：在 Mid 或全流程路由下，對 **煳後圖** 再問 BLIP，",
        "  以 **`pred_denoised ≠ correct_ans`** 計算（與 `sss_per_theta_raw.csv` 的 `fooled_after` 一致）。",
        "- **Stage 2 偵測**：報 **TPR / FPR / J**；「放行後仍攻擊」= 未被攔截的對抗圖中，",
        "  **對抗圖 VQA 答案仍 ≠ GT**（非翻轉探針本身）。",
        "",
        "### 實驗一（混合流量）",
        "",
        f"- S0 攻擊率 **{pct(g('attack_wrong_vs_gt')['exp1_rate'])}**（462/1000），代表近半對抗樣本未改答案，",
        "  後續防禦同時面對「真成功攻擊」與「高風險但未改答」兩類。",
        f"- Stage1 將攻擊分散：High **289** 張（其中 **158** 張仍≠GT），是單靠 Stage1 攔截的主要來源。",
        f"- Stage2 **緩解**（Mid 煳 σ=0.8）：Mid 內攻擊率 **42.2%→28.6%**（vs GT）；",
        f"  全流程殘存 **{pct(g('routed_attack_wrong_vs_gt')['exp1_rate'])}**，",
        f"  相對 S0 維持率 **54.3%**。",
        f"- Stage2 **偵測**（flip≥1）：TPR **{pct(g('e2e_TPR')['exp1_rate'])}**，FPR **{pct(g('e2e_FPR')['exp1_rate'])}**；",
        f"  放行後仍≠GT **{pct(g('residual_wrong_after_det')['exp1_rate'])}**（165/1000）。",
        f"- 乾淨圖 Mid 煳後被改答 **{pct(g('mid_clean_wrong_after_blur')['exp1_rate'])}**（11/395），副作用可控。",
        "",
        "### 實驗二（壓力測試，1000 皆攻擊成功）",
        "",
        f"- S0 攻擊率 **100%**；Stage1 更多攻擊進 **High（431）**，Stage1 單獨 TPR 明顯高於實驗一。",
        f"- Mid 緩解：煳前 100%→煳後 **54.3%**（仍≠GT）；全流程殘存 **{pct(g('routed_attack_wrong_vs_gt')['exp2_rate'])}**。",
        f"- 偵測 TPR **{pct(g('e2e_TPR')['exp2_rate'])}** 但 FPR 升至 **{pct(g('e2e_FPR')['exp2_rate'])}**；",
        "  這是壓力情境下「抓得更多、也誤殺更多」的權衡。",
        "",
        "### 實驗一 vs 實驗二（為何數字差這麼多？）",
        "",
        f"- **High 路由差 +142 張**（431−289）：Stage1 對「已成功攻擊」分數偏高 → 實驗二 Stage1 TPR 高 **+14.2 pp**。",
        f"- **E2E TPR 差 +{100*(g('e2e_TPR')['delta_exp2_minus_exp1']):.1f} pp**；其中 Stage2 翻轉追加部分約 **+10.7 pp**，",
        "  其餘來自 Stage1。",
        f"- **殘存緩解攻擊率** 實驗二仍 **{pct(g('routed_attack_wrong_vs_gt')['exp2_rate'])}** vs 實驗一 **{pct(g('routed_attack_wrong_vs_gt')['exp1_rate'])}**：",
        "  因 Low 路徑在實驗二上全是成功攻擊且不做煳，拉低緩解效果。",
        "- **論文／部署敘事**：以 **實驗一** 為主（真實混合流量）；**實驗二** 用於解釋機制上限，",
        "  不可直接把 72% TPR 當上線期望。",
        "",
    ]
    return lines


def get_stage(df: pd.DataFrame, metric_id: str) -> dict:
    r = df[df["metric_id"] == metric_id].iloc[0]
    return r


def fmt_count_rate(count, rate, *, is_pct=True) -> str:
    if pd.isna(count) and pd.isna(rate):
        return "—"
    c = "—" if pd.isna(count) else str(int(count))
    if pd.isna(rate):
        return c
    if is_pct:
        return f"{c}（{pct(rate)}）"
    return f"{c}（{float(rate):.4f}）"


def build_master_pipeline_table(stages: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """One summary table: 初始 → Stage1 → Stage2 → 全流程（per experiment)."""
    exp = meta["key"]
    title = meta["title"]
    s0a = get_stage(stages, "attack_wrong_vs_gt")
    s0c = get_stage(stages, "clean_correct_vs_gt")
    r_low = get_stage(stages, "attack_route_low_n")
    r_mid = get_stage(stages, "attack_route_mid_n")
    r_high = get_stage(stages, "attack_route_high_n")
    cl_low = get_stage(stages, "clean_route_low_n")
    cl_mid = get_stage(stages, "clean_route_mid_n")
    cl_high = get_stage(stages, "clean_route_high_n")
    s1tpr = get_stage(stages, "stage1_TPR")
    s1fpr = get_stage(stages, "stage1_FPR")
    auroc = get_stage(stages, "auroc_p1")
    mit_mid = get_stage(stages, "mid_attack_wrong_after")
    mit_route = get_stage(stages, "routed_attack_wrong_vs_gt")
    mit_ret = get_stage(stages, "routed_retention_vs_s0")
    clean_harm = get_stage(stages, "mid_clean_wrong_after_blur")
    e2e_tpr = get_stage(stages, "e2e_TPR")
    e2e_fpr = get_stage(stages, "e2e_FPR")
    e2e_j = get_stage(stages, "youden_j")
    res_det = get_stage(stages, "residual_wrong_after_det")
    sigma = chk.load_theta_star(meta["root"])

    rows = [
        {
            "experiment": exp,
            "experiment_title": title,
            "phase_order": 0,
            "step_order": 0,
            "phase_id": "P0",
            "phase_name": "初始（無防禦）",
            "step": "BLIP 直接 VQA",
            "attack_summary": fmt_count_rate(s0a["count"], s0a["rate"]),
            "attack_metric": "攻擊率（答案≠GT）",
            "clean_summary": fmt_count_rate(s0c["count"], s0c["rate"]),
            "clean_metric": "原圖答對率（=GT）",
            "remark": "尚未經 TC-VAD／Stage2",
        },
        {
            "experiment": exp,
            "experiment_title": title,
            "phase_order": 1,
            "step_order": 1,
            "phase_id": "P1",
            "phase_name": "Stage 1｜TC-VAD",
            "step": "風險路由 Low / Mid / High",
            "attack_summary": f"Low {int(r_low['count'])}｜Mid {int(r_mid['count'])}｜High {int(r_high['count'])}",
            "attack_metric": "1000 對抗圖分流",
            "clean_summary": f"Low {int(cl_low['count'])}｜Mid {int(cl_mid['count'])}｜High {int(cl_high['count'])}",
            "clean_metric": "1000 乾淨圖分流",
            "remark": f"P1 OOF AUROC={auroc['rate']:.4f}；此階段不改 VQA 答案，攻擊率仍={pct(s0a['rate'])}",
        },
        {
            "experiment": exp,
            "experiment_title": title,
            "phase_order": 1,
            "step_order": 2,
            "phase_id": "P1",
            "phase_name": "Stage 1｜TC-VAD",
            "step": "僅 High 攔截（尚未 Stage2）",
            "attack_summary": fmt_count_rate(s1tpr["count"], s1tpr["rate"]),
            "attack_metric": "攔截率 TPR（對 1000 對抗）",
            "clean_summary": fmt_count_rate(s1fpr["count"], s1fpr["rate"]),
            "clean_metric": "誤攔 FPR（對 1000 乾淨）",
            "remark": "Low 放行、Mid 待 Stage2",
        },
        {
            "experiment": exp,
            "experiment_title": title,
            "phase_order": 2,
            "step_order": 1,
            "phase_id": "P2",
            "phase_name": "Stage 2｜緩解（SSS / 高斯 θ*）",
            "step": f"Mid 池 5×5 σ={sigma:g} 煳後再問 BLIP（vs GT）",
            "attack_summary": fmt_count_rate(mit_mid["count"], mit_mid["rate"]),
            "attack_metric": f"Mid 內攻擊率（n={int(mit_mid['denominator_n'])}）",
            "clean_summary": fmt_count_rate(clean_harm["count"], clean_harm["rate"]),
            "clean_metric": f"Mid 乾淨副作用（n={int(clean_harm['denominator_n'])}）",
            "remark": "僅中風險列執行煳圖；Low/High 對抗圖不在此池",
        },
        {
            "experiment": exp,
            "experiment_title": title,
            "phase_order": 2,
            "step_order": 2,
            "phase_id": "P2",
            "phase_name": "Stage 2｜偵測（答案翻轉）",
            "step": "Mid：多 σ 探針 flip_count≥1（θ_det）",
            "attack_summary": fmt_count_rate(e2e_tpr["count"], e2e_tpr["rate"]),
            "attack_metric": "E2E 攔截 TPR（含 S1 路由）",
            "clean_summary": fmt_count_rate(e2e_fpr["count"], e2e_fpr["rate"]),
            "clean_metric": "E2E 誤攔 FPR",
            "remark": f"Youden J={e2e_j['rate']:.4f}",
        },
        {
            "experiment": exp,
            "experiment_title": title,
            "phase_order": 3,
            "step_order": 1,
            "phase_id": "P3",
            "phase_name": "全流程｜緩解線",
            "step": "Low 放行 ＋ Mid 煳（θ*）＋ High 阻擋",
            "attack_summary": fmt_count_rate(mit_route["count"], mit_route["rate"]),
            "attack_metric": "殘存攻擊率（≠GT）",
            "clean_summary": "—",
            "clean_metric": "（緩解線不另計乾淨 ASR）",
            "remark": f"相對 P0 維持率 {pct(mit_ret['rate'])}",
        },
        {
            "experiment": exp,
            "experiment_title": title,
            "phase_order": 3,
            "step_order": 2,
            "phase_id": "P3",
            "phase_name": "全流程｜偵測線",
            "step": "S1 路由 ＋ S2 翻轉（flip_count≥1）",
            "attack_summary": fmt_count_rate(res_det["count"], res_det["rate"]),
            "attack_metric": "放行後仍攻擊（≠GT）",
            "clean_summary": fmt_count_rate(e2e_fpr["count"], e2e_fpr["rate"]),
            "clean_metric": "端到端 FPR",
            "remark": f"攔截 TPR={pct(e2e_tpr['rate'])}；J={e2e_j['rate']:.4f}",
        },
    ]
    return pd.DataFrame(rows)


def master_table_markdown(df: pd.DataFrame, heading: str) -> list[str]:
    lines = [
        f"## {heading}",
        "",
        "管線：**P0 初始 VQA → P1 TC-VAD 分流 → P2 緩解／偵測 → P3 全流程（緩解線與偵測線）**",
        "",
        "| 階段 | 步驟 | 攻擊（1000 對抗） | 乾淨（1000 原圖） | 備註 |",
        "|------|------|-------------------|-------------------|------|",
    ]
    for _, r in df.sort_values(["phase_order", "step_order"]).iterrows():
        lines.append(
            f"| **{r['phase_name']}** | {r['step']} | "
            f"{r['attack_metric']}：{r['attack_summary']} | "
            f"{r['clean_metric']}：{r['clean_summary']} | {r['remark']} |"
        )
    lines.append("")
    return lines


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d1 = compute_stages(EXP1)
    d2 = compute_stages(EXP2)
    d1.to_csv(OUT / "exp1_stages_gt.csv", index=False, encoding="utf-8-sig")
    d2.to_csv(OUT / "exp2_stages_gt.csv", index=False, encoding="utf-8-sig")
    cmp = comparison_table(d1, d2)
    cmp.to_csv(OUT / "exp1_exp2_final_comparison_gt.csv", index=False, encoding="utf-8-sig")

    m1 = build_master_pipeline_table(d1, EXP1)
    m2 = build_master_pipeline_table(d2, EXP2)
    m1.to_csv(OUT / "exp1_master_pipeline_table.csv", index=False, encoding="utf-8-sig")
    m2.to_csv(OUT / "exp2_master_pipeline_table.csv", index=False, encoding="utf-8-sig")
    (OUT / "exp1_master_pipeline_table.md").write_text(
        "\n".join(master_table_markdown(m1, "實驗一｜三階段流程總表")),
        encoding="utf-8",
    )
    (OUT / "exp2_master_pipeline_table.md").write_text(
        "\n".join(master_table_markdown(m2, "實驗二｜三階段流程總表")),
        encoding="utf-8",
    )

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    md = [
        "# 實驗一／實驗二：分階段結果與對照分析",
        "",
        f"更新：{ts}  ",
        "數據來源：`tcvad_stage1_4_academic_dataset.csv`、`sss_per_theta_raw.csv`、`dataset_manifest.csv`  ",
        "CSV：`exp1_stages_gt.csv`、`exp2_stages_gt.csv`、`exp1_exp2_final_comparison_gt.csv`",
        "**總表**：`exp1_master_pipeline_table.md` / `exp2_master_pipeline_table.md`（及同名 `.csv`）",
        "",
        "---",
        "",
    ]
    md.extend(master_table_markdown(m1, "實驗一｜三階段流程總表"))
    md += ["---", ""]
    md.extend(master_table_markdown(m2, "實驗二｜三階段流程總表"))
    md += ["---", "", "## 實驗一結果（逐步明細）", ""]
    md.extend(section_markdown(EXP1["title"], d1))
    md += ["---", "", "## 實驗二結果（逐步）", ""]
    md.extend(section_markdown(EXP2["title"], d2))
    md += ["---", "", "## 最終對照：實驗一 vs 實驗二", ""]
    md.append("| 指標 | 實驗一 | 實驗二 | 差（二−一） |")
    md.append("|------|--------|--------|-------------|")
    for _, r in cmp.iterrows():
        if r["metric_id"] in ("auroc_p1", "youden_j"):
            e1, e2, d = f"{r['exp1_rate']:.4f}", f"{r['exp2_rate']:.4f}", f"{r['delta_exp2_minus_exp1']:+.4f}"
        else:
            e1, e2, d = pct(r["exp1_rate"]), pct(r["exp2_rate"]), f"{100*r['delta_exp2_minus_exp1']:+.2f} pp"
        md.append(f"| {r['label']} | {e1} | {e2} | {d} |")
    md.append("")
    md.extend(analysis_text(cmp))
    (OUT / "EXPERIMENT_STAGED_RESULTS_ANALYSIS.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {OUT / 'EXPERIMENT_STAGED_RESULTS_ANALYSIS.md'}")


if __name__ == "__main__":
    main()
