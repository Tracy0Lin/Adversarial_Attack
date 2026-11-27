"""Collect every summary.json / pipeline_variants.csv into RESULTS_tables.md."""
import json

import pandas as pd

from common import CAPTIONERS, DATA_ROOT, EXPERIMENTS

NAMES = {"blip": "BLIP", "vitgpt2": "ViT-GPT2", "git": "GIT-large-COCO", "gemma3": "gemma3:4b (LLM)"}
EXP_NAMES = {"exp1": "實驗一", "exp2": "實驗二", "exp3": "實驗三"}
VARIANT_NAMES = {
    "A0_no_defense": "A0 不防禦",
    "A1_stage1_high_block": "A1 只有第一階段（攔 High）",
    "A1b_stage1_mid_high_block": "A1′ 只有第一階段（攔 Mid+High）",
    "A2_stage2_only": "A2 只有第二階段（全部翻轉偵測 + θ\\* 緩解）",
    "A2b_stage2_mitigate_only": "A2′ 只有第二階段緩解（全部糊 θ\\*，不攔）",
    "A3_full_mitigate_only": "A3′ 完整路由，Mid 只緩解不攔",
    "A3_full_pipeline": "A3 完整流程",
}


def pct(x, d=1):
    return "—" if x is None or pd.isna(x) else f"{100 * x:.{d}f}%"


def num(x, d=3):
    return "—" if x is None or pd.isna(x) else f"{x:.{d}f}"


def ci(v):
    return "" if not isinstance(v, (list, tuple)) else f" [{100 * v[0]:.1f}, {100 * v[1]:.1f}]"


def load():
    out = {}
    for e in EXPERIMENTS:
        for c in CAPTIONERS:
            p = DATA_ROOT / e / c / "summary.json"
            if p.is_file():
                out[(e, c)] = (json.loads(p.read_text(encoding="utf-8")),
                               pd.read_csv(DATA_ROOT / e / c / "pipeline_variants.csv"))
    return out


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(lines)


def main():
    data = load()
    md = ["# 表格彙整（由 make_tables.py 產生）", ""]

    md += ["## 第一階段", ""]
    rows = []
    for (e, c), (s, _) in data.items():
        r, h = s["stage1_risk"], s["stage1_high_only"]
        b = s["bins"].get("attack", {})
        bc = s["bins"].get("clean", {})
        rows.append([EXP_NAMES[e], NAMES[c], num(r.get("auroc")), num(r.get("auprc")), pct(r.get("tpr_at_fpr_5")),
                     pct(r.get("tpr_at_fpr_1")), pct(s["mid_rate"]), pct(h["TPR"]), pct(h["FPR"]),
                     num(h["youden_j"]),
                     f"{b.get('low', 0)}/{b.get('mid', 0)}/{b.get('high', 0)}",
                     f"{bc.get('low', 0)}/{bc.get('mid', 0)}/{bc.get('high', 0)}" if bc else "—",
                     pct(s["caption_rewrite_rate"]["clean"]), pct(s["caption_rewrite_rate"]["attack"])])
    md += [table(["實驗", "caption 模型", "AUROC", "AUPRC", "TPR@FPR5%", "TPR@FPR1%", "Mid 率", "只攔 High TPR",
                  "只攔 High FPR", "J", "攻擊 L/M/H", "乾淨 L/M/H", "乾淨改寫率", "攻擊改寫率"], rows), ""]

    md += ["## 第二階段參數（Mid）", ""]
    rows = []
    for (e, c), (s, _) in data.items():
        f = s["mid_flip_at_theta_det"]
        rows.append([EXP_NAMES[e], NAMES[c], s["theta_source"], s["theta_star_mid"], num(s["sss_at_theta_star"]),
                     s["theta_det_mid"], pct(f.get("TPR")), pct(f.get("FPR")), pct(f.get("TPR_fooled_only"))])
    md += [table(["實驗", "caption 模型", "θ 來源", "θ\\*（σ）", "SSS", "θ_det（σ）", "Mid 翻轉 TPR", "Mid 翻轉 FPR",
                  "Mid 已騙子集 TPR"], rows), ""]

    md += ["## 完整流程（A3）", ""]
    rows = []
    for (e, c), (s, _) in data.items():
        f = s["full_pipeline"]
        rows.append([EXP_NAMES[e], NAMES[c], pct(f["TPR"]) + ci(f.get("TPR_ci95")),
                     pct(f["FPR"]) + ci(f.get("FPR_ci95")), num(f["youden_j"]), pct(f["precision"]), num(f["F1"]),
                     pct(f["balanced_acc"]), pct(f["asr_before"]), pct(f["effective_asr"]) + ci(f.get("effective_asr_ci95")),
                     pct(f["clean_service_rate"]), num(f["vqa_calls_per_row"], 2)])
    md += [table(["實驗", "caption 模型", "TPR [95% CI]", "FPR [95% CI]", "J", "Precision", "F1", "Balanced Acc",
                  "ASR 防禦前", "有效 ASR [95% CI]", "乾淨服務率", "VQA 次數/列"], rows), ""]

    for e in EXPERIMENTS:
        md += [f"## {EXP_NAMES[e]}：各流程變體", ""]
        rows = []
        for c in CAPTIONERS:
            if (e, c) not in data:
                continue
            v = data[(e, c)][1]
            for r in v.itertuples(index=False):
                rows.append([NAMES[c], VARIANT_NAMES[r.variant], pct(r.TPR), pct(r.FPR), num(r.youden_j),
                             num(r.F1), pct(r.effective_asr), pct(r.clean_service_rate),
                             "—" if pd.isna(r.theta_star) else r.theta_star,
                             "—" if pd.isna(r.theta_det) else r.theta_det, num(r.vqa_calls_per_row, 2)])
        if rows:
            md += [table(["caption 模型", "流程", "TPR", "FPR", "J", "F1", "有效 ASR", "乾淨服務率", "θ\\*", "θ_det",
                          "VQA 次數/列"], rows), ""]

    rows = []
    for c in CAPTIONERS:
        if ("exp1", c) not in data:
            continue
        sub = data[("exp1", c)][0].get("exp1_by_attack_outcome", {})
        for tag, label in (("attack_success", "成功攻擊"), ("attack_failed", "失敗攻擊")):
            if tag in sub:
                d = sub[tag]
                b = d["bins"]
                rows.append([NAMES[c], label, d["n"], num(d.get("stage1_auroc_vs_clean")), pct(d["stage1_high_TPR"]),
                             pct(d["full_TPR"]), f"{b['low']}/{b['mid']}/{b['high']}"])
    if rows:
        md += ["## 實驗一：成功與失敗攻擊分開看", "",
               table(["caption 模型", "子集", "n", "AUROC（對全部乾淨）", "只攔 High TPR", "完整流程 TPR", "L/M/H"], rows), ""]

    (DATA_ROOT / "RESULTS_tables.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Saved {DATA_ROOT / 'RESULTS_tables.md'} ({len(data)} combos)")


if __name__ == "__main__":
    main()
