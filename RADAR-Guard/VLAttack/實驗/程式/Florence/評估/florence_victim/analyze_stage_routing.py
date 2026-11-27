"""Classify rows into 7 defense-routing outcome buckets (full pipeline) for exp1–3."""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
FEATURES = ["d_center", "d_iqr", "d_tail", "d_q95"]
VIS_ROOT = {
    "exp1": ROOT / "exp1_mixed1000" / "visual_results",
    "exp2": ROOT / "exp2_success1000" / "visual_results",
    "exp3": ROOT / "exp3_success1000" / "visual_results",
}
ROWS = {
    exp: ROOT / exp / "florence2" / "vqa_florence2" / "rows_scored.csv"
    for exp in ("exp1", "exp2", "exp3")
}
OUT = ROOT / "routing_analysis"
EXAMPLES_PER = 5

CAT_LABELS = {
    "C1": "攻擊成功→低風險→二階後仍攻擊成功",
    "C2": "攻擊成功→中風險→二階後仍攻擊成功",
    "C3": "攻擊成功→低風險→二階後防禦成功",
    "C4": "攻擊成功→中風險→二階後防禦成功",
    "C5": "攻擊成功→高風險（一階）",
    "C6": "乾淨圖→中風險→二階後變攻擊成功",
    "C7": "乾淨圖→高風險→二階後變攻擊成功",
}


def load_exp(exp: str) -> pd.DataFrame:
    df = pd.read_csv(ROWS[exp], dtype={"image_id": str}, keep_default_na=False)
    for c in FEATURES + ["rewrite_rate", "risk"]:
        df[c] = pd.to_numeric(df[c])
    df["label"] = df["label"].astype(int)
    df["fooled_before"] = df["fooled_before"].astype(bool)
    df["blocked_full"] = df["blocked_full"].astype(bool)
    df["exp"] = exp
    victim = df["original_prediction"]
    df["atk_still"] = (df["label"] == 1) & df["fooled_before"] & ~df["blocked_full"] & (
        df["served_full"] != victim
    )
    df["clean_becomes_attack"] = (df["label"] == 0) & ~df["blocked_full"] & (
        df["served_full"] != victim
    )
    return df


def assign_category(row) -> str:
    b = row["bin"]
    if row["label"] == 1 and row["fooled_before"]:
        if b == "high":
            return "C5"
        if b == "low" and row["atk_still"]:
            return "C1"
        if b == "mid" and row["atk_still"]:
            return "C2"
        if b == "low" and not row["atk_still"]:
            return "C3"
        if b == "mid" and not row["atk_still"]:
            return "C4"
        return ""
    if row["label"] == 0:
        if b == "mid" and row["clean_becomes_attack"]:
            return "C6"
        if b == "high" and row["clean_becomes_attack"]:
            return "C7"
    return ""


def feature_profile(sub: pd.DataFrame) -> dict:
    if sub.empty:
        return {}
    out = {f: float(sub[f].mean()) for f in FEATURES}
    out["rewrite_rate_mean"] = float(sub["rewrite_rate"].mean())
    out["risk_mean"] = float(sub["risk"].mean())
    return out


def copy_examples(sub: pd.DataFrame, exp: str, cat: str):
    dest = OUT / "examples" / exp / cat
    dest.mkdir(parents=True, exist_ok=True)
    vis = VIS_ROOT[exp]
    if not vis.is_dir():
        return []
    copied = []
    for _, r in sub.head(EXAMPLES_PER).iterrows():
        qid = r["image_id"]
        for suffix in ("_1_ori.png", "_2_atk.png"):
            src = vis / f"{qid}{suffix}"
            if src.is_file():
                dst = dest / f"{qid}{suffix}"
                shutil.copy2(src, dst)
                copied.append(str(dst.name))
    return copied


def summarize(df: pd.DataFrame) -> dict:
    full = df.copy()
    full["category"] = full.apply(assign_category, axis=1)
    tagged = full[full["category"] != ""]
    rows = []
    photos = []
    for exp in full["exp"].unique():
        e = full[full["exp"] == exp]
        fooled_base = e[(e["label"] == 1) & e["fooled_before"]]
        clean_base = e[e["label"] == 0]
        for cat in CAT_LABELS:
            sub = tagged[(tagged["exp"] == exp) & (tagged["category"] == cat)]
            if sub.empty:
                continue
            prof = feature_profile(sub)
            row_n = len(sub)
            photo_n = sub["photo_id"].nunique()
            rows.append(
                {
                    "exp": exp,
                    "category": cat,
                    "label_zh": CAT_LABELS[cat],
                    "n_rows": row_n,
                    "n_photos": photo_n,
                    "pct_of_fooled_before": round(100 * row_n / len(fooled_base), 2)
                    if cat.startswith("C") and cat <= "C5" and len(fooled_base)
                    else None,
                    "pct_of_clean": round(100 * row_n / len(clean_base), 2)
                    if cat in ("C6", "C7") and len(clean_base)
                    else None,
                    **prof,
                }
            )
            photos.append(
                {
                    "exp": exp,
                    "category": cat,
                    "photo_ids": sorted(sub["photo_id"].unique().tolist())[:50],
                }
            )
            copy_examples(sub, exp, cat)
    return {"counts": rows, "photo_samples": photos, "labeled_rows": len(tagged)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    parts = [load_exp(e) for e in ("exp1", "exp2", "exp3")]
    all_df = pd.concat(parts, ignore_index=True)
    all_df["category"] = all_df.apply(assign_category, axis=1)
    tagged = all_df[all_df["category"] != ""].copy()
    tagged.to_csv(OUT / "all_categorized_rows.csv", index=False, encoding="utf-8-sig")

    summary = summarize(all_df)
    with open(OUT / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    counts = pd.DataFrame(summary["counts"])
    counts.to_csv(OUT / "counts_by_exp_category.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# Florence 受害者實驗：防禦路由七類情境",
        "",
        "資料：`rows_scored.csv`（full pipeline：`blocked_full` / `served_full`）",
        "「攻擊成功」= `fooled_before`（對抗圖 VQA ≠ 乾淨正確答案）；",
        "「二階後仍攻擊成功」= 未封鎖且 `served_full` ≠ `original_prediction`；",
        "「乾淨圖變攻擊成功」= label=0 且未封鎖但答案被改錯。",
        "",
    ]
    for exp in ("exp1", "exp2", "exp3"):
        sub = counts[counts["exp"] == exp]
        if sub.empty:
            lines.append(f"## {exp}\n（無符合列）\n")
            continue
        lines.append(f"## {exp}\n")
        lines.append("| 類別 | 說明 | 行數 | 照片數 | 佔 fooled/clean% | d_center | rewrite |")
        lines.append("|------|------|------|--------|------------------|----------|---------|")
        for _, r in sub.iterrows():
            pct = r["pct_of_fooled_before"]
            if pd.isna(pct):
                pct = r["pct_of_clean"]
            pct_s = f"{pct:.1f}%" if pd.notna(pct) else "—"
            lines.append(
                f"| {r['category']} | {r['label_zh']} | {int(r['n_rows'])} | {int(r['n_photos'])} | {pct_s} | "
                f"{r.get('d_center', 0):.4f} | {r.get('rewrite_rate_mean', 0):.3f} |"
            )
        lines.append("")
    lines.append("範例圖（若本機有 visual_results）：`routing_analysis/examples/`")
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT}")
    print(counts.to_string(index=False))


if __name__ == "__main__":
    main()
