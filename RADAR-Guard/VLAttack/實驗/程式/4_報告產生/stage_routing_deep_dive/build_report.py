"""BLIP + Florence victim: 7-way routing analysis with 5–10 image samples per bucket."""
import json
import shutil
from pathlib import Path

import pandas as pd

VQA = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
FEATURES = ["d_center", "d_iqr", "d_tail", "d_q95"]
MIN_SAMPLES = 5
MAX_SAMPLES = 10

CAT = {
    "C1": "攻擊成功→低風險→二階後仍攻擊成功",
    "C2": "攻擊成功→中風險→二階後仍攻擊成功",
    "C3": "攻擊成功→低風險→二階後防禦成功",
    "C4": "攻擊成功→中風險→二階後防禦成功",
    "C5": "攻擊成功→高風險（一階）",
    "C6": "乾淨圖→中風險→二階後變攻擊成功",
    "C7": "乾淨圖→高風險→二階後變攻擊成功",
}

STUDIES = {
    "BLIP_victim": {
        "title": "BLIP 受害者（multi_captioner，VQA=BLIP）",
        "rows": lambda exp: VQA / "multi_captioner" / exp / "blip" / "rows_scored.csv",
        "visual": {
            "exp1": VQA / "steps100_alpha0.002_run1000" / "visual_results",
            "exp2": VQA / "exp2_success1000" / "visual_results",
            "exp3": VQA / "exp3_success1000" / "visual_results",
        },
    },
    "Florence_victim": {
        "title": "Florence-2-large-ft 受害者（florence_victim，VQA=Florence）",
        "rows": lambda exp: VQA / "florence_victim" / exp / "florence2" / "vqa_florence2" / "rows_scored.csv",
        "visual": {
            "exp1": VQA / "florence_victim" / "exp1_mixed1000" / "visual_results",
            "exp2": VQA / "florence_victim" / "exp2_success1000" / "visual_results",
            "exp3": VQA / "florence_victim" / "exp3_success1000" / "visual_results",
        },
    },
}


def classify(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    victim = df["original_prediction"]
    df["atk_still"] = (df["label"] == 1) & df["fooled_before"] & ~df["blocked_full"] & (
        df["served_full"] != victim
    )
    df["clean_becomes_attack"] = (df["label"] == 0) & ~df["blocked_full"] & (
        df["served_full"] != victim
    )

    def cat(row):
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

    df["category"] = df.apply(cat, axis=1)
    return df


def load_rows(path: Path, exp: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"image_id": str}, keep_default_na=False)
    for c in FEATURES + ["rewrite_rate", "risk"]:
        df[c] = pd.to_numeric(df[c])
    df["label"] = df["label"].astype(int)
    df["fooled_before"] = df["fooled_before"].astype(bool)
    df["blocked_full"] = df["blocked_full"].astype(bool)
    df["exp"] = exp
    if "question" not in df.columns:
        df["question"] = ""
    return classify(df)


def pick_samples(sub: pd.DataFrame) -> pd.DataFrame:
    if sub.empty:
        return sub
    # one row per photo first, then fill up to MAX_SAMPLES
    sub = sub.sort_values(["photo_id", "image_id"])
    picked = []
    seen = set()
    for _, r in sub.iterrows():
        if r["photo_id"] in seen:
            continue
        seen.add(r["photo_id"])
        picked.append(r)
        if len(picked) >= MAX_SAMPLES:
            break
    if len(picked) < MIN_SAMPLES:
        for _, r in sub.iterrows():
            if any(p["image_id"] == r["image_id"] for p in picked):
                continue
            picked.append(r)
            if len(picked) >= MIN_SAMPLES or len(picked) >= len(sub):
                break
    return pd.DataFrame(picked)


def copy_images(row, vis_dir: Path, dest: Path) -> list[str]:
    dest.mkdir(parents=True, exist_ok=True)
    qid = row["image_id"]
    copied = []
    if int(row["label"]) == 1:
        names = [f"{qid}_1_ori.png", f"{qid}_2_atk.png"]
    else:
        names = [f"{qid}_1_ori.png"]
        atk = vis_dir / f"{qid}_2_atk.png"
        if atk.is_file():
            names.append(f"{qid}_2_atk.png")
    for name in names:
        src = vis_dir / name
        if src.is_file():
            shutil.copy2(src, dest / name)
            copied.append(name)
    return copied


def feature_means(sub: pd.DataFrame) -> dict:
    if sub.empty:
        return {}
    return {
        **{f: round(float(sub[f].mean()), 5) for f in FEATURES},
        "rewrite_rate": round(float(sub["rewrite_rate"].mean()), 3),
        "risk": round(float(sub["risk"].mean()), 3),
    }


def run_study(study_key: str, meta: dict) -> dict:
    study_dir = OUT / study_key
    study_dir.mkdir(parents=True, exist_ok=True)
    all_parts = []
    md_lines = [
        f"# {meta['title']}",
        "",
        "## 定義",
        "",
        "- **攻擊成功（原本）**：`fooled_before`（對抗圖 VQA ≠ 乾淨正確答案）",
        "- **二階後仍攻擊成功**：未封鎖且 `served_full` ≠ `original_prediction`",
        "- **二階後防禦成功**（攻擊行）：已 fooled 但未再成功（含封鎖或答回正確）",
        "- **乾淨圖變攻擊成功**：`label=0` 且未封鎖但 `served_full` ≠ `original_prediction`",
        "- 風險箱：`risk≤0.3` 低、`≥0.7` 高，其間為中（與 `analyze.py` 一致）",
        "",
    ]
    summary_rows = []

    for exp in ("exp1", "exp2", "exp3"):
        path = meta["rows"](exp)
        if not path.is_file():
            md_lines.append(f"## {exp}\n\n缺少 `{path}`，跳過。\n")
            continue
        df = load_rows(path, exp)
        fooled = df[(df["label"] == 1) & df["fooled_before"]]
        clean = df[df["label"] == 0]
        tagged = df[df["category"] != ""]
        all_parts.append(tagged)
        vis = meta["visual"][exp]

        md_lines.append(f"## {exp}")
        md_lines.append("")

        for cid in CAT:
            sub = df[df["category"] == cid]
            if sub.empty:
                md_lines.append(f"### {cid} — {CAT[cid]}")
                md_lines.append("")
                md_lines.append("*本實驗無此類樣本。*")
                md_lines.append("")
                continue

            feats = feature_means(sub)
            pct_f = (
                round(100 * len(sub) / len(fooled), 2)
                if cid in ("C1", "C2", "C3", "C4", "C5") and len(fooled)
                else None
            )
            pct_c = round(100 * len(sub) / len(clean), 2) if cid in ("C6", "C7") and len(clean) else None
            summary_rows.append(
                {
                    "study": study_key,
                    "exp": exp,
                    "category": cid,
                    "label_zh": CAT[cid],
                    "n_rows": len(sub),
                    "n_photos": sub["photo_id"].nunique(),
                    "pct_fooled_before": pct_f,
                    "pct_clean": pct_c,
                    **feats,
                }
            )

            sample_df = pick_samples(sub)
            bucket = study_dir / exp / cid
            bucket.mkdir(parents=True, exist_ok=True)
            sample_df.to_csv(bucket / "samples_meta.csv", index=False, encoding="utf-8-sig")

            md_lines.append(f"### {cid} — {CAT[cid]}")
            md_lines.append("")
            md_lines.append(
                f"- 行數 **{len(sub)}**，照片 **{sub['photo_id'].nunique()}**"
                + (f"，佔 fooled **{pct_f}%**" if pct_f is not None else "")
                + (f"，佔 clean **{pct_c}%**" if pct_c is not None else "")
            )
            md_lines.append(
                f"- 平均特徵：d_center={feats.get('d_center', '—')}, "
                f"rewrite={feats.get('rewrite_rate', '—')}, risk={feats.get('risk', '—')}"
            )
            md_lines.append("")
            md_lines.append("| qid | photo | bin | pred_ref | served | 圖片 |")
            md_lines.append("|-----|-------|-----|----------|--------|------|")

            for _, r in sample_df.iterrows():
                img_dir = bucket / r["image_id"]
                copied = copy_images(r, vis, img_dir)
                rel = "/".join([study_key, exp, cid, r["image_id"], copied[0]]) if copied else "—"
                q = str(r.get("question", ""))[:40].replace("|", " ")
                md_lines.append(
                    f"| {r['image_id']} | {r['photo_id']} | {r['bin']} | {r['pred_ref']} | "
                    f"{r['served_full']} | `{rel}` |"
                )
                if q:
                    md_lines.append(f"| | *Q:* {q} | | | | |")
            md_lines.append("")

    if all_parts:
        pd.concat(all_parts, ignore_index=True).to_csv(
            study_dir / "all_categorized_rows.csv", index=False, encoding="utf-8-sig"
        )
    counts = pd.DataFrame(summary_rows)
    counts.to_csv(study_dir / "counts_by_exp_category.csv", index=False, encoding="utf-8-sig")
    (study_dir / "REPORT.md").write_text("\n".join(md_lines), encoding="utf-8")
    return {"study": study_key, "n_summary_rows": len(summary_rows)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    index = [
        "# 防禦路由七類深入分析（BLIP + Florence）",
        "",
        "由 `build_report.py` 產生。每類別每實驗保留 **5–10** 個樣本（優先不同 `photo_id`），",
        "圖片在子目錄 `樣本_qid/` 下（`_*_ori.png` / `_*_atk.png`）。",
        "",
        "## 目錄",
        "",
        "- [BLIP_victim/REPORT.md](BLIP_victim/REPORT.md)",
        "- [Florence_victim/REPORT.md](Florence_victim/REPORT.md)",
        "",
        "## 類別代碼",
        "",
    ]
    for k, v in CAT.items():
        index.append(f"- **{k}**：{v}")
    index.append("")

    results = []
    for key, meta in STUDIES.items():
        results.append(run_study(key, meta))

    index.append("## 產生摘要")
    index.append("")
    index.append("```json")
    index.append(json.dumps(results, ensure_ascii=False, indent=2))
    index.append("```")
    (OUT / "README.md").write_text("\n".join(index), encoding="utf-8")
    print(f"Done -> {OUT}")


if __name__ == "__main__":
    main()
