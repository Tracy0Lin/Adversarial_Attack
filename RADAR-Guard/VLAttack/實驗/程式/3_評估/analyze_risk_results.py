"""Analyze stage-1 risk bins, clean low-vs-mid consistency, and teacher examples."""
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

EXPERIMENT_ROOT = Path(__file__).resolve().parent
FEATURE_COLS = ["d_center", "d_iqr", "d_tail", "d_q95"]
LOW_RISK = 0.3
HIGH_RISK = 0.7
PRIMARY_METHOD = "scale_crop_mild"


def answer_type(ans):
    ans = str(ans).lower().strip()
    if ans in {"yes", "no"}:
        return "Yes/No"
    number_words = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "none"}
    if ans.isdigit() or ans in number_words:
        return "Number"
    return "Other"


def load_log(log_path):
    with open(log_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for item in data:
        rows.append(
            {
                "image_id": str(item["question_id"]),
                "question": item.get("question_original", ""),
                "ground_truth": item.get("ground_truth", ""),
                "original_prediction": item.get("original_prediction", ""),
                "adversarial_prediction": item.get("adversarial_prediction", ""),
                "attack_success": bool(item.get("attack_success", False)),
                "attack_type": item.get("attack_type", ""),
                "ssim_ori_atk": item.get("ssim", {}).get("original_vs_attack", np.nan),
                "lpips_ori_atk": item.get("lpips", {}).get("original_vs_attack", np.nan),
                "answer_type": answer_type(item.get("ground_truth", "")),
            }
        )
    return pd.DataFrame(rows)


def mean_row(sub, prefix):
    out = {f"{prefix}_n": int(len(sub))}
    for col in FEATURE_COLS:
        out[f"{prefix}_{col}_mean"] = float(sub[col].mean()) if len(sub) else np.nan
        out[f"{prefix}_{col}_std"] = float(sub[col].std(ddof=1)) if len(sub) > 1 else np.nan
    return out


def pick_example(df, kind):
    if df.empty:
        return None
    if kind in {"correct_clean", "miss_attack"}:
        return df.sort_values("d_center").iloc[0]
    if kind in {"correct_attack", "false_alarm"}:
        return df.sort_values("d_center", ascending=False).iloc[0]
    if kind == "clean_mid":
        return df.sort_values("d_center", ascending=False).iloc[0]
    return df.iloc[0]


def write_example(out_root, visual_dir, slot, row, note):
    qid = str(row["image_id"])
    dest = out_root / f"{slot}_{qid}"
    dest.mkdir(parents=True, exist_ok=True)
    ori = visual_dir / f"{qid}_1_ori.png"
    atk = visual_dir / f"{qid}_2_atk.png"
    if ori.exists():
        shutil.copy2(ori, dest / "ori.png")
    if atk.exists():
        shutil.copy2(atk, dest / "atk.png")
    info = [
        f"slot: {slot}",
        f"question_id: {qid}",
        f"question: {row.get('question', '')}",
        f"ground_truth: {row.get('ground_truth', '')}",
        f"original_prediction: {row.get('original_prediction', '')}",
        f"adversarial_prediction: {row.get('adversarial_prediction', '')}",
        f"attack_success: {row.get('attack_success', '')}",
        f"attack_type: {row.get('attack_type', '')}",
        f"answer_type: {row.get('answer_type', '')}",
        f"method_type: {row.get('method_type', '')}",
        f"feat_split: {row.get('attack_type', '')}",
        f"risk_score: {row.get('risk_score', '')}",
        f"risk_bin: {row.get('risk_bin', '')}",
        f"d_center: {row.get('d_center', '')}",
        f"d_iqr: {row.get('d_iqr', '')}",
        f"d_tail: {row.get('d_tail', '')}",
        f"d_q95: {row.get('d_q95', '')}",
        f"ssim_ori_atk: {row.get('ssim_ori_atk', '')}",
        f"lpips_ori_atk: {row.get('lpips_ori_atk', '')}",
        f"why: {note}",
    ]
    (dest / "info.txt").write_text("\n".join(info), encoding="utf-8")
    return dest


def run_analysis(experiment_root=None, method=PRIMARY_METHOD, out_suffix=None):
    root = Path(experiment_root or EXPERIMENT_ROOT).resolve()
    csv_path = root / "tcvad_stage1_4_academic_dataset.csv"
    log_path = root / "attack_dual_defense_log.json"
    visual_dir = root / "visual_results"
    tag = "" if out_suffix is None else f"_{out_suffix}"
    out_dir = root / f"risk_tables{tag}"
    out_dir.mkdir(exist_ok=True)
    examples_dir = root / f"teacher_examples{tag}"
    if examples_dir.exists():
        shutil.rmtree(examples_dir)
    examples_dir.mkdir()

    df = pd.read_csv(csv_path)
    df["image_id"] = df["image_id"].astype(str)
    log = load_log(log_path).rename(columns={"attack_type": "log_attack_type"})
    merged = df.merge(log, on="image_id", how="left")

    primary = merged[merged["method_type"] == method].copy()
    primary["kind"] = np.where(primary["label"] == 0, "clean", "attack")
    bin_counts = primary.groupby(["kind", "risk_bin"]).size().unstack(fill_value=0)
    for col in ["low", "mid", "high"]:
        if col not in bin_counts.columns:
            bin_counts[col] = 0
    bin_counts = bin_counts[["low", "mid", "high"]]
    bin_counts.to_csv(out_dir / "risk_bin_summary.csv")
    print(f"\n=== 風險分級（主通道 {method}）===")
    print(bin_counts.to_string())

    clean = primary[primary["label"] == 0]
    attack = primary[primary["label"] == 1]
    clean_low = clean[clean["risk_bin"] == "low"]
    clean_mid = clean[clean["risk_bin"] == "mid"]
    clean_high = clean[clean["risk_bin"] == "high"]
    atk_low = attack[attack["risk_bin"] == "low"]
    atk_mid = attack[attack["risk_bin"] == "mid"]
    atk_high = attack[attack["risk_bin"] == "high"]

    cmp_rows = []
    low_stats = mean_row(clean_low, "low")
    mid_stats = mean_row(clean_mid, "mid")
    row = {"group": "clean_low_vs_mid"}
    row.update(low_stats)
    row.update(mid_stats)
    for col in FEATURE_COLS:
        row[f"delta_{col}_mid_minus_low"] = mid_stats[f"mid_{col}_mean"] - low_stats[f"low_{col}_mean"]
    cmp_rows.append(row)
    cmp_df = pd.DataFrame(cmp_rows)
    cmp_df.to_csv(out_dir / "clean_low_vs_mid_consistency.csv", index=False)

    print("\n=== 乾淨圖：低風險 vs 中風險一致性（d_center 越小越穩）===")
    print(f"clean low n={len(clean_low)}  mean d_center={low_stats['low_d_center_mean']:.4f}")
    print(f"clean mid n={len(clean_mid)}  mean d_center={mid_stats['mid_d_center_mean']:.4f}")
    print(f"Δ d_center (mid - low) = {row['delta_d_center_mid_minus_low']:.4f}")
    print("正值代表中風險乾淨圖比低風險乾淨圖更不穩，分級有特徵依據。")

    false_alarms = clean_high.sort_values("risk_score", ascending=False)
    misses = atk_low.sort_values("d_center")
    false_alarms.to_csv(out_dir / "clean_high_risk_false_alarms.csv", index=False)
    misses.to_csv(out_dir / "attack_low_risk_misses.csv", index=False)
    print(f"\n假警報（乾淨→高風險）: {len(false_alarms)}")
    print(f"漏抓（攻擊→低風險）: {len(misses)}")

    type_tab = (
        primary.groupby(["kind", "answer_type", "risk_bin"])
        .size()
        .unstack(fill_value=0)
    )
    type_tab.to_csv(out_dir / "risk_by_answer_type.csv")
    success_tab = (
        attack.groupby(["attack_success", "risk_bin"]).size().unstack(fill_value=0)
    )
    success_tab.to_csv(out_dir / "attack_success_vs_risk.csv")

    rotate = merged[merged["method_type"] == "rotate_5"][["image_id", "label", "risk_bin", "d_center"]].rename(
        columns={"risk_bin": "rotate_bin", "d_center": "rotate_d_center"}
    )
    scale = merged[merged["method_type"] == "scale_crop_mild"][["image_id", "label", "risk_bin", "d_center"]].rename(
        columns={"risk_bin": "scale_bin", "d_center": "scale_d_center"}
    )
    both = rotate.merge(scale, on=["image_id", "label"])
    both["same_bin"] = both["rotate_bin"] == both["scale_bin"]
    both.to_csv(out_dir / "channel_consistency.csv", index=False)
    print(f"\n兩通道同分級比例: {both['same_bin'].mean():.2%}")

    examples = []
    a = pick_example(clean_low, "correct_clean")
    if a is not None:
        examples.append(("A_correct_clean", a, "乾淨圖、低風險、d_center 低：一階正確放行"))
    b = pick_example(atk_high[atk_high["attack_success"] == True] if "attack_success" in atk_high else atk_high, "correct_attack")
    if b is None or (hasattr(b, "empty") and False):
        b = pick_example(atk_high, "correct_attack")
    if b is not None:
        examples.append(("B_correct_attack", b, "攻擊圖、高風險：一階正確攔截"))
    c = pick_example(false_alarms, "false_alarm")
    if c is not None:
        examples.append(("C_false_alarm_clean_high", c, "乾淨圖被判高風險：假警報"))
    miss_success = misses[misses["attack_success"] == True] if "attack_success" in misses else misses
    d = pick_example(miss_success if len(miss_success) else misses, "miss_attack")
    if d is not None:
        examples.append(("D_miss_attack_low", d, "攻擊圖被判低風險：漏抓"))
    e = pick_example(clean_mid, "clean_mid")
    if e is not None:
        examples.append(("E_clean_mid_consistency", e, "乾淨圖中風險：一致性比低風險差，應送二階"))
    failed_high = atk_high[atk_high["attack_success"] == False] if "attack_success" in atk_high else atk_high.iloc[0:0]
    f = pick_example(failed_high if len(failed_high) else atk_mid, "correct_attack")
    if f is not None:
        examples.append(("F_boundary", f, "邊界案例：攻擊失敗卻高風險，或中風險攻擊"))

    catalog = []
    for slot, row_ex, note in examples:
        dest = write_example(examples_dir, visual_dir, slot, row_ex, note)
        catalog.append({"slot": slot, "image_id": row_ex["image_id"], "path": str(dest), "note": note})
    pd.DataFrame(catalog).to_csv(out_dir / "teacher_examples_index.csv", index=False)
    print(f"\n代表圖寫入 {examples_dir}，共 {len(catalog)} 組")
    print("Analysis done.")
    return primary


if __name__ == "__main__":
    run_analysis()
