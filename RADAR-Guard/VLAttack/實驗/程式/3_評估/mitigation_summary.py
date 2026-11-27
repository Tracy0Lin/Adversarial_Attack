"""Mitigation ASR summary: does Gaussian denoise (theta*) lower attack success?

Uses sss_per_theta_raw.csv (Mid, per-sigma VQA preds) + attack_dual_defense_log.json
+ tcvad_stage1_4_academic_dataset.csv (routing). No GPU.

Attack success (aligned with attack log):
    success  <=>  VQA(adv) != original_prediction  (ori image answer)

After denoise on Mid path @ theta*:
    success_after  <=>  VQA(blur(adv)) != original_prediction
"""
import argparse
import json
from pathlib import Path

import pandas as pd

THIS_FILE = Path(__file__).resolve()
EXPERIMENT_ROOT = THIS_FILE.parent


def load_attack_log(log_path):
    with open(log_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for item in data:
        qid = str(item["question_id"])
        ori = item.get("original_prediction", "")
        rows.append(
            {
                "image_id": qid,
                "original_prediction": ori,
                "attack_success_before": bool(item.get("attack_success", False)),
            }
        )
    return pd.DataFrame(rows)


def attack_success_series(pred, ori_pred):
    return pred.astype(str) != ori_pred.astype(str)


def summarize_cohort(name, before, after, n):
    b = float(before.mean()) if n else float("nan")
    a = float(after.mean()) if n else float("nan")
    rescued = before & ~after
    return {
        "cohort": name,
        "n": int(n),
        "asr_before": b,
        "asr_after": a,
        "asr_drop_pp": (b - a) * 100 if n else float("nan"),
        "rescued_count": int(rescued.sum()) if n else 0,
        "rescued_rate": float(rescued.mean()) if n else float("nan"),
    }


def run(experiment_root=None, sigma=None):
    root = Path(experiment_root or EXPERIMENT_ROOT).resolve()
    raw_path = root / "sss_per_theta_raw.csv"
    log_path = root / "attack_dual_defense_log.json"
    stage1_path = root / "tcvad_stage1_4_academic_dataset.csv"
    theta_path = root / "sss_best_theta.json"

    missing = [p.name for p in (raw_path, log_path, stage1_path) if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing under {root}: {', '.join(missing)}. "
            "Run stage1 + SSS first (--skip_stage1 if stage1 CSV exists)."
        )

    if theta_path.is_file():
        with open(theta_path, "r", encoding="utf-8") as f:
            theta_star = json.load(f)
        sigma = float(sigma if sigma is not None else theta_star["sigma"])
    elif sigma is None:
        raise FileNotFoundError(f"No sss_best_theta.json and --sigma not set under {root}")
    else:
        sigma = float(sigma)

    raw = pd.read_csv(raw_path)
    raw["image_id"] = raw["image_id"].astype(str)
    sub = raw[raw["sigma"] == sigma].copy()
    if sub.empty:
        raise ValueError(f"No rows for sigma={sigma} in {raw_path}")

    log = load_attack_log(log_path)
    stage1 = pd.read_csv(stage1_path)
    stage1["image_id"] = stage1["image_id"].astype(str)
    method = sub["method_type"].iloc[0]
    stage1 = stage1[stage1["method_type"] == method].copy()
    exp_attack_ids = stage1.loc[stage1["label"] == 1, "image_id"].unique()
    log = log[log["image_id"].isin(exp_attack_ids)].copy()

    mid_atk = sub[(sub["label"] == 1) & (sub["kind"] == "attack")].copy()
    mid_clean = sub[(sub["label"] == 0) & (sub["kind"] == "clean")].copy()

    ori_by_qid = log.set_index("image_id")["original_prediction"]
    mid_atk["ori_pred"] = mid_atk["image_id"].map(ori_by_qid)
    mid_clean["ori_pred"] = mid_clean["image_id"].map(ori_by_qid)

    mid_atk["success_before"] = attack_success_series(mid_atk["pred_ref"], mid_atk["ori_pred"])
    mid_atk["success_after"] = attack_success_series(mid_atk["pred_denoised"], mid_atk["ori_pred"])
    mid_atk["fooled_before"] = attack_success_series(mid_atk["pred_ref"], mid_atk["correct_ans"])

    fooled = mid_atk[mid_atk["fooled_before"]]
    rows = [
        summarize_cohort(
            f"mid_attack_all_sigma_{sigma}",
            mid_atk["success_before"],
            mid_atk["success_after"],
            len(mid_atk),
        ),
        summarize_cohort(
            f"mid_attack_fooled_before_sigma_{sigma}",
            fooled["success_before"],
            fooled["success_after"],
            len(fooled),
        ),
    ]

    clean_harm = (mid_clean["pred_denoised"].astype(str) != mid_clean["pred_ref"].astype(str)).mean()
    rows.append(
        {
            "cohort": f"mid_clean_answer_changed_sigma_{sigma}",
            "n": int(len(mid_clean)),
            "asr_before": float("nan"),
            "asr_after": float("nan"),
            "asr_drop_pp": float("nan"),
            "rescued_count": int((mid_clean["pred_denoised"].astype(str) != mid_clean["pred_ref"].astype(str)).sum()),
            "rescued_rate": float(clean_harm),
        }
    )

    # --- 1000 attacks: Mid denoised @ theta*, others unchanged (mitigation-only routing) ---
    atk_log = log.copy()
    denoised = mid_atk.set_index("image_id")["pred_denoised"]
    ori = atk_log["original_prediction"]
    before = atk_log["attack_success_before"]
    after_pred = denoised.reindex(atk_log["image_id"]).fillna(
        mid_atk.set_index("image_id")["pred_ref"].reindex(atk_log["image_id"])
    )
    after_pred = after_pred.reset_index(drop=True)
    ori = ori.reset_index(drop=True)
    before = before.reset_index(drop=True)
    # Non-mid attacks: no row in raw — keep log before state for after
    in_mid = atk_log["image_id"].isin(mid_atk["image_id"]).reset_index(drop=True)
    after_success = attack_success_series(after_pred, ori)
    after_success = after_success.where(in_mid, before)

    rows.append(
        summarize_cohort(
            f"all_1000_attack_mid_denoise_only_sigma_{sigma}",
            before,
            after_success,
            len(atk_log),
        )
    )

    success_only = atk_log[before]
    if len(success_only):
        so_after = after_success.loc[success_only.index]
        n_ok = len(success_only)
        rows.append(
            summarize_cohort(
                f"attack_success_log_n{n_ok}_mid_denoise_only_sigma_{sigma}",
                success_only["attack_success_before"],
                so_after,
                n_ok,
            )
        )

    # --- Routed: Low pass (unchanged), Mid denoise, High block (attack thwarted) ---
    s1_atk = stage1[stage1["label"] == 1].merge(
        atk_log, on="image_id", how="inner", suffixes=("_s1", "")
    )
    routed_after = []
    for r in s1_atk.itertuples(index=False):
        qid = r.image_id
        b = bool(r.attack_success_before)
        if r.risk_bin == "low":
            routed_after.append(b)
        elif r.risk_bin == "high":
            routed_after.append(False)
        else:
            if qid in denoised.index:
                p = denoised.loc[qid]
                routed_after.append(str(p) != str(r.original_prediction))
            else:
                routed_after.append(b)
    s1_atk = s1_atk.copy()
    s1_atk["success_after_routed"] = routed_after
    rows.append(
        summarize_cohort(
            f"routed_low_pass_mid_denoise_high_block_sigma_{sigma}",
            s1_atk["attack_success_before"],
            s1_atk["success_after_routed"],
            len(s1_atk),
        )
    )

    for bin_name in ("low", "mid", "high"):
        g = s1_atk[s1_atk["risk_bin"] == bin_name]
        if g.empty:
            continue
        rows.append(
            summarize_cohort(
                f"routed_attack_{bin_name}_sigma_{sigma}",
                g["attack_success_before"],
                g["success_after_routed"],
                len(g),
            )
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(root / "mitigation_summary.csv", index=False, encoding="utf-8")

    lines = [
        f"Mitigation summary @ sigma={sigma} (theta* from sss_best_theta unless --sigma)",
        f"Method: {method}",
        "",
        summary.to_string(index=False),
    ]
    txt = "\n".join(lines)
    (root / "mitigation_summary.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)
    print(f"\nSaved {root / 'mitigation_summary.csv'}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mitigation ASR summary at theta*")
    parser.add_argument("--experiment_root", default=str(EXPERIMENT_ROOT))
    parser.add_argument("--sigma", type=float, default=None, help="override theta* sigma")
    args = parser.parse_args()
    run(experiment_root=args.experiment_root, sigma=args.sigma)
