"""Export metrics for 成果報告/2026-10-20_三實驗_完整指標報告.md."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent / "VQA" / "multi_captioner"
OUT = ROOT / "three_exp_report_metrics.json"


def metrics_block(y_true, y_pred):
    y_true = np.asarray(y_true, bool)
    y_pred = np.asarray(y_pred, bool)
    tp = int((y_pred & y_true).sum())
    fp = int((y_pred & ~y_true).sum())
    fn = int((~y_pred & y_true).sum())
    tn = int((~y_pred & ~y_true).sum())
    acc = (tp + tn) / (tp + tn + fp + fn)
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else float("nan")
    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1": round(f1, 4),
    }


def add_vc(df):
    if (df.label == 0).any():
        clean = df[df.label == 0][["image_id", "pred_ref"]].rename(columns={"pred_ref": "vc"})
        df = df.merge(clean, on="image_id", how="left")
        df.loc[df.label == 0, "vc"] = df.loc[df.label == 0, "pred_ref"]
    else:
        df = df.copy()
        df["vc"] = df["original_prediction"]
    return df


def bin_counts(df):
    out = {}
    for lab, name in [(1, "attack"), (0, "clean")]:
        sub = df[df.label == lab]
        vc = sub["bin"].value_counts()
        out[name] = {b: int(vc.get(b, 0)) for b in ("low", "mid", "high")}
    return out


def attack_success_by_bin(atk):
    out = {}
    for b in ("low", "mid", "high"):
        sub = atk[atk.bin == b]
        out[b] = int((sub["pred_ref"].astype(str) != sub["vc"].astype(str)).sum())
    return out


def stage2_mid_only(df, theta_star, theta_det):
    """Stage2 Gaussian applies only to Mid; full A3 also blocks Mid when θ_det flips answer."""
    mid = df[df.bin == "mid"].copy()
    cs, cd = f"pred_{theta_star}", f"pred_{theta_det}"
    flip = mid[cd].astype(str) != mid["pred_ref"].astype(str)
    atk = mid[mid.label == 1]
    cln = mid[mid.label == 0]
    fooled_before = atk["pred_ref"].astype(str) != atk["vc"].astype(str)
    fooled_after_blur = atk[cs].astype(str) != atk["vc"].astype(str)
    bf = cln["blocked_full"].astype(str).str.lower() == "true" if "blocked_full" in cln.columns else flip[cln.index]
    return {
        "mid_attack_n": len(atk),
        "mid_clean_n": len(cln),
        "attack_success_before_gaussian": int(fooled_before.sum()),
        "attack_success_after_gaussian_theta_star_only": int(fooled_after_blur.sum()),
        "attack_flip_flag_theta_det_in_mid": int(flip[atk.index].sum()),
        "attack_rescued_by_blur_only": int((fooled_before & ~fooled_after_blur).sum()),
        "clean_blocked_e2e_in_mid": int((cln["blocked_full"].astype(str).str.lower() == "true").sum()) if len(cln) else 0,
        "clean_pass_e2e_in_mid": int(len(cln) - (cln["blocked_full"].astype(str).str.lower() == "true").sum()) if len(cln) else 0,
        "clean_theta_det_flip_in_mid": int(flip[cln.index].sum()) if len(cln) else 0,
        "clean_theta_star_answer_changed_in_mid": int((cln[cs].astype(str) != cln["pred_ref"].astype(str)).sum()) if len(cln) else 0,
    }


def attack_e2e_by_bin(atk):
    bf = atk["blocked_full"].astype(str).str.lower() == "true"
    fooled = atk["served_full"].astype(str) != atk["vc"].astype(str)
    e2e = ~bf & fooled
    out = {}
    for b in ("low", "mid", "high"):
        m = atk.bin == b
        out[b] = {
            "blocked": int((bf & m).sum()),
            "attack_still_succeeds_e2e": int((e2e & m).sum()),
        }
    out["total_still_succeeds_e2e"] = int(e2e.sum())
    return out


def clean_routing_stage1(cln):
    n = len(cln)
    return {
        "direct_pass_low": int((cln.bin == "low").sum()),
        "into_repair_mid": int((cln.bin == "mid").sum()),
        "false_intercept_high": int((cln.bin == "high").sum()),
        "direct_pass_pct": round((cln.bin == "low").sum() / n * 100, 2) if n else None,
        "into_repair_pct": round((cln.bin == "mid").sum() / n * 100, 2) if n else None,
        "false_intercept_pct": round((cln.bin == "high").sum() / n * 100, 2) if n else None,
    }


def clean_e2e_by_bin(cln):
    """Clean full pipeline by Stage 1 bin. Answer change is served ≠ pred_ref (this report's vc)."""
    bf = cln["blocked_full"].astype(str).str.lower() == "true"
    chg = cln["served_full"].astype(str) != cln["pred_ref"].astype(str)
    out = {}
    for b in ("low", "mid", "high"):
        m = cln.bin == b
        out[b] = {
            "n": int(m.sum()),
            "blocked": int((bf & m).sum()),
            "served": int((~bf & m).sum()),
            "served_answer_changed": int((chg & ~bf & m).sum()),
        }
    n = len(cln)
    out["total_blocked"] = int(bf.sum())
    out["total_served"] = int((~bf).sum())
    out["total_served_answer_changed"] = int((chg & ~bf).sum())
    out["blocked_pct"] = round(bf.sum() / n * 100, 2) if n else None
    out["service_pct"] = round((~bf).sum() / n * 100, 2) if n else None
    return out


def clean_routing_e2e(cln):
    bf = cln["blocked_full"].astype(str).str.lower() == "true"
    n = len(cln)
    chg = (cln["served_full"].astype(str) != cln["pred_ref"].astype(str)) & ~bf
    return {
        "blocked_any_e2e": int(bf.sum()),
        "blocked_pct": round(bf.sum() / n * 100, 2) if n else None,
        "served": int((~bf).sum()),
        "service_pct": round((~bf).sum() / n * 100, 2) if n else None,
        "served_unchanged_low": int(((cln.bin == "low") & ~bf).sum()),
        "served_answer_changed": int(chg.sum()),
        "served_answer_changed_pct": round(chg.sum() / n * 100, 2) if n else None,
    }


def process(exp):
    s = json.loads((ROOT / exp / "blip" / "summary.json").read_text(encoding="utf-8"))
    df = pd.read_csv(ROOT / exp / "blip" / "rows_scored.csv", dtype=str, keep_default_na=False)
    df["label"] = df["label"].astype(int)
    df = add_vc(df)
    atk = df[df.label == 1]
    cln = df[df.label == 0]
    asr0 = (atk["pred_ref"].astype(str) != atk["vc"].astype(str)).mean() * 100
    y = (df.label == 1).values
    b = df.bin.values
    risk = pd.to_numeric(df["risk"])
    m_high = metrics_block(y, b == "high")
    m_mh = metrics_block(y, (b == "mid") | (b == "high"))
    if len(cln):
        auc = round(float(roc_auc_score(y, risk)), 4)
        m_high["AUC"] = auc
        m_mh["AUC"] = auc
    else:
        m_high["AUC"] = None
        m_mh["AUC"] = None
    ts, td = s["theta_star_mid"], s["theta_det_mid"]
    s2 = stage2_mid_only(df, ts, td)
    pre_bin = attack_success_by_bin(atk)
    e2e_bin = attack_e2e_by_bin(atk)
    fp = s["full_pipeline"]
    return {
        "experiment": exp,
        "n_attack": len(atk),
        "n_clean": len(cln),
        "initial_attack_rate_pct": round(asr0, 2),
        "stage1_bins": bin_counts(df),
        "stage1_metrics_high_block": m_high,
        "stage1_metrics_mid_or_high_flag": m_mh,
        "attack_success_before_gaussian_by_risk": pre_bin,
        "stage2_theta_star": ts,
        "stage2_theta_det": td,
        "stage2_mid_only": s2,
        "attack_e2e_by_risk_bin": e2e_bin,
        "clean_stage1_routing": clean_routing_stage1(cln) if len(cln) else None,
        "clean_e2e": clean_routing_e2e(cln) if len(cln) else None,
        "clean_e2e_by_risk_bin": clean_e2e_by_bin(cln) if len(cln) else None,
        "full_pipeline": {
            "attacks_fooled_before": fp["attacks_fooled_before"],
            "attacks_still_fooling_e2e": fp["attacks_still_fooling"],
            "effective_asr_pct": round(fp["effective_asr"] * 100, 2),
            "clean_blocked_e2e": int(fp["FPR"] * len(cln)) if len(cln) else None,
            "clean_blocked_rate_pct": round(fp["clean_blocked_rate"] * 100, 2) if len(cln) else None,
        },
    }


def main():
    data = {e: process(e) for e in ("exp1", "exp2", "exp3")}
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
