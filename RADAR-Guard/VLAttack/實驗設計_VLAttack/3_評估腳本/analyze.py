"""Score stage 1, route, apply stage 2, and compute every metric for one (exp, captioner).

No GPU work: consumes caption_features.csv (captioner-dependent) and
probe_gaussian.csv (captioner-independent) plus the attack log.

Decision rule (full pipeline):
    Low  -> pass, answer = VQA(x)
    High -> block
    Mid  -> block if VQA(blur_theta_det(x)) != VQA(x), else answer = VQA(blur_theta*(x))

Florence judges both attack and clean answers against ground_truth (the screened
max-weight VQAv2 answer). BLIP still uses the clean-image answer
(original_prediction).
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import HIGH_RISK, LOW_RISK, SIGMAS, exp_dir, experiment_rows

FEATURES = ["d_center", "d_iqr", "d_tail", "d_q95"]
N_FOLDS = 5
N_BOOT = 1000
EPS = 1e-6


def geo_mean(*arrays):
    stacked = np.clip(np.vstack([np.asarray(a, dtype=float) for a in arrays]), EPS, 1.0)
    return np.exp(np.log(stacked).mean(axis=0))


def svm():
    return Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel="rbf", C=1.0, class_weight="balanced"))])


def to_risk(margin):
    return 1.0 / (1.0 + np.exp(-np.clip(margin, -50, 50)))


def risk_bin(r):
    return np.where(r <= LOW_RISK, "low", np.where(r >= HIGH_RISK, "high", "mid"))


# ---------------------------------------------------------------- stage 1

def oof_scores(df):
    """5-fold GroupKFold by photo: no photo appears in both train and test."""
    X, y, g = df[FEATURES].values, df["label"].values, df["photo_id"].values
    risk = np.zeros(len(df))
    lr_score = np.zeros(len(df))
    for tr, te in GroupKFold(n_splits=N_FOLDS).split(X, y, g):
        m = svm().fit(X[tr], y[tr])
        risk[te] = to_risk(m.decision_function(X[te]))
        lr = Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(class_weight="balanced", max_iter=1000))])
        lr.fit(X[tr][:, :1], y[tr])
        lr_score[te] = lr.predict_proba(X[te][:, :1])[:, 1]
    return risk, lr_score


def load_features(exp, captioner):
    df = pd.read_csv(exp_dir(exp, captioner) / "caption_features.csv", dtype={"image_id": str},
                     keep_default_na=False)
    for c in FEATURES + ["rewrite_rate"] + [f"mss_{s}" for s in SIGMAS]:
        df[c] = pd.to_numeric(df[c])
    return df


def stage1(exp, captioner):
    df = load_features(exp, captioner)
    if exp == "exp3":
        train = load_features("exp2", captioner)
        model = svm().fit(train[FEATURES].values, train["label"].values)
        df["risk"] = to_risk(model.decision_function(df[FEATURES].values))
        df["dcenter_lr"] = np.nan
        df["stage1_protocol"] = "trained on exp2 (2000 rows), applied to exp3"
    else:
        df["risk"], df["dcenter_lr"] = oof_scores(df)
        df["stage1_protocol"] = f"{N_FOLDS}-fold GroupKFold by photo, out-of-fold"
    df["bin"] = risk_bin(df["risk"].values)
    return df


# ---------------------------------------------------------------- stage 2 inputs

def probe_csv(exp, vqa_backend="blip"):
    name = "probe_gaussian.csv" if vqa_backend == "blip" else f"probe_gaussian_{vqa_backend}.csv"
    return exp_dir(exp) / name


def attach_probe(df, exp, vqa_backend="blip"):
    probe = pd.read_csv(probe_csv(exp, vqa_backend), dtype=str, keep_default_na=False)
    probe["label"] = probe["label"].astype(int)
    probe["sigma"] = probe["sigma"].astype(float)
    for c in ["iqp", "clip_sim"]:
        probe[c] = pd.to_numeric(probe[c], errors="coerce")
    ref = probe.drop_duplicates(["image_id", "label"])[["image_id", "label", "pred_ref"]]
    df = df.merge(ref, on=["image_id", "label"], how="left")
    for s in SIGMAS:
        p = probe[probe["sigma"] == s][["image_id", "label", "pred_denoised", "iqp", "clip_sim"]]
        p = p.rename(columns={"pred_denoised": f"pred_{s}", "iqp": f"iqp_{s}", "clip_sim": f"clip_{s}"})
        df = df.merge(p, on=["image_id", "label"], how="left")
        df[f"sp_{s}"] = geo_mean(df[f"clip_{s}"].clip(0, 1), df[f"mss_{s}"])
    meta_cols = ["image_id", "label", "question", "original_prediction", "log_attack_success"]
    meta = pd.DataFrame(experiment_rows(exp))
    if "ground_truth" in meta.columns:
        meta_cols.append("ground_truth")
    df = df.merge(meta[meta_cols], on=["image_id", "label"], how="left")
    if df["pred_ref"].isna().any():
        raise RuntimeError("probe rows missing")
    df["log_attack_success"] = df["log_attack_success"].astype(bool)
    clean_ref = ref[ref["label"] == 0][["image_id", "pred_ref"]].rename(columns={"pred_ref": "victim_clean_pred"})
    df = df.merge(clean_ref, on="image_id", how="left")
    if vqa_backend == "florence2":
        if "ground_truth" not in df.columns:
            df["ground_truth"] = ""
        gt = df["ground_truth"].fillna("").astype(str).str.strip()
        ori = df["original_prediction"].fillna("").astype(str)
        use_gt = gt.to_numpy() != ""
        df["victim_answer"] = np.where(use_gt, gt.to_numpy(), ori.to_numpy())
        df["fooled_before"] = (df["label"] == 1) & (df["pred_ref"].astype(str) != df["victim_answer"].astype(str))
    elif vqa_backend == "blip":
        df["victim_answer"] = df["original_prediction"]
        df["fooled_before"] = (df["label"] == 1) & (df["pred_ref"] != df["victim_answer"])
    else:
        df["victim_answer"] = df["victim_clean_pred"]
        df["fooled_before"] = (df["label"] == 1) & (df["pred_ref"] != df["victim_clean_pred"])
    return df


def pick_theta_star(sub):
    """SSS = geo-mean(delta_ASR, mean IQP, mean SP) over a pool, as in sss_denoise.py."""
    rows = []
    fooled = sub[(sub["label"] == 1) & sub["fooled_before"]]
    for s in SIGMAS:
        ref_col = "victim_answer" if "victim_answer" in fooled.columns else "original_prediction"
        d_asr = 1.0 - float((fooled[f"pred_{s}"] != fooled[ref_col]).mean()) if len(fooled) else np.nan
        iqp, sp = float(sub[f"iqp_{s}"].mean()), float(sub[f"sp_{s}"].mean())
        clean = sub[sub["label"] == 0]
        rows.append({
            "sigma": s, "n_pool": len(sub), "n_attack_fooled": len(fooled), "delta_asr": d_asr,
            "iqp": iqp, "sp": sp,
            "sss": float(geo_mean([max(d_asr, 0.0)], [iqp], [sp])[0]) if not np.isnan(d_asr) else np.nan,
            "clean_answer_changed": float((clean[f"pred_{s}"] != clean["pred_ref"]).mean()) if len(clean) else np.nan,
        })
    grid = pd.DataFrame(rows)
    if grid.empty or grid["sss"].isna().all():
        return 0.8, grid
    best = grid["sss"].idxmax(skipna=True)
    return float(grid.loc[best, "sigma"]), grid


def pick_theta_det(sub):
    rows = []
    y = sub["label"].values == 1
    for s in SIGMAS:
        flip = (sub[f"pred_{s}"] != sub["pred_ref"]).values
        tpr = flip[y].mean() if y.any() else np.nan
        fpr = flip[~y].mean() if (~y).any() else np.nan
        fooled = y & sub["fooled_before"].values
        rows.append({"sigma": s, "TPR": tpr, "FPR": fpr, "youden_j": tpr - fpr,
                     "TPR_fooled_only": flip[fooled].mean() if fooled.any() else np.nan})
    grid = pd.DataFrame(rows)
    if grid.empty or grid["youden_j"].isna().all():
        return 1.5, grid
    best = grid["youden_j"].idxmax(skipna=True)
    return float(grid.loc[best, "sigma"]), grid


# ---------------------------------------------------------------- decision + metrics

def decide(df, mode, theta_star, theta_det):
    """Return (blocked, served_answer, vqa_calls) arrays for one pipeline variant."""
    n = len(df)
    ref = df["pred_ref"].values
    flip_det = (df[f"pred_{theta_det}"] != df["pred_ref"]).values if theta_det else np.zeros(n, bool)
    mit = df[f"pred_{theta_star}"].values if theta_star else ref
    b = df["bin"].values
    low, mid, high = b == "low", b == "mid", b == "high"
    blocked = np.zeros(n, bool)
    served = ref.copy()
    calls = np.ones(n)
    extra = 2.0 if theta_star != theta_det else 1.0
    if mode == "none":
        pass
    elif mode == "s1_high":
        blocked = high
        calls = np.where(high, 0, 1)
    elif mode == "s1_midhigh":
        blocked = mid | high
        calls = np.where(blocked, 0, 1)
    elif mode == "s2_only":
        blocked = flip_det
        served = np.where(blocked, ref, mit)
        calls = np.where(blocked, 2, 1 + extra)
    elif mode == "s2_mitigate_only":
        served = mit
        calls = np.full(n, 1.0)
    elif mode == "full":
        blocked = high | (mid & flip_det)
        served = np.where(mid & ~blocked, mit, ref)
        calls = np.where(high, 0, np.where(mid, np.where(flip_det, 2, 1 + extra), 1))
    elif mode == "full_mitigate_only":
        blocked = high
        served = np.where(mid, mit, ref)
        calls = np.where(high, 0, 1)
    else:
        raise ValueError(mode)
    return blocked, served, calls


def detection_metrics(y, flagged):
    y, f = np.asarray(y, bool), np.asarray(flagged, bool)
    tp, fp = int((f & y).sum()), int((f & ~y).sum())
    fn, tn = int((~f & y).sum()), int((~f & ~y).sum())
    tpr = tp / (tp + fn) if tp + fn else np.nan
    fpr = fp / (fp + tn) if fp + tn else np.nan
    prec = tp / (tp + fp) if tp + fp else np.nan
    f1 = 2 * prec * tpr / (prec + tpr) if tp else (0.0 if tp + fn else np.nan)
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "TPR": tpr, "FPR": fpr,
            "youden_j": tpr - fpr if not np.isnan(fpr) else np.nan,
            "precision": prec, "F1": f1,
            "balanced_acc": (tpr + 1 - fpr) / 2 if not np.isnan(fpr) else np.nan}


def outcome_metrics(df, blocked, served, calls):
    y = df["label"].values == 1
    ori = df["victim_answer"].values if "victim_answer" in df.columns else df["original_prediction"].values
    atk_success_after = y & ~blocked & (served != ori)
    clean_ok = ~y & ~blocked & (served == ori)
    out = detection_metrics(y, blocked)
    out.update({
        "asr_before": float(df.loc[y, "fooled_before"].mean()) if y.any() else np.nan,
        "effective_asr": float(atk_success_after[y].mean()) if y.any() else np.nan,
        "attacks_fooled_before": int(df.loc[y, "fooled_before"].sum()),
        "attacks_still_fooling": int(atk_success_after.sum()),
        "clean_service_rate": float(clean_ok[~y].mean()) if (~y).any() else np.nan,
        "clean_blocked_rate": float(blocked[~y].mean()) if (~y).any() else np.nan,
        "clean_answer_changed_rate": float((~blocked[~y] & (served[~y] != ori[~y])).mean()) if (~y).any() else np.nan,
        "vqa_calls_per_row": float(np.mean(calls)),
    })
    return out


def bootstrap(df, blocked, served, calls, keys=("TPR", "FPR", "youden_j", "effective_asr", "clean_service_rate"),
              n=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    photos = df["photo_id"].values
    uniq = np.unique(photos)
    idx_by_photo = {p: np.where(photos == p)[0] for p in uniq}
    samples = {k: [] for k in keys}
    for _ in range(n):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_photo[p] for p in pick])
        m = outcome_metrics(df.iloc[idx], blocked[idx], served[idx], calls[idx])
        for k in keys:
            samples[k].append(m[k])
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in samples.items()}


def score_metrics(y, score):
    y = np.asarray(y, bool)
    if y.all() or (~y).all():
        return {}
    fpr, tpr, _ = roc_curve(y, score)
    return {
        "auroc": float(roc_auc_score(y, score)),
        "auprc": float(average_precision_score(y, score)),
        "tpr_at_fpr_5": float(tpr[fpr <= 0.05].max()),
        "tpr_at_fpr_1": float(tpr[fpr <= 0.01].max()),
    }


# ---------------------------------------------------------------- main

VARIANTS = [
    ("A0_no_defense", "none"),
    ("A1_stage1_high_block", "s1_high"),
    ("A1b_stage1_mid_high_block", "s1_midhigh"),
    ("A2_stage2_only", "s2_only"),
    ("A2b_stage2_mitigate_only", "s2_mitigate_only"),
    ("A3_full_mitigate_only", "full_mitigate_only"),
    ("A3_full_pipeline", "full"),
]


def run(exp, captioner, vqa_backend="blip", output_tag=None, fix_theta_star=None, fix_theta_det=None):
    out_dir = exp_dir(exp, captioner)
    if vqa_backend != "blip":
        suffix = f"vqa_{vqa_backend}"
        if output_tag:
            suffix = f"{suffix}_{output_tag}"
        out_dir = exp_dir(exp, captioner) / suffix
        out_dir.mkdir(parents=True, exist_ok=True)
    elif output_tag:
        out_dir = exp_dir(exp, captioner) / output_tag
        out_dir.mkdir(parents=True, exist_ok=True)
    df = attach_probe(stage1(exp, captioner), exp, vqa_backend=vqa_backend)
    y = df["label"].values == 1
    has_clean = (~y).any()
    mid = df[df["bin"] == "mid"]

    theta_star, sss_grid = pick_theta_star(mid)
    theta_det, det_grid = pick_theta_det(mid)
    theta_star_all, _ = pick_theta_star(df)
    theta_det_all, _ = pick_theta_det(df)
    theta_source = "selected on this experiment"
    if exp == "exp3" and fix_theta_star is None and fix_theta_det is None:
        e2 = exp_dir("exp2", captioner)
        if vqa_backend != "blip":
            e2 = e2 / f"vqa_{vqa_backend}"
        ref = json.loads((e2 / "summary.json").read_text(encoding="utf-8"))
        theta_star, theta_det = ref["theta_star_mid"], ref["theta_det_mid"]
        theta_star_all, theta_det_all = ref["theta_star_all"], ref["theta_det_all"]
        sss_grid = pd.read_csv(e2 / "sss_grid_mid.csv")
        det_grid = pd.read_csv(e2 / "theta_det_grid_mid.csv")
        theta_source = "exp2 (same captioner)"
    if fix_theta_star is not None:
        theta_star = fix_theta_star
        theta_star_all = fix_theta_star
    if fix_theta_det is not None:
        theta_det = fix_theta_det
        theta_det_all = fix_theta_det
    if fix_theta_star is not None or fix_theta_det is not None:
        theta_source = (
            f"fixed override theta_star={fix_theta_star} theta_det={fix_theta_det}"
        )
    sss_grid.to_csv(out_dir / "sss_grid_mid.csv", index=False)
    det_grid.to_csv(out_dir / "theta_det_grid_mid.csv", index=False)

    variants = []
    for name, mode in VARIANTS:
        ts, td = (theta_star_all, theta_det_all) if mode.startswith("s2") else (theta_star, theta_det)
        blocked, served, calls = decide(df, mode, ts, td)
        m = outcome_metrics(df, blocked, served, calls)
        m.update({"variant": name, "theta_star": ts if mode not in ("none", "s1_high", "s1_midhigh") else None,
                  "theta_det": td if mode in ("s2_only", "full") else None})
        if name in ("A3_full_pipeline", "A1_stage1_high_block", "A2_stage2_only"):
            ci = bootstrap(df, blocked, served, calls)
            m.update({f"{k}_ci95": v for k, v in ci.items()})
        variants.append(m)
        if mode == "full":
            df["blocked_full"] = blocked
            df["served_full"] = served
    var_df = pd.DataFrame(variants)
    var_df.to_csv(out_dir / "pipeline_variants.csv", index=False)

    bins = df.groupby(["label", "bin"]).size().unstack(fill_value=0).reindex(columns=["low", "mid", "high"],
                                                                            fill_value=0)
    full = var_df.set_index("variant").loc["A3_full_pipeline"].to_dict()
    summary = {
        "experiment": exp,
        "captioner": captioner,
        "vqa_backend": vqa_backend,
        "n_rows": int(len(df)),
        "n_attack": int(y.sum()),
        "n_clean": int((~y).sum()),
        "n_photos": int(df["photo_id"].nunique()),
        "stage1_protocol": df["stage1_protocol"].iloc[0],
        "stage1_risk": score_metrics(y, df["risk"].values) if has_clean else {},
        "stage1_dcenter_lr": score_metrics(y, df["dcenter_lr"].values) if has_clean else {},
        "stage1_high_only": detection_metrics(y, df["bin"].values == "high"),
        "stage1_midhigh": detection_metrics(y, df["bin"].values != "low"),
        "mid_rate": float((df["bin"] == "mid").mean()),
        "bins": {("attack" if k == 1 else "clean"): v for k, v in bins.to_dict(orient="index").items()},
        "theta_source": theta_source,
        "output_tag": output_tag,
        "theta_star_mid": theta_star,
        "theta_det_mid": theta_det,
        "theta_star_all": theta_star_all,
        "theta_det_all": theta_det_all,
        "sss_at_theta_star": float(sss_grid.set_index("sigma").loc[theta_star, "sss"]),
        "mid_flip_at_theta_det": det_grid.set_index("sigma").loc[theta_det].to_dict(),
        "full_pipeline": full,
        "caption_rewrite_rate": {
            "clean": float(df.loc[~y, "rewrite_rate"].mean()) if has_clean else None,
            "attack": float(df.loc[y, "rewrite_rate"].mean()),
        },
        "feature_means": {
            ("attack" if k == 1 else "clean"): v for k, v in df.groupby("label")[FEATURES].mean().to_dict(
                orient="index").items()
        },
        "caption_calls_per_row": {"total": 11, "unique_inputs": 3},
    }
    if exp == "exp1":
        sub = {}
        for tag, mask in (("attack_success", y & df["log_attack_success"].values),
                          ("attack_failed", y & ~df["log_attack_success"].values)):
            sub[tag] = {
                "n": int(mask.sum()),
                "stage1_high_TPR": float((df.loc[mask, "bin"] == "high").mean()),
                "full_TPR": float(df.loc[mask, "blocked_full"].mean()),
                "bins": df.loc[mask, "bin"].value_counts().reindex(["low", "mid", "high"], fill_value=0).to_dict(),
            }
            if has_clean:
                sub[tag]["stage1_auroc_vs_clean"] = float(
                    roc_auc_score(np.r_[np.ones(mask.sum()), np.zeros((~y).sum())],
                                  np.r_[df.loc[mask, "risk"].values, df.loc[~y, "risk"].values]))
        summary["exp1_by_attack_outcome"] = sub

    keep = ["image_id", "photo_id", "label", *FEATURES, "rewrite_rate", "risk", "bin", "pred_ref",
            *[f"pred_{s}" for s in SIGMAS], "original_prediction", "ground_truth", "log_attack_success",
            "fooled_before", "blocked_full", "served_full", "caption_orig", "caption_rot_minus5",
            "caption_rot_plus5"]
    if "ground_truth" not in df.columns:
        df["ground_truth"] = ""
    df[keep].to_csv(out_dir / "rows_scored.csv", index=False, encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float),
                                          encoding="utf-8")
    print(f"[{exp}/{captioner}] AUROC={summary['stage1_risk'].get('auroc', float('nan')):.4f} "
          f"mid={summary['mid_rate']:.3f} theta*={theta_star} theta_det={theta_det} "
          f"full TPR={full['TPR']:.3f} FPR={full['FPR']:.3f} effASR={full['effective_asr']:.3f}", flush=True)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--exp", nargs="+", required=True)
    p.add_argument("--captioner", nargs="+", required=True)
    p.add_argument("--vqa-backend", default=None, choices=("blip", "git", "gemma3", "florence2"))
    p.add_argument("--output-tag", default=None, help="subdir suffix; default vqa_<backend>_<tag>")
    p.add_argument("--fix-theta-star", type=float, default=None)
    p.add_argument("--fix-theta-det", type=float, default=None)
    args = p.parse_args()
    import os
    default_vqa = "florence2" if os.getenv("VQA_STUDY") == "florence_victim" else "blip"
    vqa_backend = args.vqa_backend or default_vqa
    for e in args.exp:
        for c in args.captioner:
            run(
                e, c,
                vqa_backend=vqa_backend,
                output_tag=args.output_tag,
                fix_theta_star=args.fix_theta_star,
                fix_theta_det=args.fix_theta_det,
            )
