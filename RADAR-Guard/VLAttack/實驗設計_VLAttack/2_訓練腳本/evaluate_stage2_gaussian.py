"""Stage-2: gaussian blur only on stage-1 mid-risk samples, then re-score with P1 SVM."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

THIS_FILE = Path(__file__).resolve()
EXPERIMENT_ROOT = THIS_FILE.parent
sys.path.insert(0, str(THIS_FILE.parent))

from evaluate_best_stage1_new import (
    BLIP_ROOT,
    FEATURE_COLS,
    HIGH_RISK,
    K_DEFAULT,
    LOW_RISK,
    METHODS,
    CaptionEmbedder,
    attach_oof_risk_scores,
    extract_four_dimensional_features,
    load_caption_model,
    load_sentence_encoder,
    margin_to_risk_score,
    resolve_device,
    risk_bin,
    set_seed,
)

PRIMARY_METHOD = "rotate_5"


def gaussian_blur_only(img, kernel=5, sigma=0.5):
    tensor = TF.to_tensor(img)
    blurred = TF.gaussian_blur(tensor, kernel_size=[kernel, kernel], sigma=[sigma, sigma])
    return TF.to_pil_image(blurred.clamp(0.0, 1.0))


def train_p1_svms(df_stage1):
    models = {}
    for method in METHODS:
        sub = df_stage1[df_stage1["method_type"] == method]
        clf = Pipeline(
            [("scaler", StandardScaler()), ("svm", SVC(kernel="rbf", C=1.0, class_weight="balanced"))]
        )
        clf.fit(sub[FEATURE_COLS].values, sub["label"].values)
        models[method] = clf
    return models


def build_oof_models(df_stage1, method):
    """Map image_id -> the fold model whose training split excludes that image_id.

    Uses the same GroupKFold-by-image_id scheme as stage 1, so a stage-2 score is
    never produced by an SVM that has already seen the sample being scored.
    """
    sub = df_stage1[df_stage1["method_type"] == method].reset_index(drop=True)
    X = sub[FEATURE_COLS].values
    y = sub["label"].values
    groups = sub["image_id"].astype(str).values
    n_splits = min(3, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    per_image = {}
    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        clf = Pipeline(
            [("scaler", StandardScaler()), ("svm", SVC(kernel="rbf", C=1.0, class_weight="balanced"))]
        )
        clf.fit(X[train_idx], y[train_idx])
        for qid in np.unique(groups[test_idx]):
            per_image[qid] = clf
    print(f"Built {n_splits}-fold OOF models for {method} covering {len(per_image)} image_ids")
    return per_image


def score_with_svm(clf, feature_row):
    X = np.array([[feature_row[c] for c in FEATURE_COLS]], dtype=np.float64)
    margin = clf.decision_function(X)[0]
    risk = float(margin_to_risk_score(margin))
    return risk, risk_bin(risk)


def run_stage2(
    experiment_root=None,
    visual_dir=None,
    caption_ckpt=None,
    device="cuda",
    k=K_DEFAULT,
    seed=42,
    kernel=5,
    sigma=0.5,
    method=PRIMARY_METHOD,
    out_suffix=None,
    theta_json=None,
    theta_mode="set",
):
    tag = "" if out_suffix is None else f"_{out_suffix}"
    # load_caption_model() chdirs into BLIP_ROOT, so every path must be absolute first.
    root = Path(experiment_root or EXPERIMENT_ROOT).resolve()
    visual_dir = Path(visual_dir or (root / "visual_results")).resolve()
    caption_ckpt = Path(caption_ckpt or (BLIP_ROOT / "checkpoints" / "model_base_caption_capfilt_large.pth")).resolve()
    device = resolve_device(device)
    set_seed(seed)

    theta_source = "cli"
    per_sample_theta = {}
    if theta_json:
        with open(theta_json, "r", encoding="utf-8") as f:
            theta = json.load(f)
        kernel, sigma = int(theta["kernel"]), float(theta["sigma"])
        theta_source = f"sss:{Path(theta_json).name}"
        print(f"theta* from SSS: kernel={kernel} sigma={sigma} (SSS={theta.get('sss')})")
    if theta_mode == "per_sample":
        ps_path = Path(theta_json).parent / "sss_per_sample.csv" if theta_json else root / "sss_per_sample.csv"
        ps = pd.read_csv(ps_path)
        per_sample_theta = {
            (str(r.image_id), int(r.label)): (int(r.theta_kernel), float(r.theta_sigma))
            for r in ps.itertuples(index=False)
        }
        theta_source = f"sss_per_sample:{ps_path.name}"
        print(f"theta* per sample: {len(per_sample_theta)} entries from {ps_path.name}")

    stage1 = pd.read_csv(root / "tcvad_stage1_4_academic_dataset.csv")
    stage1["image_id"] = stage1["image_id"].astype(str)
    if "risk_bin" not in stage1.columns:
        stage1 = attach_oof_risk_scores(stage1)

    mid = stage1[(stage1["method_type"] == method) & (stage1["risk_bin"] == "mid")].copy()
    print(f"Stage-2 gaussian blur only (k={kernel}, sigma={sigma})")
    print(f"Mid-risk rows on {method}: {len(mid)} / {len(stage1[stage1.method_type==method])}")
    if mid.empty:
        print("No mid-risk samples. Stage-2 skipped.")
        empty = pd.DataFrame()
        empty.to_csv(root / f"stage2_gaussian_results{tag}.csv", index=False)
        return empty

    oof_models = build_oof_models(stage1, method)
    caption_model, processor = load_caption_model(str(caption_ckpt), device)
    tokenizer, sent_model = load_sentence_encoder(device)
    embedder = CaptionEmbedder(caption_model, processor, tokenizer, sent_model, device)

    rows = []
    for i, rec in enumerate(mid.itertuples(index=False), 1):
        qid = str(rec.image_id)
        src = visual_dir / (f"{qid}_1_ori.png" if rec.label == 0 else f"{qid}_2_atk.png")
        img = Image.open(src).convert("RGB").resize((384, 384))
        k_i, s_i = per_sample_theta.get((qid, int(rec.label)), (kernel, sigma))
        blurred = gaussian_blur_only(img, kernel=k_i, sigma=s_i)
        feats = extract_four_dimensional_features(
            embedder, blurred, rec.method_type, k, seed, qid, f"s2|{rec.method_type}|{rec.label}"
        )
        risk2, bin2 = score_with_svm(oof_models[qid], feats)
        rows.append(
            {
                "image_id": qid,
                "label": int(rec.label),
                "kind": "clean" if rec.label == 0 else "attack",
                "method_type": rec.method_type,
                "theta_kernel": k_i,
                "theta_sigma": s_i,
                "risk_stage1": float(rec.risk_score),
                "bin_stage1": rec.risk_bin,
                "d_center_stage1": float(rec.d_center),
                "d_center_stage2": feats["d_center"],
                "d_iqr_stage2": feats["d_iqr"],
                "d_tail_stage2": feats["d_tail"],
                "d_q95_stage2": feats["d_q95"],
                "risk_stage2": risk2,
                "bin_stage2": bin2,
            }
        )
        print(f"stage2 [{i}/{len(mid)}] qid={qid} label={rec.label} {rec.risk_bin}->{bin2} risk={risk2:.3f}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(root / f"stage2_gaussian_results{tag}.csv", index=False, encoding="utf-8")

    clean = out[out["kind"] == "clean"]
    attack = out[out["kind"] == "attack"]
    clean_preserved = (clean["bin_stage2"] != "high").mean() if len(clean) else np.nan
    attack_caught = (attack["bin_stage2"] == "high").mean() if len(attack) else np.nan
    still_mid = (out["bin_stage2"] == "mid").mean()

    full = stage1[stage1["method_type"] == method].copy()
    full["key"] = full["image_id"].astype(str) + "_" + full["label"].astype(str)
    out["key"] = out["image_id"].astype(str) + "_" + out["label"].astype(str)
    full = full.merge(out[["key", "bin_stage2"]], on="key", how="left")
    full["final_bin"] = full["bin_stage2"].fillna(full["risk_bin"])

    n_clean = (full["label"] == 0).sum()
    n_atk = (full["label"] == 1).sum()
    fpr = ((full["label"] == 0) & (full["final_bin"] == "high")).sum() / max(n_clean, 1)
    tpr = ((full["label"] == 1) & (full["final_bin"] == "high")).sum() / max(n_atk, 1)
    fpr1 = ((full["label"] == 0) & (full["risk_bin"] == "high")).sum() / max(n_clean, 1)
    tpr1 = ((full["label"] == 1) & (full["risk_bin"] == "high")).sum() / max(n_atk, 1)

    summary = pd.DataFrame(
        [
            {
                "mid_risk_n": len(out),
                "mid_clean_n": int(len(clean)),
                "mid_attack_n": int(len(attack)),
                "clean_not_high_after_s2": float(clean_preserved),
                "attack_high_after_s2": float(attack_caught),
                "still_mid_rate": float(still_mid),
                "TPR_stage1_high_only": float(tpr1),
                "FPR_stage1_high_only": float(fpr1),
                "TPR_after_stage2": float(tpr),
                "FPR_after_stage2": float(fpr),
                "gaussian_kernel": kernel,
                "gaussian_sigma": sigma,
                "theta_mode": theta_mode,
                "theta_source": theta_source,
                "svm_scoring": "group_kfold_oof",
                "method_type": method,
            }
        ]
    )
    summary.to_csv(root / f"stage2_gaussian_summary{tag}.csv", index=False, encoding="utf-8")
    print("\n=== Stage-2 gaussian evaluation ===")
    print(summary.to_string(index=False))
    print(f"Saved {root / f'stage2_gaussian_results{tag}.csv'}")
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TC-VAD stage-2 gaussian on mid-risk samples")
    parser.add_argument("--experiment_root", default=str(EXPERIMENT_ROOT))
    parser.add_argument("--visual_dir", default=None)
    parser.add_argument("--caption_ckpt", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--k", type=int, default=K_DEFAULT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--method", default=PRIMARY_METHOD, choices=METHODS)
    parser.add_argument("--kernel", type=int, default=5)
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--out_suffix", default=None)
    parser.add_argument("--theta_json", default=None, help="sss_best_theta.json; overrides --kernel/--sigma")
    parser.add_argument("--theta_mode", default="set", choices=["set", "per_sample"])
    args = parser.parse_args()
    run_stage2(
        experiment_root=args.experiment_root,
        visual_dir=args.visual_dir,
        caption_ckpt=args.caption_ckpt,
        device=args.device,
        k=args.k,
        seed=args.seed,
        kernel=args.kernel,
        sigma=args.sigma,
        method=args.method,
        out_suffix=args.out_suffix,
        theta_json=args.theta_json,
        theta_mode=args.theta_mode,
    )
