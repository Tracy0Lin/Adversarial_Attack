"""Answer-flip detector for the TC-VAD stage-2 mid-risk path.

Rationale: gaussian blur strips the adversarial perturbation, so re-scoring the
blurred image with the stage-1 caption SVM destroys the very signal it looks for
(measured AUROC 0.4653, i.e. worse than chance). This detector instead treats
successful purification as the evidence: a clean image's answer is held up by
real content and survives mild blur, while an adversarial answer is held up by
the perturbation and collapses once it is blurred away.

    is_adversarial  <=>  VQA(blur(x)) != VQA(x)

theta_det is swept for detection (maximise TPR-FPR) independently of the SSS
theta*, which is swept for mitigation (maximise SSS). The blurred image here is
only a probe and is never shown to a user, so its quality does not matter.

Consumes sss_per_theta_raw.csv, which already holds pred_ref / pred_denoised for
every (sample, sigma) pair, so no GPU work is needed.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

THIS_FILE = Path(__file__).resolve()
EXPERIMENT_ROOT = THIS_FILE.parent
LOW_RISK = 0.3
HIGH_RISK = 0.7


def rates(y_true, flagged):
    """TPR / FPR / Youden J of a binary decision."""
    y_true = np.asarray(y_true, dtype=bool)
    flagged = np.asarray(flagged, dtype=bool)
    n_pos, n_neg = y_true.sum(), (~y_true).sum()
    tpr = float((flagged & y_true).sum() / n_pos) if n_pos else float("nan")
    fpr = float((flagged & ~y_true).sum() / n_neg) if n_neg else float("nan")
    return tpr, fpr, tpr - fpr


def sweep_theta_det(raw):
    """Per-sigma flip detector, scored over every mid-risk sample."""
    rows = []
    for sigma, grp in raw.groupby("sigma"):
        flip = grp["pred_denoised"].astype(str) != grp["pred_ref"].astype(str)
        y = grp["label"].values == 1
        tpr, fpr, j = rates(y, flip)
        # Restricted view: only attacks that actually fooled the model can flip
        # back, so this is the optimistic ceiling rather than the honest rate.
        fooled = grp[grp["fooled_before"]]
        fooled_flip = fooled["pred_denoised"].astype(str) != fooled["pred_ref"].astype(str)
        rows.append(
            {
                "kernel": int(grp["kernel"].iloc[0]),
                "sigma": float(sigma),
                "n_total": int(len(grp)),
                "n_attack": int(y.sum()),
                "n_clean": int((~y).sum()),
                "TPR": tpr,
                "FPR": fpr,
                "youden_j": j,
                "n_attack_fooled": int(len(fooled[fooled["label"] == 1])),
                "TPR_fooled_only": float(
                    fooled_flip[fooled["label"] == 1].mean()
                ) if (fooled["label"] == 1).any() else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values("youden_j", ascending=False)


def multi_sigma_score(raw):
    """Graded score: how many sigmas in the grid flip the answer (0..n_sigma)."""
    raw = raw.copy()
    raw["flip"] = (raw["pred_denoised"].astype(str) != raw["pred_ref"].astype(str)).astype(int)
    per_sample = raw.groupby(["image_id", "label"], as_index=False)["flip"].sum()
    per_sample = per_sample.rename(columns={"flip": "flip_count"})
    n_sigma = raw["sigma"].nunique()
    y = per_sample["label"].values == 1
    auroc = float(roc_auc_score(y, per_sample["flip_count"].values))
    best = None
    for thr in range(1, n_sigma + 1):
        tpr, fpr, j = rates(y, per_sample["flip_count"].values >= thr)
        if best is None or j > best["youden_j"]:
            best = {"threshold": thr, "TPR": tpr, "FPR": fpr, "youden_j": j}
    return per_sample, auroc, best, n_sigma


def end_to_end(stage1, mid_flag, name):
    """Route stage-1 bins, delegate mid to a detector, then score the whole set."""
    df = stage1.copy()
    key = list(zip(df["image_id"].astype(str), df["label"].astype(int)))
    flagged = []
    for (qid, lab), rbin in zip(key, df["risk_bin"]):
        if rbin == "high":
            flagged.append(True)
        elif rbin == "low":
            flagged.append(False)
        else:
            flagged.append(bool(mid_flag.get((qid, lab), False)))
    tpr, fpr, j = rates(df["label"].values == 1, flagged)
    return {"pipeline": name, "n": int(len(df)), "TPR": tpr, "FPR": fpr, "youden_j": j}


def flip_flags_at_sigma(raw, sigma):
    sub = raw[raw["sigma"] == sigma]
    return {
        (r.image_id, int(r.label)): str(r.pred_denoised) != str(r.pred_ref)
        for r in sub.itertuples(index=False)
    }


def run(experiment_root=None, out_suffix=None):
    tag = "" if out_suffix is None else f"_{out_suffix}"
    root = Path(experiment_root or EXPERIMENT_ROOT).resolve()
    raw = pd.read_csv(root / "sss_per_theta_raw.csv")
    raw["image_id"] = raw["image_id"].astype(str)
    stage1 = pd.read_csv(root / "tcvad_stage1_4_academic_dataset.csv")
    stage1["image_id"] = stage1["image_id"].astype(str)
    method = raw["method_type"].iloc[0]
    stage1 = stage1[stage1["method_type"] == method].copy()

    print(f"method={method}  mid-risk rows={raw.groupby(['image_id','label']).ngroups}  stage1 rows={len(stage1)}")
    print(f"stage1 bins: {stage1['risk_bin'].value_counts().to_dict()}\n")

    sweep = sweep_theta_det(raw)
    sweep.to_csv(root / f"theta_det_sweep{tag}.csv", index=False, encoding="utf-8")
    print("=== theta_det sweep (flip detector on all mid-risk samples) ===")
    print(sweep.to_string(index=False))

    best = sweep.iloc[0]
    theta_det = {
        "scope": "detection",
        "method_type": method,
        "kernel": int(best["kernel"]),
        "sigma": float(best["sigma"]),
        "TPR": float(best["TPR"]),
        "FPR": float(best["FPR"]),
        "youden_j": float(best["youden_j"]),
        "TPR_fooled_only": float(best["TPR_fooled_only"]),
    }

    per_sample, auroc, best_multi, n_sigma = multi_sigma_score(raw)
    per_sample.to_csv(root / f"flip_count_per_sample{tag}.csv", index=False, encoding="utf-8")
    theta_det["multi_sigma"] = {
        "n_sigma": int(n_sigma),
        "auroc": auroc,
        **{k: float(v) for k, v in best_multi.items()},
    }
    print(f"\n=== multi-sigma flip_count score (0..{n_sigma}) ===")
    print(f"AUROC = {auroc:.4f}")
    print(
        f"best threshold >= {best_multi['threshold']}  TPR={best_multi['TPR']:.4f} "
        f"FPR={best_multi['FPR']:.4f} J={best_multi['youden_j']:.4f}"
    )

    # ---- three-way end-to-end comparison ----
    comparisons = [end_to_end(stage1, {}, "stage1 only (high bin)")]

    s2_path = root / "stage2_gaussian_results.csv"
    if s2_path.is_file():
        s2 = pd.read_csv(s2_path)
        s2["image_id"] = s2["image_id"].astype(str)
        flag_svm = {
            (r.image_id, int(r.label)): r.bin_stage2 == "high" for r in s2.itertuples(index=False)
        }
        comparisons.append(end_to_end(stage1, flag_svm, "stage1 + caption-SVM stage2"))

    sigma_det = theta_det["sigma"]
    comparisons.append(
        end_to_end(
            stage1,
            flip_flags_at_sigma(raw, sigma_det),
            f"stage1 + answer-flip (sigma={sigma_det})",
        )
    )

    theta_star_path = root / "sss_best_theta.json"
    if theta_star_path.is_file():
        with open(theta_star_path, "r", encoding="utf-8") as f:
            theta_star = json.load(f)
        sigma_mit = float(theta_star["sigma"])
        if sigma_mit != sigma_det:
            comparisons.append(
                end_to_end(
                    stage1,
                    flip_flags_at_sigma(raw, sigma_mit),
                    f"stage1 + answer-flip (sigma={sigma_mit}, theta*)",
                )
            )

    thr = int(best_multi["threshold"])
    counts = dict(zip(zip(per_sample["image_id"].astype(str), per_sample["label"].astype(int)),
                      per_sample["flip_count"]))
    flag_multi = {k: v >= thr for k, v in counts.items()}
    comparisons.append(
        end_to_end(stage1, flag_multi, f"stage1 + flip_count >= {thr}")
    )

    comp = pd.DataFrame(comparisons)
    comp.to_csv(root / f"detector_comparison{tag}.csv", index=False, encoding="utf-8")
    print("\n=== end-to-end comparison (low pass / high flag / mid delegated) ===")
    print(comp.to_string(index=False))

    with open(root / f"theta_det{tag}.json", "w", encoding="utf-8") as f:
        json.dump(theta_det, f, ensure_ascii=False, indent=2)
    print(f"\ntheta_det: kernel={theta_det['kernel']} sigma={theta_det['sigma']} J={theta_det['youden_j']:.4f}")
    print(f"Saved {root / f'detector_comparison{tag}.csv'}")
    print(f"Saved {root / f'theta_det{tag}.json'}")
    return sweep, comp, theta_det


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Answer-flip detector and theta_det sweep")
    parser.add_argument("--experiment_root", default=str(EXPERIMENT_ROOT))
    parser.add_argument("--out_suffix", default=None)
    args = parser.parse_args()
    run(experiment_root=args.experiment_root, out_suffix=args.out_suffix)
