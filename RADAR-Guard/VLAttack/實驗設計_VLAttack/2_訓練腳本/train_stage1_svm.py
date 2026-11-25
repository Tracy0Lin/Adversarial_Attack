"""Standalone Stage 1 RBF-SVM trainer & verifier for RADAR-Guard (VLAttack).

Reads the extracted 4-D angular dispersion features (d_center, d_iqr, d_tail, d_q95),
runs 5-fold GroupKFold (grouped by photo_id to avoid photo leakage) for
out-of-fold evaluation, then fits the deployment Pipeline(StandardScaler, SVC)
on all rows and exports .joblib artifacts + model_metadata.json.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

FEATURE_COLS = ["d_center", "d_iqr", "d_tail", "d_q95"]
ANG_COLS = ["d_center_ang", "d_iqr_ang", "d_tail_ang", "d_q95_ang"]

THIS_FILE = Path(__file__).resolve()
EXP_ROOT = THIS_FILE.parents[1]  # 實驗設計_VLAttack
DELIVERABLE_ROOT = EXP_ROOT.parent
DATA_TAB_ROOT = EXP_ROOT / "5_資料集" / "tabular_features_and_logs"
PARAMS_OUT_DIR = EXP_ROOT / "6_模型參數"
SYS_MODELS_DIR = DELIVERABLE_ROOT / "系統" / "1_系統完整程式" / "models_params"


def infer_photo_id(series: pd.Series) -> pd.Series:
    def _parse(val: str) -> str:
        s = str(val)
        m = re.search(r"(\d{6,})", s)
        if m:
            return m.group(1)[:-3] if len(m.group(1)) >= 9 else m.group(1)
        return s.split("_")[0]

    return series.map(_parse)


def make_svm_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=1.0, class_weight="balanced", probability=False)),
    ])


def evaluate_group_kfold(df: pd.DataFrame, feature_cols: list[str], group_col: str, n_splits: int = 5) -> dict:
    X = df[feature_cols].to_numpy(dtype=float)
    y = df["label"].astype(int).to_numpy()
    groups = df[group_col].astype(str).to_numpy()

    gkf = GroupKFold(n_splits=n_splits)
    oof_margin = np.zeros(len(df), dtype=float)
    oof_pred = np.zeros(len(df), dtype=int)

    for tr_idx, va_idx in gkf.split(X, y, groups):
        pipe = make_svm_pipeline()
        pipe.fit(X[tr_idx], y[tr_idx])
        oof_margin[va_idx] = pipe.decision_function(X[va_idx])
        oof_pred[va_idx] = (oof_margin[va_idx] >= 0.0).astype(int)

    return {
        "n_rows": int(len(df)),
        "n_groups": int(len(np.unique(groups))),
        "n_splits": int(n_splits),
        "oof_auroc": round(float(roc_auc_score(y, oof_margin)), 4),
        "oof_recall": round(float(recall_score(y, oof_pred)), 4),
        "oof_precision": round(float(precision_score(y, oof_pred)), 4),
        "oof_f1": round(float(f1_score(y, oof_pred)), 4),
    }


def train_and_export(
    name: str,
    csv_path: Path,
    out_joblib: Path,
    feature_cols: list[str],
    method_filter: str | None = None,
) -> dict:
    df = pd.read_csv(csv_path)
    if method_filter is not None and "method_type" in df.columns:
        df = df[df["method_type"] == method_filter].copy()
    if df.empty:
        raise ValueError(f"Empty dataset after filter for {csv_path} ({method_filter})")

    if "photo_id" in df.columns:
        group_col = "photo_id"
    elif "source_image_id" in df.columns:
        group_col = "source_image_id"
    else:
        df["_inferred_photo_id"] = infer_photo_id(df["image_id"])
        group_col = "_inferred_photo_id"

    cv_metrics = evaluate_group_kfold(df, feature_cols=feature_cols, group_col=group_col, n_splits=5)

    X_all = df[feature_cols].to_numpy(dtype=float)
    y_all = df["label"].astype(int).to_numpy()
    full_pipe = make_svm_pipeline()
    full_pipe.fit(X_all, y_all)

    out_joblib.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(full_pipe, out_joblib)

    scaler: StandardScaler = full_pipe.named_steps["scaler"]
    svm: SVC = full_pipe.named_steps["svm"]
    return {
        "name": name,
        "artifact": str(out_joblib.name),
        "source_csv": str(csv_path.relative_to(DELIVERABLE_ROOT)),
        "method_filter": method_filter,
        "feature_cols": feature_cols,
        "scaler_mean": [round(float(v), 6) for v in scaler.mean_],
        "scaler_scale": [round(float(v), 6) for v in scaler.scale_],
        "n_support_vectors": [int(v) for v in svm.n_support_],
        "dual_intercept": [round(float(v), 6) for v in svm.intercept_],
        "cv_5fold_group": cv_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Train and export Stage 1 RBF-SVM models for RADAR-Guard.")
    parser.add_argument("--out-dir", type=Path, default=PARAMS_OUT_DIR)
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    SYS_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    jobs = [
        (
            "stage1_rotate5_exp1_deploy",
            DATA_TAB_ROOT / "exp1_mixed1000" / "tcvad_stage1_4_academic_dataset.csv",
            out_dir / "stage1_rbf_svm_rotate5_exp1.joblib",
            FEATURE_COLS,
            "rotate_5",
        ),
        (
            "stage1_formal1122_exp1_angular4d",
            DATA_TAB_ROOT / "rotation_cos_vs_ang" / "exp1_features.csv",
            out_dir / "stage1_rbf_svm_formal1122_exp1.joblib",
            ANG_COLS,
            None,
        ),
        (
            "stage1_formal1122_exp2_angular4d",
            DATA_TAB_ROOT / "rotation_cos_vs_ang" / "exp2_features.csv",
            out_dir / "stage1_rbf_svm_formal1122_exp2.joblib",
            ANG_COLS,
            None,
        ),
    ]

    records = []
    for name, csv_path, out_joblib, fcols, mfilter in jobs:
        if not csv_path.is_file():
            print(f"[WARN] Missing {csv_path}, skipping {name}")
            continue
        rec = train_and_export(name, csv_path, out_joblib, feature_cols=fcols, method_filter=mfilter)
        records.append(rec)
        print(
            f"[OK] {name} -> {out_joblib.name} | "
            f"OOF AUROC={rec['cv_5fold_group']['oof_auroc']:.4f}, "
            f"F1={rec['cv_5fold_group']['oof_f1']:.4f}"
        )

    deploy_src = out_dir / "stage1_rbf_svm_rotate5_exp1.joblib"
    if deploy_src.is_file():
        deploy_dst = SYS_MODELS_DIR / deploy_src.name
        joblib.dump(joblib.load(deploy_src), deploy_dst)
        print(f"[OK] Synced deployment model -> {deploy_dst}")

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "joblib_version": joblib.__version__,
        "models": records,
    }
    meta_path = out_dir / "model_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] Wrote {meta_path}")


if __name__ == "__main__":
    main()
