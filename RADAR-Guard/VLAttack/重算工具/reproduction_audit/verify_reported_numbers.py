"""Reproduction audit for the VLAttack headline numbers (CPU only, read-only).

Recomputes every number quoted in README.md "核心實驗驗證摘要" from the CSV/JSON
files already in this repo, reloads the shipped .joblib models, and writes
reproduction_audit.json next to this script. No experiment file is modified.

Run from anywhere with a Python that has numpy, pandas, scikit-learn==1.6.1, joblib:
    python 重算工具/reproduction_audit/verify_reported_numbers.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import sys
import warnings
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # RADAR-Guard/VLAttack
EXP = ROOT / "實驗"
DATA = EXP / "資料"
OLD = DATA / "BLIP" / "舊流程"          # exp1_mixed1000, exp2_success1000, steps100_alpha0.002_run1000
MAIN = DATA / "BLIP" / "主流程_multi_captioner"
FORMAL = DATA / "BLIP" / "Stage1正式_11-22"
FLO = DATA / "Florence"
PARAMS = EXP / "模型"
SYS_MODELS = ROOT / "系統" / "1_系統完整程式" / "models_params"

STATS = ["d_center", "d_iqr", "d_tail", "d_q95"]
LOW, HIGH = 0.30, 0.70

checks: list[dict] = []


def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def check(group, name, reported, recomputed, source, tol, note=""):
    """Record one comparison. `reported` is what README / a result file states."""
    diff = None if reported is None or recomputed is None else abs(float(reported) - float(recomputed))
    status = "PASS" if diff is not None and diff <= tol else "MISMATCH"
    checks.append({
        "group": group, "name": name, "reported": reported,
        "recomputed": None if recomputed is None else round(float(recomputed), 6),
        "abs_diff": None if diff is None else round(diff, 6), "tolerance": tol,
        "status": status, "source": source, "note": note,
    })


def skip(group, name, source, note):
    """Record a check that cannot run because its input is not in this repo."""
    checks.append({"group": group, "name": name, "reported": None, "recomputed": None, "abs_diff": None,
                   "tolerance": None, "status": "SKIPPED", "source": source, "note": note})


def svm():
    return Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel="rbf", C=1.0, class_weight="balanced"))])


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ------------------------------------------------------------------ 1. attack
def audit_attack():
    p = OLD / "exp1_mixed1000" / "dataset_manifest.csv"
    df = pd.read_csv(p)
    succ = df["attack_success"].astype(str).str.lower().eq("true")
    check("1 攻擊", "Exp1 ASR（462/1000）", 0.462, succ.mean(), rel(p), 1e-9)
    check("1 攻擊", "Exp1 平均 SSIM", 0.8003, df["ssim"].mean(), rel(p), 5e-5)


# ------------------------------------------------- 2. Stage 1 formal 11/22 OOF
def audit_stage1_formal():
    """Same protocol as 實驗/程式/3_評估/run_rotation_cos_vs_ang.py: 5-fold GroupKFold by photo_id."""
    summary = pd.read_csv(FORMAL / "svm_summary.csv")
    rpf = pd.read_csv(FORMAL / "formal_1122_rpf1.csv", encoding="utf-8-sig")
    readme = {  # README.md 核心實驗驗證摘要 2.
        ("exp1", "cosine_4d"): {"auroc": 0.6640},
        ("exp1", "angular_4d"): {"auroc": 0.6653, "recall": 0.4920, "precision": 0.6694, "f1": 0.5671},
        ("exp2", "cosine_4d"): {"auroc": 0.7217},
        ("exp2", "angular_4d"): {"auroc": 0.7233, "recall": 0.6790, "precision": 0.6893, "f1": 0.6841},
    }
    for exp in ["exp1", "exp2"]:
        p = FORMAL / f"{exp}_features.csv"
        df = pd.read_csv(p).reset_index(drop=True)
        y = df["label"].astype(int).to_numpy()
        g = df["photo_id"].to_numpy()
        for feat, suffix in [("cosine_4d", "_cos"), ("angular_4d", "_ang")]:
            X = df[[s + suffix for s in STATS]].to_numpy(float)
            margin = np.zeros(len(df))
            pred = np.zeros(len(df), int)
            for tr, te in GroupKFold(n_splits=5).split(X, y, g):
                m = svm().fit(X[tr], y[tr])
                margin[te] = m.decision_function(X[te])
                pred[te] = m.predict(X[te])
            got = {"auroc": roc_auc_score(y, margin), "recall": recall_score(y, pred),
                   "precision": precision_score(y, pred), "f1": f1_score(y, pred)}
            src = rel(p) + "（5-fold GroupKFold by photo_id）"
            row = summary[(summary.experiment == exp) & (summary.feature == feat)].iloc[0]
            check("2 Stage1 正式 11/22", f"{exp} {feat} AUROC vs svm_summary.csv", float(row.auroc), got["auroc"], src, 1e-6)
            cfg = "有做角距離 (角距四維)" if feat == "angular_4d" else "未做角距離 (cosine 四維)"
            r = rpf[(rpf.experiment == exp) & (rpf.configuration == cfg)].iloc[0]
            for k, col in [("recall", "Recall"), ("precision", "Precision"), ("f1", "F1")]:
                check("2 Stage1 正式 11/22", f"{exp} {feat} {k} vs formal_1122_rpf1.csv", float(r[col]), got[k], src, 5e-5)
            for k, v in readme[(exp, feat)].items():
                check("2 Stage1 正式 11/22", f"{exp} {feat} {k}（README）", v, got[k], src, 5e-5)


# ------------------------------- 3. stored OOF risk_score used by Stage 2 routing
def audit_stage1_routing():
    """Same protocol as evaluate_best_stage1_new.attach_oof_risk_scores (3-fold, grouped by question id)."""
    for exp in ["exp1_mixed1000", "exp2_success1000"]:
        p = OLD / exp / "tcvad_stage1_4_academic_dataset.csv"
        df = pd.read_csv(p)
        sub = df[df.method_type == "rotate_5"].reset_index(drop=True)
        X, y = sub[STATS].to_numpy(float), sub["label"].astype(int).to_numpy()
        groups = sub["image_id"].astype(str).to_numpy()
        risk = np.zeros(len(sub))
        for tr, te in GroupKFold(n_splits=min(3, len(np.unique(groups)))).split(X, y, groups=groups):
            m = svm().fit(X[tr], y[tr])
            risk[te] = 1 / (1 + np.exp(-np.clip(m.decision_function(X[te]), -50, 50)))
        diff = float(np.abs(risk - sub["risk_score"].to_numpy()).max())
        check("3 Stage1 分流（Stage 2 前置）", f"{exp} rotate_5 risk_score 逐列最大誤差", 0.0, diff, rel(p), 1e-9,
              "此流程以題號（image_id）分組 3 折，同一照片的不同題目可能跨折；見 README 說明")
        binned = np.where(risk <= LOW, "low", np.where(risk >= HIGH, "high", "mid"))
        mism = int((binned != sub["risk_bin"].to_numpy()).sum())
        check("3 Stage1 分流（Stage 2 前置）", f"{exp} rotate_5 risk_bin 不一致筆數", 0, mism, rel(p), 0)


# ------------------------------------------------------------- 4. Stage 2 (exp2)
def audit_stage2():
    raw_p = OLD / "exp2_success1000" / "sss_per_theta_raw.csv"
    best_p = OLD / "exp2_success1000" / "sss_best_theta.json"
    raw = pd.read_csv(raw_p)
    best = json.loads(best_p.read_text(encoding="utf-8"))
    g = raw[(raw.kernel == 5) & (raw.sigma == 0.8) & (raw.method_type == "rotate_5")]
    atk = g[(g.kind == "attack") & (g.fooled_before.astype(bool))]
    delta_asr = atk.fooled_before.astype(bool).mean() - atk.fooled_after.astype(bool).mean()
    iqp, sp = g.iqp.mean(), g.sp.mean()
    sss = (max(delta_asr, 0) * iqp * sp) ** (1 / 3)
    for k, v in [("delta_asr", delta_asr), ("iqp", iqp), ("sp", sp), ("sss", sss)]:
        check("4 Stage2 SSS σ*=0.8", f"Exp2 中風險 {k} vs sss_best_theta.json", best[k], v, rel(raw_p), 1e-9)
    for k, v in [("delta_asr", 0.4571), ("iqp", 0.4596), ("sp", 0.9126), ("sss", 0.5766)]:
        check("4 Stage2 SSS σ*=0.8", f"Exp2 中風險 {k}（README）", v, best[k], rel(best_p), 5e-5)

    det_p = OLD / "exp2_success1000" / "baseline_exp1" / "theta_det.json"
    sweep_p = OLD / "exp2_success1000" / "baseline_exp1" / "theta_det_sweep.csv"
    det = json.loads(det_p.read_text(encoding="utf-8"))
    for k, v in [("TPR", 0.4066), ("FPR", 0.0532), ("youden_j", 0.3535), ("TPR_fooled_only", 0.8303)]:
        check("4 Stage2 翻轉偵測 σ_det=1.5", f"Exp1 {k}（README）", v, det[k], rel(det_p), 5e-5)
    sweep = pd.read_csv(sweep_p)
    row = sweep[(sweep.get("kernel", 5) == 5) & (np.isclose(sweep.sigma, 1.5))]
    if len(row):
        for k in ["TPR", "FPR"]:
            if k in row.columns:
                check("4 Stage2 翻轉偵測 σ_det=1.5", f"Exp1 {k} theta_det.json vs theta_det_sweep.csv",
                      det[k], float(row.iloc[0][k]), rel(sweep_p), 1e-9, "兩個結果檔彼此一致性（逐張翻轉紀錄不在 repo）")


# --------------------------------------------------------- 5. end-to-end ASR
def audit_end_to_end():
    # Exp2 40.9%: exp2_success1000 pipeline (stage1 bins + σ=0.8 answers)
    s1 = pd.read_csv(OLD / "exp2_success1000" / "tcvad_stage1_4_academic_dataset.csv")
    s1 = s1[(s1.method_type == "rotate_5") & (s1.label == 1)]
    raw = pd.read_csv(OLD / "exp2_success1000" / "sss_per_theta_raw.csv")
    fa = raw[(raw.sigma == 0.8) & (raw.kind == "attack")].set_index("image_id").fooled_after.astype(bool)
    low = (s1.risk_bin == "low").sum()
    mid_ids = s1[s1.risk_bin == "mid"].image_id
    still = low + fa.reindex(mid_ids).fillna(False).astype(bool).sum()
    check("5 端到端 ASR", "Exp2 防禦後 ASR（README 40.9%）", 0.409, still / len(s1),
          rel(OLD / "exp2_success1000") + "/{tcvad_stage1_4_academic_dataset,sss_per_theta_raw}.csv", 5e-4,
          "低風險放行 + 中風險 σ=0.8 後仍答錯；高風險攔截")

    # Exp1 24.1% / Exp3 36.0%: multi_captioner_blip sigma08_fixed, A3_full_mitigate_only (analyze.py)
    for exp, reported in [("exp1", 0.241), ("exp3", 0.360)]:
        p = MAIN / f"{exp}_blip" / "sigma08_fixed" / "rows_scored.csv"
        df = pd.read_csv(p)
        y = df.label.to_numpy() == 1
        blocked = df.bin.to_numpy() == "high"
        served = np.where(df.bin == "mid", df["pred_0.8"], df["pred_ref"])
        ori = df["victim_answer"] if "victim_answer" in df.columns else df["original_prediction"]
        still = (y & ~blocked & (served != ori.to_numpy())).sum()
        rate = still / y.sum()
        pv = pd.read_csv(p.with_name("pipeline_variants.csv"))
        stored = float(pv[pv.variant == "A3_full_mitigate_only"].effective_asr.iloc[0])
        check("5 端到端 ASR", f"{exp} A3_full_mitigate_only vs pipeline_variants.csv", stored, rate, rel(p), 1e-9)
        check("5 端到端 ASR", f"{exp} 防禦後 ASR（README {reported:.1%}）", reported, rate, rel(p), 5e-4,
              f"重算 {still}/{int(y.sum())}")


# ------------------------------------------------------------ 6. shipped models
def audit_models():
    spec = importlib.util.spec_from_file_location("train_stage1_svm", EXP / "程式" / "2_訓練" / "train_stage1_svm.py")
    tr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tr)
    meta = json.loads((PARAMS / "model_metadata.json").read_text(encoding="utf-8"))
    jobs = {
        "stage1_rotate5_exp1_deploy": (OLD / "exp1_mixed1000" / "tcvad_stage1_4_academic_dataset.csv", tr.FEATURE_COLS, "rotate_5"),
        "stage1_formal1122_exp1_angular4d": (FORMAL / "exp1_features.csv", tr.ANG_COLS, None),
        "stage1_formal1122_exp2_angular4d": (FORMAL / "exp2_features.csv", tr.ANG_COLS, None),
    }
    for rec in meta["models"]:
        csv, cols, mfilter = jobs[rec["name"]]
        df = pd.read_csv(csv)
        if mfilter:
            df = df[df.method_type == mfilter].copy()
        art = PARAMS / rec["artifact"]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            model = joblib.load(art)
        X, y = df[cols].to_numpy(float), df["label"].astype(int).to_numpy()
        refit = svm().fit(X, y)
        diff = float(np.abs(model.decision_function(X) - refit.decision_function(X)).max())
        check("6 模型檔", f"{rec['artifact']} 與 CSV 重訓 margin 最大誤差", 0.0, diff, rel(art), 1e-9,
              "載入時無 sklearn 版本警告")
        if "photo_id" in df.columns:
            gcol = "photo_id"
        else:
            df["_g"] = tr.infer_photo_id(df["image_id"])
            gcol = "_g"
        cv = tr.evaluate_group_kfold(df, cols, gcol, 5)
        for k in ["oof_auroc", "oof_recall", "oof_precision", "oof_f1"]:
            check("6 模型檔", f"{rec['name']} {k} vs model_metadata.json", rec["cv_5fold_group"][k], cv[k], rel(PARAMS / "model_metadata.json"), 1e-9,
                  "此列以訓練腳本原本的分組方式重算（207 張照片切成 333 組）" if gcol == "_g" else "")
    # train_stage1_svm.py copies the deploy model with joblib.dump(joblib.load(...)), so the bytes
    # can differ; what matters is that the website model scores identically.
    deploy = PARAMS / "stage1_rbf_svm_rotate5_exp1.joblib"
    sys_copy = SYS_MODELS / deploy.name
    df = pd.read_csv(jobs["stage1_rotate5_exp1_deploy"][0])
    X = df[tr.FEATURE_COLS].to_numpy(float)
    a, b = joblib.load(deploy), joblib.load(sys_copy)
    diff = float(np.abs(a.decision_function(X) - b.decision_function(X)).max())
    same_hash = sha256(deploy) == sha256(sys_copy)
    check("6 模型檔", "網站 models_params 部署模型與 實驗/模型 版本 margin 最大誤差", 0.0, diff, rel(sys_copy), 0.0,
          "SHA-256 相同" if same_hash else "SHA-256 不同（複製時重新序列化），但模型內容與輸出完全相同")


# ------------------------------------- 7. analyze.py runs (BLIP + Florence-2 victims)
# Mirrors 實驗/程式/3_評估/analyze.py: stage 1 OOF risk, routing, stage 2 decisions, pipeline variants.
SIGMAS = ["0.3", "0.5", "0.8", "1.0", "1.5"]
VARIANT_MODES = {
    "A0_no_defense": "none", "A1_stage1_high_block": "s1_high", "A1b_stage1_mid_high_block": "s1_midhigh",
    "A2_stage2_only": "s2_only", "A2b_stage2_mitigate_only": "s2_mitigate_only",
    "A3_full_mitigate_only": "full_mitigate_only", "A3_full_pipeline": "full",
}
VARIANT_FIELDS = ["TP", "FP", "FN", "TN", "asr_before", "effective_asr", "attacks_fooled_before",
                  "attacks_still_fooling", "clean_service_rate", "clean_blocked_rate",
                  "clean_answer_changed_rate", "vqa_calls_per_row"]


def _true(s: pd.Series) -> np.ndarray:
    return s.astype(str).str.lower().eq("true").to_numpy()


def load_scored(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    df["label"] = df["label"].astype(int)
    # analyze.py reads photo_id as an integer; GroupKFold orders groups by value, so a string
    # photo_id gives different folds.
    df["photo_id"] = df["photo_id"].astype(int)
    for c in STATS + ["risk"]:
        df[c] = df[c].astype(float)
    for c in ["fooled_before", "blocked_full", "log_attack_success"]:
        if c in df.columns:
            df[c] = _true(df[c])
    return df


def victim_answer(df: pd.DataFrame, backend: str) -> np.ndarray:
    ori = df["original_prediction"].to_numpy()
    if backend == "florence2" and "ground_truth" in df.columns:
        gt = df["ground_truth"].str.strip().to_numpy()
        return np.where(gt != "", gt, ori)
    return ori


def _theta(v) -> str | None:
    if v is None or (isinstance(v, float) and np.isnan(v)) or v == "":
        return None
    return str(float(v))


def decide(df, mode, ts, td):
    n = len(df)
    ref = df["pred_ref"].to_numpy()
    flip = (df[f"pred_{td}"].to_numpy() != ref) if td else np.zeros(n, bool)
    mit = df[f"pred_{ts}"].to_numpy() if ts else ref
    b = df["bin"].to_numpy()
    low, mid, high = b == "low", b == "mid", b == "high"
    blocked, served, calls = np.zeros(n, bool), ref.copy(), np.ones(n)
    extra = 2.0 if ts != td else 1.0
    if mode == "s1_high":
        blocked, calls = high, np.where(high, 0, 1)
    elif mode == "s1_midhigh":
        blocked = mid | high
        calls = np.where(blocked, 0, 1)
    elif mode == "s2_only":
        blocked = flip
        served = np.where(blocked, ref, mit)
        calls = np.where(blocked, 2, 1 + extra)
    elif mode == "s2_mitigate_only":
        served = mit
    elif mode == "full":
        blocked = high | (mid & flip)
        served = np.where(mid & ~blocked, mit, ref)
        calls = np.where(high, 0, np.where(mid, np.where(flip, 2, 1 + extra), 1))
    elif mode == "full_mitigate_only":
        blocked = high
        served = np.where(mid, mit, ref)
        calls = np.where(high, 0, 1)
    return blocked, served, calls


def outcome(df, ori, blocked, served, calls):
    y = df["label"].to_numpy() == 1
    tp, fp = int((blocked & y).sum()), int((blocked & ~y).sum())
    fn, tn = int((~blocked & y).sum()), int((~blocked & ~y).sum())
    still = y & ~blocked & (served != ori)
    nan = float("nan")
    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "asr_before": float(df.loc[y, "fooled_before"].mean()) if y.any() else nan,
        "effective_asr": float(still[y].mean()) if y.any() else nan,
        "attacks_fooled_before": int(df.loc[y, "fooled_before"].sum()),
        "attacks_still_fooling": int(still.sum()),
        "clean_service_rate": float((~blocked & (served == ori))[~y].mean()) if (~y).any() else nan,
        "clean_blocked_rate": float(blocked[~y].mean()) if (~y).any() else nan,
        "clean_answer_changed_rate": float((~blocked[~y] & (served[~y] != ori[~y])).mean()) if (~y).any() else nan,
        "vqa_calls_per_row": float(np.mean(calls)),
    }


def _maxdiff(a: dict, b: dict, keys) -> float:
    worst = 0.0
    for k in keys:
        x, z = a.get(k), b.get(k)
        x = np.nan if x is None or x == "" else float(x)
        z = np.nan if z is None or z == "" else float(z)
        if np.isnan(x) and np.isnan(z):
            continue
        if np.isnan(x) or np.isnan(z):
            return float("inf")
        worst = max(worst, abs(x - z))
    return worst


EXPS = ["exp1", "exp2", "exp3"]


def _blip_inputs(e):
    d = MAIN / f"{e}_blip"
    return d / "probe_gaussian.csv", d / "caption_features.csv"


def _florence_inputs(study):
    return lambda e: (FLO / study / e / "probe_gaussian_florence2.csv", FLO / study / e / "florence2" / "caption_features.csv")


ANALYZE_RUNS = [
    # (label, backend, {exp: result directory}, exp -> (probe csv, caption_features csv))
    ("BLIP sigma08_fixed", "blip",
     {e: MAIN / f"{e}_blip" / "sigma08_fixed" for e in EXPS}, _blip_inputs),
    ("BLIP 預設 θ（C1～C7 來源）", "blip",
     {e: MAIN / f"{e}_blip" for e in EXPS}, _blip_inputs),
    ("Florence-2 sigma08_fixed（florence_vlattack，報告用）", "florence2",
     {e: FLO / "florence_vlattack" / e / "florence2" / "vqa_florence2_sigma08_fixed" for e in EXPS}, _florence_inputs("florence_vlattack")),
    ("Florence-2 預設 θ（florence_vlattack）", "florence2",
     {e: FLO / "florence_vlattack" / e / "florence2" / "vqa_florence2" for e in EXPS}, _florence_inputs("florence_vlattack")),
    ("Florence_victim sigma08_fixed", "florence2",
     {e: FLO / "florence_victim" / e / "florence2" / "vqa_florence2_sigma08_fixed" for e in EXPS}, _florence_inputs("florence_victim")),
    ("Florence_victim 預設 θ（C1～C7 來源）", "florence2",
     {e: FLO / "florence_victim" / e / "florence2" / "vqa_florence2" for e in EXPS}, _florence_inputs("florence_victim")),
]


def _geo(*vals):
    return float(np.exp(np.mean(np.log(np.clip(np.asarray(vals, float), 1e-6, 1.0)))))


def sss_grid_from_probe(mid: pd.DataFrame, probe_p: Path, cap_p: Path, backend: str) -> pd.DataFrame:
    """analyze.pick_theta_star: IQP and SP per sigma over the mid-risk pool, SSS = geo-mean(ΔASR, IQP, SP)."""
    probe = pd.read_csv(probe_p, dtype=str, keep_default_na=False)
    probe["label"] = probe["label"].astype(int)
    probe["sigma"] = probe["sigma"].astype(float)
    for c in ["iqp", "clip_sim"]:
        probe[c] = pd.to_numeric(probe[c], errors="coerce")
    cap = pd.read_csv(cap_p, dtype={"image_id": str}, keep_default_na=False)
    cap["label"] = cap["label"].astype(int)
    keys = mid[["image_id", "label"]].copy()
    cap = keys.merge(cap, on=["image_id", "label"], how="left")
    fo = mid[(mid["label"] == 1) & mid["fooled_before"]]
    fo_ori = victim_answer(fo, backend)
    rows = []
    for s in SIGMAS:
        p = keys.merge(probe[probe["sigma"] == float(s)], on=["image_id", "label"], how="left")
        sp_rows = np.exp(np.mean(np.log(np.clip(np.vstack([p["clip_sim"].clip(0, 1).to_numpy(float),
                                                             pd.to_numeric(cap[f"mss_{s}"]).to_numpy(float)]), 1e-6, 1.0)), axis=0))
        d_asr = 1 - float((fo[f"pred_{s}"].to_numpy() != fo_ori).mean()) if len(fo) else np.nan
        iqp, sp = float(p["iqp"].mean()), float(np.mean(sp_rows))
        rows.append({"sigma": float(s), "iqp": iqp, "sp": sp, "sss": _geo(max(d_asr, 0.0), iqp, sp) if not np.isnan(d_asr) else np.nan})
    return pd.DataFrame(rows)


def audit_analyze_runs():
    for label, backend, dirs, inputs in ANALYZE_RUNS:
        grp = f"7 {label}"
        for exp, d in dirs.items():
            p = d / "rows_scored.csv"
            if not p.is_file():
                skip(grp, f"{exp} 全部檢查", rel(d), "rows_scored.csv 不在 repo（原電腦未複製）")
                continue
            df = load_scored(p)
            summ = json.loads((d / "summary.json").read_text(encoding="utf-8"))
            y = df["label"].to_numpy()
            X = df[STATS].to_numpy(float)
            if exp == "exp3":
                train_p = dirs["exp2"] / "rows_scored.csv"
                tr_df = load_scored(train_p)
                m = svm().fit(tr_df[STATS].to_numpy(float), tr_df["label"].to_numpy())
                risk = 1 / (1 + np.exp(-np.clip(m.decision_function(X), -50, 50)))
                how = f"以 {rel(train_p)} 全部 2,000 列訓練後套用"
            else:
                risk = np.zeros(len(df))
                for tr, te in GroupKFold(n_splits=5).split(X, y, df["photo_id"].to_numpy()):
                    m = svm().fit(X[tr], y[tr])
                    risk[te] = 1 / (1 + np.exp(-np.clip(m.decision_function(X[te]), -50, 50)))
                how = "5-fold GroupKFold by photo_id"
            check(grp, f"{exp} 風險分數逐列最大誤差", 0.0, float(np.abs(risk - df["risk"]).max()), rel(p) + f"（{how}）", 1e-9)
            binned = np.where(risk <= LOW, "low", np.where(risk >= HIGH, "high", "mid"))
            check(grp, f"{exp} 低／中／高分類不一致筆數", 0, int((binned != df["bin"]).sum()), rel(p), 0)

            ori = victim_answer(df, backend)
            fooled = (df["label"] == 1).to_numpy() & (df["pred_ref"].to_numpy() != ori)
            bad_rows = df.loc[fooled != df["fooled_before"].to_numpy()]
            note = ""
            if len(bad_rows) and (bad_rows["label"] == 0).all():
                note = (f"不一致的 {len(bad_rows)} 列都是乾淨圖（label=0）卻標成 fooled_before=True（題號 "
                        + "、".join(bad_rows["image_id"].astype(str)) + "）；現行 analyze.py 只會對攻擊圖標 True，"
                        "推測此檔由較早版本產生。C1～C5 只計攻擊圖，不受影響")
            check(grp, f"{exp} fooled_before 不一致筆數", 0, len(bad_rows), rel(p), 0, note)

            ts, td = _theta(summ["theta_star_mid"]), _theta(summ["theta_det_mid"])
            blocked, served, _ = decide(df, "full", ts, td)
            bad = int((blocked != df["blocked_full"]).sum() + (served != df["served_full"].to_numpy()).sum())
            check(grp, f"{exp} 完整流程攔截／回答不一致筆數（θ*={ts}, θ_det={td}）", 0, bad, rel(p), 0)

            pv = pd.read_csv(d / "pipeline_variants.csv")
            for _, row in pv.iterrows():
                mode = VARIANT_MODES[row["variant"]]
                vt_s, vt_d = _theta(row.get("theta_star")), _theta(row.get("theta_det"))
                if mode in ("s2_mitigate_only", "full_mitigate_only"):
                    vt_d = vt_d or (_theta(summ["theta_det_all"]) if mode.startswith("s2") else td)
                got = outcome(df, ori, *decide(df, mode, vt_s, vt_d))
                check(grp, f"{exp} {row['variant']}（{len(VARIANT_FIELDS)} 個欄位最大誤差）", 0.0,
                      _maxdiff(got, row.to_dict(), VARIANT_FIELDS), rel(d / "pipeline_variants.csv"), 1e-9)

            bins = df.groupby(["label", "bin"]).size().unstack(fill_value=0).reindex(columns=["low", "mid", "high"], fill_value=0)
            got_bins = {("attack" if k == 1 else "clean"): v for k, v in bins.to_dict(orient="index").items()}
            bad = sum(int(got_bins.get(k, {}).get(b, 0) != v) for k, d2 in summ["bins"].items() for b, v in d2.items())
            check(grp, f"{exp} summary.json 低／中／高計數不一致格數", 0, bad, rel(d / "summary.json"), 0)
            check(grp, f"{exp} summary.json 中風險比例", summ["mid_rate"], float((df["bin"] == "mid").mean()), rel(d / "summary.json"), 1e-9)
            if summ.get("stage1_risk"):
                check(grp, f"{exp} summary.json Stage 1 AUROC", summ["stage1_risk"]["auroc"],
                      roc_auc_score(y, df["risk"]), rel(d / "summary.json"), 1e-9)

            mid = df[df["bin"] == "mid"]
            ym = mid["label"].to_numpy() == 1
            if not (d / "theta_det_grid_mid.csv").is_file() or not (d / "sss_grid_mid.csv").is_file():
                skip(grp, f"{exp} 翻轉偵測與 SSS 格點", rel(d), "此資料夾沒有 sss_grid_mid.csv／theta_det_grid_mid.csv（原電腦只複製了計分表、流程變體與 summary）")
                continue
            if str(summ.get("theta_source", "")).startswith("exp2"):
                # analyze.py copies exp2's grids for exp3 when theta is not fixed
                names = ["sss_grid_mid.csv", "theta_det_grid_mid.csv"]
                if all((dirs["exp2"] / n).is_file() for n in names):
                    same = all(pd.read_csv(d / n).equals(pd.read_csv(dirs["exp2"] / n)) for n in names)
                    check(grp, f"{exp} 格點檔與 exp2 相同（θ 取自 exp2）", 1, int(same), rel(d), 0)
                else:
                    skip(grp, f"{exp} 格點檔與 exp2 比對（θ 取自 exp2）", rel(dirs["exp2"]), "exp2 的格點檔不在 repo")
                continue
            det = pd.read_csv(d / "theta_det_grid_mid.csv")
            worst = 0.0
            for _, r in det.iterrows():
                s = _theta(r["sigma"])
                flip = mid[f"pred_{s}"].to_numpy() != mid["pred_ref"].to_numpy()
                fo = ym & mid["fooled_before"].to_numpy()
                got = {"TPR": flip[ym].mean(), "FPR": flip[~ym].mean() if (~ym).any() else np.nan}
                got["youden_j"] = got["TPR"] - got["FPR"]
                got["TPR_fooled_only"] = flip[fo].mean() if fo.any() else np.nan
                worst = max(worst, _maxdiff(got, r.to_dict(), ["TPR", "FPR", "youden_j", "TPR_fooled_only"]))
            check(grp, f"{exp} theta_det_grid_mid（翻轉偵測格點）最大誤差", 0.0, worst, rel(d / "theta_det_grid_mid.csv"), 1e-9)

            sss = pd.read_csv(d / "sss_grid_mid.csv")
            fo = mid[(mid["label"] == 1) & mid["fooled_before"]]
            fo_ori = victim_answer(fo, backend)
            cl = mid[mid["label"] == 0]
            worst = 0.0
            for _, r in sss.iterrows():
                s = _theta(r["sigma"])
                got = {"n_pool": len(mid), "n_attack_fooled": len(fo),
                       "delta_asr": 1 - float((fo[f"pred_{s}"].to_numpy() != fo_ori).mean()) if len(fo) else np.nan,
                       "clean_answer_changed": float((cl[f"pred_{s}"] != cl["pred_ref"]).mean()) if len(cl) else np.nan}
                worst = max(worst, _maxdiff(got, r.to_dict(), list(got)))
            check(grp, f"{exp} sss_grid_mid 的 ΔASR／乾淨圖改答率最大誤差", 0.0, worst, rel(d / "sss_grid_mid.csv"), 1e-9)
            probe_p, cap_p = inputs(exp)
            if probe_p.is_file() and cap_p.is_file():
                got = sss_grid_from_probe(mid, probe_p, cap_p, backend)
                worst = 0.0
                for _, r in sss.iterrows():
                    g = got[np.isclose(got["sigma"], float(r["sigma"]))].iloc[0].to_dict()
                    worst = max(worst, _maxdiff(g, r.to_dict(), ["iqp", "sp", "sss"]))
                check(grp, f"{exp} sss_grid_mid 的 IQP／SP／SSS 最大誤差（由 probe 與 caption_features 重算）", 0.0, worst,
                      rel(probe_p), 1e-9)
            else:
                skip(grp, f"{exp} sss_grid_mid 的 IQP／SP／SSS", rel(d / "sss_grid_mid.csv"),
                     "需要逐張的影像品質分數（probe_gaussian 的 iqp、clip_sim 與 caption_features 的 mss），不在 repo")


# ------------------------------------------------------------- 8. C1～C7 routing deep dive
CAT_IDS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]


def classify(df: pd.DataFrame) -> np.ndarray:
    """Same rule as 實驗/程式/4_報告產生/stage_routing_deep_dive/build_report.classify (victim = original_prediction)."""
    v = df["original_prediction"].to_numpy()
    lab, b = df["label"].to_numpy(), df["bin"].to_numpy()
    fb, blk, srv = df["fooled_before"].to_numpy(), df["blocked_full"].to_numpy(), df["served_full"].to_numpy()
    still = (lab == 1) & fb & ~blk & (srv != v)
    clean_bad = (lab == 0) & ~blk & (srv != v)
    f = (lab == 1) & fb
    out = np.full(len(df), "", dtype=object)
    out[f & (b == "high")] = "C5"
    out[f & (b == "low") & still] = "C1"
    out[f & (b == "mid") & still] = "C2"
    out[f & (b == "low") & ~still] = "C3"
    out[f & (b == "mid") & ~still] = "C4"
    out[(lab == 0) & (b == "mid") & clean_bad] = "C6"
    out[(lab == 0) & (b == "high") & clean_bad] = "C7"
    return out


def audit_c1c7():
    base = DATA / "分流分析_C1到C7"
    for study, src_dirs in [("BLIP_victim", ANALYZE_RUNS[1][2]), ("Florence_victim", ANALYZE_RUNS[5][2])]:
        grp = f"8 C1～C7 {study}"
        counts = pd.read_csv(base / study / "counts_by_exp_category.csv", encoding="utf-8-sig")
        allrows = pd.read_csv(base / study / "all_categorized_rows.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
        allrows["label"] = allrows["label"].astype(int)
        for c in ["fooled_before", "blocked_full"]:
            allrows[c] = _true(allrows[c])
        for exp in ["exp1", "exp2", "exp3"]:
            stored = counts[counts["exp"] == exp].set_index("category")
            tagged = allrows[allrows["exp"] == exp]
            got_self = classify(tagged)
            check(grp, f"{exp} all_categorized_rows 用分類規則重算後不一致筆數", 0,
                  int((got_self != tagged["category"].to_numpy()).sum()), rel(base / study / "all_categorized_rows.csv"), 0)
            n_from_rows = tagged["category"].value_counts()
            bad = sum(int(n_from_rows.get(c, 0) != stored["n_rows"].get(c, 0)) for c in CAT_IDS)
            check(grp, f"{exp} counts_by_exp_category 與逐列資料的計數不一致類別數", 0, bad,
                  rel(base / study / "counts_by_exp_category.csv"), 0)
            src = None if src_dirs is None else src_dirs[exp] / "rows_scored.csv"
            if src is None or not src.is_file():
                skip(grp, f"{exp} 從原始計分表重算 C1～C7", rel(src), "原始計分表不在 repo，只能做上面兩項內部一致性檢查")
                continue
            df = load_scored(src)
            cat = classify(df)
            fooled_n = int(((df["label"] == 1) & df["fooled_before"]).sum())
            worst = 0.0
            for c in CAT_IDS:
                n = int((cat == c).sum())
                if c not in stored.index:
                    worst = max(worst, float(n))
                    continue
                worst = max(worst, abs(n - stored.loc[c, "n_rows"]))
                photos = df.loc[cat == c, "photo_id"].nunique()
                worst = max(worst, abs(photos - stored.loc[c, "n_photos"]))
                if c in ("C1", "C2", "C3", "C4", "C5"):
                    worst = max(worst, abs(round(100 * n / fooled_n, 2) - stored.loc[c, "pct_fooled_before"]))
            stale = ""
            if worst:
                tagged_ids = set(zip(tagged["image_id"], tagged["label"]))
                extra = [i for i, l, c in zip(df["image_id"], df["label"], cat) if c and (i, l) not in tagged_ids]
                if extra:
                    stale = (f"計分表比報告多出 {len(extra)} 筆分類樣本（題號 {'、'.join(map(str, extra[:5]))}）；"
                             f"報告的 C1～C5 百分比以 {round(100 * stored['n_rows'].get('C1', 0) / stored['pct_fooled_before'].get('C1', 1)) if stored['pct_fooled_before'].get('C1') else '?'} "
                             f"張攻擊成功圖為分母，計分表現為 {fooled_n} 張，推測報告產生後又替換了樣本（見 實驗/程式/Florence/攻擊/florence_victim/replace_exp3_one.py）")
            check(grp, f"{exp} 從 rows_scored 重算 C1～C7 的筆數／照片數／百分比最大誤差", 0.0, worst, rel(src), 1e-9, stale)
        if study == "Florence_victim":
            a = allrows[allrows["exp"] == "exp1"].drop(columns="exp").sort_values(["image_id", "label"]).reset_index(drop=True)
            b = allrows[allrows["exp"] == "exp2"].drop(columns="exp").sort_values(["image_id", "label"]).reset_index(drop=True)
            # Florence_victim attacks Florence-2 with white-box PGD: every exp1 question succeeded, and
            # build_exp2.py takes the first 1,000 successes from exp1, so exp1 and exp2 are the same set.
            logs = DATA / "實驗清單"
            l1 = json.loads((logs / "florence_victim_exp1_mixed1000_attack_dual_defense_log.json").read_text(encoding="utf-8"))
            l2 = json.loads((logs / "florence_victim_exp2_success1000_attack_dual_defense_log.json").read_text(encoding="utf-8"))
            all_success = all(bool(r.get("attack_success")) for r in l1)
            expected_exp2 = [r for r in l1 if r.get("attack_success")][:1000]
            explained = all_success and l2 == expected_exp2 and a.equals(b)
            check(grp, "exp1 與 exp2 相同的原因成立（exp1 攻擊全部成功、exp2 為 exp1 的前 1,000 筆成功題）", 1, int(explained),
                  rel(logs), 0,
                  f"exp1 攻擊紀錄 {len(l1)} 題、成功 {sum(bool(r.get('attack_success')) for r in l1)} 題；"
                  "exp2 由 build_exp2.py 從 exp1 挑出，故兩者資料與 C1～C7 計數相同，屬正常結果")


# ------------------------------------------------------------- 9. question difficulty
def _read_jsonl(folder: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(folder.glob("batch_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df["question_id"] = df["question_id"].astype(str)
    return df


def _binary(v):
    return "hard" if v == "hard" else ("not_hard" if v in {"easy", "medium"} else v)


def audit_difficulty():
    qd = DATA / "題目難度"
    grp = "9 題目難度"
    # BLIP: labels.csv = samples.csv + labels/*.jsonl (實驗/程式/5_題目難度/merge_labels.py)
    samples = pd.read_csv(qd / "samples.csv", dtype={"question_id": str})
    merged = samples.merge(_read_jsonl(qd / "labels"), on="question_id", how="left")
    stored = pd.read_csv(qd / "labels.csv", dtype={"question_id": str})
    m = merged.merge(stored[["question_id", "difficulty_3", "difficulty_binary"]], on="question_id", suffixes=("", "_s"), how="outer")
    check(grp, "BLIP labels.csv 列數 vs samples.csv", len(samples), len(stored), rel(qd / "labels.csv"), 0)
    bad = int((m["difficulty_3"].fillna("") != m["difficulty_3_s"].fillna("")).sum())
    check(grp, "BLIP labels.csv 難度與 labels/*.jsonl 合併結果不一致筆數", 0, bad, rel(qd / "labels"), 0)
    bad = int((stored["difficulty_3"].map(_binary).fillna("") != stored["difficulty_binary"].fillna("")).sum())
    check(grp, "BLIP difficulty_binary 與三級難度對應不一致筆數", 0, bad, rel(qd / "labels.csv"), 0)

    # Florence: BLIP labels reused, then florence/labels/*.jsonl overrides (merge_florence_labels.py)
    fq = qd / "florence"
    fs = pd.read_csv(fq / "samples.csv", dtype={"question_id": str})
    blip = stored[["question_id", "difficulty_3"]].copy()
    blip["question_id"] = blip["question_id"].str.replace(r"\.0$", "", regex=True)
    fm = fs.drop(columns=[c for c in ("difficulty_3", "reason", "from_blip") if c in fs.columns]).merge(blip, on="question_id", how="left")
    fresh = _read_jsonl(fq / "labels").drop_duplicates("question_id", keep="last")
    fm = fm.merge(fresh[["question_id", "difficulty_3"]], on="question_id", how="left", suffixes=("", "_new"))
    fm["difficulty_3"] = fm["difficulty_3_new"].where(fm["difficulty_3_new"].notna(), fm["difficulty_3"])
    fstored = pd.read_csv(fq / "labels.csv", dtype={"question_id": str})
    check(grp, "Florence labels.csv 列數 vs samples.csv", len(fs), len(fstored), rel(fq / "labels.csv"), 0)
    cmp = fm[["question_id", "difficulty_3"]].merge(fstored[["question_id", "difficulty_3"]], on="question_id", suffixes=("", "_s"), how="outer")
    bad = int((cmp["difficulty_3"].fillna("") != cmp["difficulty_3_s"].fillna("")).sum())
    check(grp, "Florence labels.csv 難度與（BLIP 標註＋florence/labels/*.jsonl）合併結果不一致筆數", 0, bad, rel(fq / "labels"), 0)

    # c1c7_by_difficulty.csv: no generating script in the repo.
    cpath = qd / "c1c7_by_difficulty.csv"
    c = pd.read_csv(cpath)
    bad = int((c["n"] != c[["easy", "medium", "hard", "unlabeled"]].sum(axis=1)).sum())
    check(grp, "c1c7_by_difficulty 每列 n = easy+medium+hard+unlabeled 不成立列數", 0, bad, rel(cpath), 0)
    # C1 and C5 have the same membership under every routing definition seen in the repo, so they can be
    # rebuilt from the sigma08_fixed tables and the label files.
    sources = {"BLIP": (ANALYZE_RUNS[0][2], stored), "FlorenceVLAttack": (ANALYZE_RUNS[2][2], fstored)}
    for study, (dirs, labels) in sources.items():
        lab = labels.drop_duplicates("question_id").set_index("question_id")["difficulty_3"]
        for exp, d in dirs.items():
            df = load_scored(d / "rows_scored.csv")
            cat = classify(df)
            for cid in ["C1", "C5"]:
                qids = df.loc[cat == cid, "image_id"].astype(str)
                diff = qids.map(lab)
                got = {"n": len(qids), "easy": int((diff == "easy").sum()), "medium": int((diff == "medium").sum()),
                       "hard": int((diff == "hard").sum()), "unlabeled": int(diff.isna().sum())}
                row = c[(c.study == study) & (c.exp == exp) & (c.category == cid)]
                rep = row.iloc[0].to_dict() if len(row) else {}
                check(grp, f"c1c7_by_difficulty {study} {exp} {cid} 的 n／easy／medium／hard／unlabeled 最大誤差", 0.0,
                      _maxdiff(got, rep, list(got)), rel(cpath), 0)
    regen = EXP / "程式" / "4_報告產生" / "regen_c1c7_by_difficulty.py"
    if regen.is_file():
        import subprocess
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "regen.csv"
            subprocess.run([sys.executable, "-I", str(regen), str(DATA), str(out)], check=True,
                           capture_output=True, cwd=tmp)
            new = pd.read_csv(out)
        key = ["study", "exp", "category"]
        m = c.merge(new, on=key, how="outer", suffixes=("", "_r"), indicator=True)
        cols = ["n", "easy", "medium", "hard", "unlabeled"]
        bad = int((m["_merge"] != "both").sum()) + int(sum((m[k] != m[k + "_r"]).sum() for k in cols))
        check(grp, f"還原版 regen_c1c7_by_difficulty.py 重算全部 {len(c)} 列，不一致格數", 0, bad, rel(regen), 0,
              "還原版腳本（原始產生程式已遺失）；規則：高風險擋、低風險 pred_ref、中風險 pred_0.8")
    else:
        skip(grp, "c1c7_by_difficulty 的 C2／C4／C6 與 X 類、FlorenceVictim 全部", rel(cpath),
             "產生此表的腳本不在 repo；C2／C4／C6 的分法與 build_report.py 不同（例如 BLIP exp1 C2=78 vs 25），且有未定義的 X 類")


def main():
    for fn in [audit_attack, audit_stage1_formal, audit_stage1_routing, audit_stage2, audit_end_to_end, audit_models,
               audit_analyze_runs, audit_c1c7, audit_difficulty]:
        try:
            fn()
        except Exception as exc:  # keep going; record the failure
            import traceback
            checks.append({"group": fn.__name__, "name": "執行失敗", "status": "ERROR",
                           "note": repr(exc) + " | " + traceback.format_exc(limit=3).replace("\n", " ")})
    counts = {s: sum(c["status"] == s for c in checks) for s in ["PASS", "MISMATCH", "SKIPPED", "ERROR"]}
    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                        "numpy": np.__version__, "pandas": pd.__version__, "joblib": joblib.__version__},
        "summary": counts,
        "checks": checks,
    }
    (HERE / "reproduction_audit.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    for c in checks:
        print(f"[{c['status']:8s}] {c['group']} | {c['name']} | reported={c.get('reported')} recomputed={c.get('recomputed')}"
              + (f" | {c['note']}" if c.get("status") != "PASS" and c.get("note") else ""))
    print(counts)
    return 0 if counts["ERROR"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
