"""Aggregate experiment tables + gap checklist for thesis / reports.

Reads existing CSVs only (no GPU). Writes:
  - output/VQA/data_supplement/*.csv
  - output/VQA/EXPERIMENT_DATA_SUPPLEMENT.md

Run from repo root:
  python BLIP_attack/output/VQA/build_experiment_data_supplement.py
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torchvision.transforms.functional as TF
    from PIL import Image

    _HAS_BLUR = True
except ImportError:
    _HAS_BLUR = False

VQA_ROOT = Path(__file__).resolve().parent
OUT_DIR = VQA_ROOT / "data_supplement"

EXP1 = VQA_ROOT / "steps100_alpha0.002_run1000"
EXP2 = VQA_ROOT / "exp2_success1000"

FEATURES = ["d_center", "d_iqr", "d_tail", "d_q95"]
STAGE2_FEATS = ["d_center_stage2", "d_iqr_stage2", "d_tail_stage2", "d_q95_stage2"]

EXP_META = {
    "exp1_steps100_alpha0.002_run1000": {
        "title": "實驗一（混合流量）",
        "folder": "steps100_alpha0.002_run1000",
        "attack_note": "1000 張對抗圖：462 成功 + 538 失敗",
    },
    "exp2_success1000": {
        "title": "實驗二（僅攻擊成功）",
        "folder": "exp2_success1000",
        "attack_note": "1000 張對抗圖皆攻擊成功",
    },
}

FEAT_ZH = {
    "d_center": "D_center（中心）",
    "d_iqr": "D_iqr（四分位距）",
    "d_tail": "D_tail（尾端）",
    "d_q95": "D_q95（95分位）",
}

MITIGATION_ZH = {
    "routed_low_pass_mid_denoise_high_block_sigma_0.8": "【主結論】全流程路由後 ASR",
    "mid_attack_fooled_before_sigma_0.8": "Mid 池：糊前已騙子集",
    "mid_clean_answer_changed_sigma_0.8": "Mid 池：乾淨圖糊後改答（副作用）",
    "mid_attack_all_sigma_0.8": "Mid 池：全部對抗列",
}

DETECTION_ZH = {
    "stage1 only (high bin)": "僅 Stage1（high 攔截）",
    "stage1 + caption-SVM stage2": "Stage1 + 舊版 caption-SVM",
    "stage1 + answer-flip (sigma=1.5)": "Stage1 + 答案翻轉 σ=1.5",
    "stage1 + answer-flip (sigma=0.8, theta*)": "Stage1 + 答案翻轉 σ=0.8（θ*）",
    "stage1 + flip_count >= 1": "Stage1 + 多 σ 翻轉計數≥1",
}

REPRESENTATIVE_CASES = [
    ("B_correct_attack_524361001", "攻擊成功；Stage1=high，一階即攔截"),
    ("D_miss_attack_low_262522002", "攻擊成功；Stage1=low，端到端漏抓 (FN)"),
    ("C_false_alarm_clean_high_393254002", "乾淨圖；Stage1=high，假警報 (FP)"),
    ("E_clean_mid_consistency_393777005", "乾淨圖；Stage1=mid，糊後答案仍穩定"),
    ("F_boundary_131089003", "邊界樣本；風險分數接近門檻"),
]


def _pct(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{100 * float(x):.2f}%"


def _exists(root: Path, name: str) -> bool:
    return (root / name).is_file()


def stage1_feature_summary(df: pd.DataFrame, experiment: str) -> pd.DataFrame:
    """Post-angular 4-D (official Stage 1 features)."""
    rows = []
    for label, kind in [(0, "clean"), (1, "attack")]:
        sub = df[df["label"] == label]
        for col in FEATURES:
            s = sub[col].astype(float)
            rows.append(
                {
                    "experiment": experiment,
                    "feature_stage": "stage1_post_angular_4d",
                    "label": label,
                    "kind": kind,
                    "n_rows": len(sub),
                    "feature": col,
                    "mean": float(s.mean()),
                    "std": float(s.std(ddof=1)) if len(sub) > 1 else 0.0,
                    "median": float(s.median()),
                    "q25": float(s.quantile(0.25)),
                    "q75": float(s.quantile(0.75)),
                }
            )
    return pd.DataFrame(rows)


def stage1_delta_mu(df: pd.DataFrame, experiment: str) -> pd.DataFrame:
    clean = df[df["label"] == 0][FEATURES].mean()
    adv = df[df["label"] == 1][FEATURES].mean()
    delta = adv - clean
    return pd.DataFrame(
        [
            {
                "experiment": experiment,
                "d_center": float(delta["d_center"]),
                "d_iqr": float(delta["d_iqr"]),
                "d_tail": float(delta["d_tail"]),
                "d_q95": float(delta["d_q95"]),
            }
        ]
    )


def stage2_feature_summary(path: Path, experiment: str) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    rows = []
    for label, kind in [(0, "clean"), (1, "attack")]:
        sub = df[df["label"] == label]
        for col in STAGE2_FEATS:
            s = sub[col].astype(float)
            rows.append(
                {
                    "experiment": experiment,
                    "feature_stage": "stage2_post_angular_4d_on_blurred_mid_only",
                    "theta_note": "sigma from stage2_gaussian_results (theta* from SSS)",
                    "label": label,
                    "kind": kind,
                    "n_rows": len(sub),
                    "feature": col.replace("_stage2", ""),
                    "mean": float(s.mean()),
                    "std": float(s.std(ddof=1)) if len(sub) > 1 else 0.0,
                }
            )
    if "theta_sigma" in df.columns and len(df):
        rows[0]["theta_sigma"] = float(df["theta_sigma"].iloc[0])
    return pd.DataFrame(rows)


def clean_answer_change_by_sigma(raw_path: Path, experiment: str) -> pd.DataFrame | None:
    if not raw_path.is_file():
        return None
    raw = pd.read_csv(raw_path)
    clean = raw[raw["kind"] == "clean"].copy()
    if clean.empty:
        return None
    rows = []
    for sigma, g in clean.groupby("sigma"):
        changed = (g["pred_denoised"].astype(str) != g["pred_ref"].astype(str))
        rows.append(
            {
                "experiment": experiment,
                "cohort": "mid_risk_clean_only",
                "sigma": float(sigma),
                "n": int(len(g)),
                "answer_change_rate": float(changed.mean()),
                "answer_change_count": int(changed.sum()),
                "note": "Not attack ASR; VQA answer changed after Gaussian blur on ori image",
            }
        )
    return pd.DataFrame(rows)


def load_mitigation_rows(path: Path, experiment: str) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    df.insert(0, "experiment", experiment)
    return df


def load_detection_rows(path: Path, experiment: str) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    df.insert(0, "experiment", experiment)
    return df


def load_attack_asr(root: Path, experiment: str) -> pd.DataFrame:
    rows = [{"experiment": experiment, "metric": "attack_asr_on_1000_adv", "value": np.nan, "detail": ""}]
    p_json = root / "attack_rate_summary.json"
    p_log = root / "attack_dual_defense_log.json"
    if p_json.is_file():
        with open(p_json, encoding="utf-8") as f:
            j = json.load(f)
        asr = j.get("attack_success_rate") or j.get("asr")
        if asr is not None:
            rows[0]["value"] = float(asr)
            rows[0]["detail"] = "attack_rate_summary.json"
    elif p_log.is_file():
        with open(p_log, encoding="utf-8") as f:
            log = json.load(f)
        n = len(log)
        ok = sum(1 for x in log if x.get("attack_success"))
        rows[0]["value"] = ok / max(n, 1)
        rows[0]["detail"] = f"attack_dual_defense_log.json {ok}/{n}"
    if experiment == "exp2_success1000":
        rows[0]["value"] = 1.0
        rows[0]["detail"] = "cohort: all 1000 adversarial samples are attack_success"
    return pd.DataFrame(rows)


def gap_checklist() -> pd.DataFrame:
    """What is NOT on disk / not in this supplement (action items)."""
    items = [
        {
            "priority": "optional",
            "item": "stage1_pre_angular_per_probe",
            "description": "11 cosine or delta values per image before 4-D aggregation",
            "exp1": "missing",
            "exp2": "missing",
            "action": "Extend evaluate_best_stage1_new.py to dump arrays; re-run Stage1 GPU ~1–1.5h each",
        },
        {
            "priority": "optional",
            "item": "stage2_4d_full_2000_rows",
            "description": "Stage2 caption features on all rows (not only mid-risk)",
            "exp1": "by_design_mid_only_786",
            "exp2": "by_design_mid_only_676",
            "action": "Only if ablation needs it; conflicts with pipeline routing definition",
        },
        {
            "priority": "optional",
            "item": "clean_blur_harm_low_high_bins",
            "description": "Answer-change rate on clean images in low/high risk bins @ theta*",
            "exp1": "not_logged",
            "exp2": "not_logged",
            "action": "Extra VQA+blur pass on non-mid clean (~0.5–1h GPU) if reviewer requires",
        },
        {
            "priority": "done",
            "item": "stage1_post_angular_4d",
            "description": "tcvad_stage1_4_academic_dataset.csv",
            "exp1": "present_2000",
            "exp2": "present_2000",
            "action": "Use data_supplement/stage1_4d_summary.csv",
        },
        {
            "priority": "done",
            "item": "stage2_post_angular_4d_mid",
            "description": "stage2_gaussian_results.csv (blur then same angular 4-D)",
            "exp1": "present_786",
            "exp2": "present_676",
            "action": "Use data_supplement/stage2_4d_summary.csv",
        },
        {
            "priority": "required_if_missing",
            "item": "mitigation_asr_summary",
            "description": "mitigation_summary.csv @ theta* (mitigation routing ASR)",
            "exp1": "present" if _exists(EXP1, "mitigation_summary.csv") else "missing",
            "exp2": "present" if _exists(EXP2, "mitigation_summary.csv") else "missing",
            "action": "python BLIP_attack/output/mitigation_summary.py --experiment_root <dir>",
        },
        {
            "priority": "done",
            "item": "stage2_detection_flip",
            "description": "detector_comparison.csv, theta_det_sweep.csv (not ASR)",
            "exp1": "present" if _exists(EXP1, "detector_comparison.csv") else "missing",
            "exp2": "present" if _exists(EXP2, "detector_comparison.csv") else "missing",
            "action": "Use data_supplement/stage2_detection_end2end.csv",
        },
        {
            "priority": "low",
            "item": "exp1_mixed1000_full_pipeline",
            "description": "Duplicate of run1000; incomplete SSS in exp1_mixed1000/",
            "exp1": "use_steps100_alpha0.002_run1000_instead",
            "exp2": "n/a",
            "action": "Do not duplicate; cite steps100_alpha0.002_run1000",
        },
    ]
    return pd.DataFrame(items)


def stage1_wide_table(stage1_sum: pd.DataFrame, exp_key: str) -> pd.DataFrame:
    """One row per feature: clean vs attack means."""
    sub = stage1_sum[stage1_sum["experiment"] == exp_key]
    rows = []
    for feat in FEATURES:
        c = sub[(sub["feature"] == feat) & (sub["kind"] == "clean")].iloc[0]
        a = sub[(sub["feature"] == feat) & (sub["kind"] == "attack")].iloc[0]
        rows.append(
            {
                "特徵（角距後）": FEAT_ZH[feat],
                "乾淨圖 mean±std": f"{c['mean']:.4f} ± {c['std']:.4f}",
                "對抗圖 mean±std": f"{a['mean']:.4f} ± {a['std']:.4f}",
                "對抗−乾淨": f"{a['mean'] - c['mean']:+.4f}",
            }
        )
    return pd.DataFrame(rows)


def stage2_wide_table(stage2_sum: pd.DataFrame, exp_key: str):
    sub = stage2_sum[stage2_sum["experiment"] == exp_key]
    if sub.empty:
        return pd.DataFrame()
    n_clean = int(sub[sub["kind"] == "clean"]["n_rows"].iloc[0])
    n_atk = int(sub[sub["kind"] == "attack"]["n_rows"].iloc[0])
    sigma = sub["theta_sigma"].dropna().iloc[0] if sub["theta_sigma"].notna().any() else 0.8
    rows = []
    for feat in FEATURES:
        c = sub[(sub["feature"] == feat) & (sub["kind"] == "clean")].iloc[0]
        a = sub[(sub["feature"] == feat) & (sub["kind"] == "attack")].iloc[0]
        rows.append(
            {
                "特徵（糊後再算角距）": FEAT_ZH[feat],
                f"乾淨 Mid n={n_clean}": f"{c['mean']:.4f} ± {c['std']:.4f}",
                f"對抗 Mid n={n_atk}": f"{a['mean']:.4f} ± {a['std']:.4f}",
            }
        )
    return pd.DataFrame(rows), float(sigma), n_clean, n_atk


def write_experiment_section(
    exp_key: str,
    stage1_sum: pd.DataFrame,
    delta_mu: pd.DataFrame,
    stage2_sum: pd.DataFrame,
    clean_harm: pd.DataFrame,
    mitigation: pd.DataFrame,
    detection: pd.DataFrame,
    attack_asr: pd.DataFrame,
) -> list[str]:
    meta = EXP_META[exp_key]
    lines = [
        f"## {meta['title']}",
        "",
        f"資料夾：`BLIP_attack/output/VQA/{meta['folder']}/`  ",
        f"樣本說明：{meta['attack_note']}；Stage1 各 **1000 乾淨 + 1000 對抗 = 2000 列**。",
        "",
        "### 你要的項目 → 有沒有？",
        "",
        "| 項目 | 狀態 | 去哪裡看 |",
        "|------|------|----------|",
        "| Stage1 四維（**角距後**，正式特徵） | ✅ 有 | 下表；原始列在 `tcvad_stage1_4_academic_dataset.csv` |",
        "| Stage1 **角距前**（11 次探針逐條 δ） | ❌ 沒存 | 論文主線可不寫；要補得重跑 Stage1 |",
        "| Stage2 四維（糊後角距，**僅 Mid**） | ✅ 有 | 下表；`stage2_gaussian_results.csv` |",
        "| 乾淨圖「ASR」 | ⚠️ 無 ASR | 乾淨圖報 **糊後答案是否被改**（下表） |",
        "| Stage2 **緩解** ASR（糊後仍攻擊成功） | ✅ 有 | 下表 θ*=0.8 |",
        "| Stage2 **偵測**（翻轉，不是 ASR） | ✅ 有 | 下表 TPR/FPR/J |",
        "",
    ]

    asr_row = attack_asr[attack_asr["experiment"] == exp_key].iloc[0]
    lines += [
        "### 攻擊成功率（只算對抗圖）",
        "",
        f"- **ASR = {_pct(asr_row['value'])}**（{asr_row['detail']}）",
        "",
        "### Stage1 四維（已是角距：caption → cosine → arccos/π → 四統計量）",
        "",
    ]
    s1_wide = stage1_wide_table(stage1_sum, exp_key)
    lines.append(_markdown_table(s1_wide))
    d = delta_mu[delta_mu["experiment"] == exp_key].iloc[0]
    lines += [
        "",
        f"整體位移 Δμ（對抗−乾淨）：D_center {d['d_center']:+.4f}，D_iqr {d['d_iqr']:+.4f}，"
        f"D_tail {d['d_tail']:+.4f}，D_q95 {d['d_q95']:+.4f}",
        "",
    ]

    if not stage2_sum.empty and exp_key in stage2_sum["experiment"].values:
        s2_wide, sigma, nc, na = stage2_wide_table(stage2_sum, exp_key)
        lines += [
            f"### Stage2 四維（先 5×5 高斯 σ={sigma}，再**同一套**角距四維；只含 Mid：乾淨 {nc} + 對抗 {na} 列）",
            "",
            _markdown_table(s2_wide),
            "",
            "舊版 Stage2（caption-SVM 重評）用這組特徵；**論文採用的是答案翻轉**，見下方偵測表。",
            "",
        ]

    if not clean_harm.empty:
        ch = clean_harm[clean_harm["experiment"] == exp_key][["sigma", "n", "answer_change_rate", "answer_change_count"]].copy()
        ch["sigma"] = ch["sigma"].map(lambda s: f"{s:g}")
        ch["改答比例"] = ch["answer_change_rate"].map(_pct)
        ch = ch.rename(columns={"sigma": "σ", "n": "乾淨 Mid 張數", "answer_change_count": "改答張數"})
        ch = ch[["σ", "乾淨 Mid 張數", "改答張數", "改答比例"]]
        lines += [
            "### 乾淨圖：糊後 VQA 答案改變（**不是 ASR**）",
            "",
            _markdown_table(ch),
            "",
        ]

    if not mitigation.empty:
        mit = mitigation[mitigation["experiment"] == exp_key].copy()
        key_cohorts = [c for c in MITIGATION_ZH if c in set(mit["cohort"])]
        mit = mit[mit["cohort"].isin(key_cohorts)]
        rows = []
        for _, r in mit.iterrows():
            rows.append(
                {
                    "說明": MITIGATION_ZH.get(r["cohort"], r["cohort"]),
                    "n": int(r["n"]),
                    "糊前 ASR": _pct(r["asr_before"]) if pd.notna(r["asr_before"]) else "（不適用）",
                    "糊後 ASR": _pct(r["asr_after"]) if pd.notna(r["asr_after"]) else "（不適用）",
                    "下降": f"{r['asr_drop_pp']:.1f} pp" if pd.notna(r.get("asr_drop_pp")) else "—",
                }
            )
        lines += [
            "### Stage2 緩解：高斯 @ θ*=0.8 後 ASR（對抗圖相對「原圖正確答案」仍答錯的比例）",
            "",
            _markdown_table(pd.DataFrame(rows)),
            "",
        ]

    if not detection.empty:
        det = detection[detection["experiment"] == exp_key].copy()
        det["方案"] = det["pipeline"].map(lambda p: DETECTION_ZH.get(p, p))
        det = det[["方案", "TPR", "FPR", "youden_j"]].copy()
        det["TPR"] = det["TPR"].map(lambda x: f"{100*x:.1f}%")
        det["FPR"] = det["FPR"].map(lambda x: f"{100*x:.1f}%")
        det["J"] = det["youden_j"].map(lambda x: f"{x:.3f}")
        det = det.drop(columns=["youden_j"])
        lines += [
            "### Stage2 偵測（端到端 2000 列；**不是 ASR**）",
            "",
            _markdown_table(det),
            "",
            "---",
            "",
        ]
    return lines


def load_stage1_table(root: Path) -> pd.DataFrame:
    df = pd.read_csv(root / "tcvad_stage1_4_academic_dataset.csv")
    df["image_id"] = df["image_id"].astype(str)
    return df[df["method_type"] == "rotate_5"].copy()


def stage1_routing_counts(stage1: pd.DataFrame) -> dict:
    """Per kind (attack/clean) counts of low/mid/high."""
    out = {}
    for label, kind in [(1, "attack"), (0, "clean")]:
        sub = stage1[stage1["label"] == label]
        vc = sub["risk_bin"].value_counts()
        out[kind] = {b: int(vc.get(b, 0)) for b in ("low", "mid", "high")}
    return out


def flip_count_mid_flags(root: Path) -> dict:
    raw = pd.read_csv(root / "sss_per_theta_raw.csv")
    raw["image_id"] = raw["image_id"].astype(str)
    raw["flip"] = (raw["pred_denoised"].astype(str) != raw["pred_ref"].astype(str)).astype(int)
    per = raw.groupby(["image_id", "label"], as_index=False)["flip"].sum()
    thr = 1
    td = root / "theta_det.json"
    if td.is_file():
        with open(td, encoding="utf-8") as f:
            thr = int(json.load(f).get("multi_sigma", {}).get("threshold", 1))
    return {
        (str(r.image_id), int(r.label)): int(r.flip) >= thr for r in per.itertuples(index=False)
    }


def end_to_end_flagged(stage1: pd.DataFrame, mid_flag: dict) -> tuple[np.ndarray, np.ndarray]:
    """Positive class = attack (label 1). flagged True = 判為對抗/攔截."""
    y_true = []
    flagged = []
    for row in stage1.itertuples(index=False):
        lab = int(row.label)
        y_true.append(lab == 1)
        if row.risk_bin == "high":
            flagged.append(True)
        elif row.risk_bin == "low":
            flagged.append(False)
        else:
            flagged.append(bool(mid_flag.get((str(row.image_id), lab), False)))
    return np.array(y_true, dtype=bool), np.array(flagged, dtype=bool)


def confusion_from_flags(y_true: np.ndarray, flagged: np.ndarray) -> dict:
    tp = int((flagged & y_true).sum())
    fn = int((~flagged & y_true).sum())
    fp = int((flagged & ~y_true).sum())
    tn = int((~flagged & ~y_true).sum())
    tpr = tp / (tp + fn) if (tp + fn) else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tpr
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else float("nan")
    j = tpr - fpr
    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "TPR": tpr,
        "FPR": fpr,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "J": j,
    }


def mid_pool_counts(stage1: pd.DataFrame) -> dict:
    mid = stage1[stage1["risk_bin"] == "mid"]
    return {
        "attack": int((mid["label"] == 1).sum()),
        "clean": int((mid["label"] == 0).sum()),
        "total": int(len(mid)),
    }


def parse_info_txt(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    out = {}
    for line in text.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def gaussian_blur_pil(img: Image.Image, sigma: float = 1.5, kernel: int = 5) -> Image.Image:
    tensor = TF.to_tensor(img)
    blurred = TF.gaussian_blur(tensor, kernel_size=[kernel, kernel], sigma=[sigma, sigma])
    return TF.to_pil_image(blurred.clamp(0.0, 1.0))


def export_case_studies(exp_root: Path, sigma_det: float = 1.5) -> list[str]:
    """Copy ori/atk, create blur, return markdown lines."""
    case_dir = OUT_DIR / "case_studies"
    case_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(exp_root / "sss_per_theta_raw.csv")
    raw["image_id"] = raw["image_id"].astype(str)
    lines = [
        "## ⑤ 代表性案例（Clean → Attack → Gaussian → 答案變化）",
        "",
        f"高斯為 **5×5、σ={sigma_det:g}**（與 θ_det 偵測探針同量級）。",
        "「翻轉」指 **同一張輸入圖** 在糊前 vs 糊後 VQA 預測是否不同（**不是**變成另一張圖的答案）。",
        "",
    ]
    if not _HAS_BLUR:
        lines.append("（環境缺少 PIL/torchvision，未生成模糊圖；僅文字說明。）")
        return lines

    for folder_name, caption in REPRESENTATIVE_CASES:
        src = exp_root / "teacher_examples" / folder_name
        if not src.is_dir():
            lines.append(f"- **{folder_name}**：目錄不存在，略過。")
            continue
        info = parse_info_txt(src / "info.txt")
        qid = info.get("question_id", folder_name.split("_")[-1])
        sub = raw[(raw["image_id"] == qid) & (raw["sigma"] == sigma_det)]
        atk_row = sub[sub["label"] == 1]
        clean_row = sub[sub["label"] == 0]
        risk = info.get("risk_bin", "")
        if len(atk_row):
            pred_atk_before = atk_row["pred_ref"].iloc[0]
            pred_atk_after = atk_row["pred_denoised"].iloc[0]
        else:
            pred_atk_before = info.get("adversarial_prediction", "—")
            pred_atk_after = f"（{risk}，未進 Mid 探針池）"
        if len(clean_row):
            pred_clean_before = clean_row["pred_ref"].iloc[0]
            pred_clean_after = clean_row["pred_denoised"].iloc[0]
        else:
            pred_clean_before = info.get("original_prediction", "—")
            pred_clean_after = f"（{risk}，未進 Mid 探針池）"
        flip_atk = len(atk_row) and str(pred_atk_before) != str(pred_atk_after)
        flip_clean = len(clean_row) and str(pred_clean_before) != str(pred_clean_after)

        dst = case_dir / folder_name
        dst.mkdir(parents=True, exist_ok=True)
        for name in ("ori.png", "atk.png"):
            if (src / name).is_file():
                shutil.copy2(src / name, dst / name)
        ori = Image.open(dst / "ori.png").convert("RGB") if (dst / "ori.png").is_file() else None
        atk = Image.open(dst / "atk.png").convert("RGB") if (dst / "atk.png").is_file() else None
        if ori:
            gaussian_blur_pil(ori, sigma_det).save(dst / "gaussian_ori.png")
        if atk:
            gaussian_blur_pil(atk, sigma_det).save(dst / "gaussian_atk.png")

        rel = f"data_supplement/case_studies/{folder_name}"
        lines += [
            f"### {folder_name}",
            "",
            f"**說明**：{caption}  ",
            f"**題目**：{info.get('question', '—')}  ",
            f"**Stage1**：{info.get('risk_bin', '—')}（score={info.get('risk_score', '—')}）  ",
            f"**原圖 VQA**：{info.get('original_prediction', '—')} → **對抗圖 VQA**：{info.get('adversarial_prediction', '—')}（attack_success={info.get('attack_success', '—')}）",
            "",
            "| Clean (ori) | Attack (atk) | Gaussian(ori) | Gaussian(atk) |",
            "|-------------|--------------|---------------|---------------|",
            f"| ![ori]({rel}/ori.png) | ![atk]({rel}/atk.png) | ![g_ori]({rel}/gaussian_ori.png) | ![g_atk]({rel}/gaussian_atk.png) |",
            "",
            f"- **對抗圖糊前→糊後**（σ={sigma_det:g}）：`{pred_atk_before}` → `{pred_atk_after}`，翻轉={flip_atk}",
            f"- **原圖糊前→糊後**：`{pred_clean_before}` → `{pred_clean_after}`，翻轉={flip_clean}",
            "",
        ]
    return lines


def build_detection_appendix() -> tuple[list[str], pd.DataFrame]:
    """Stage1 routing, confusion matrices, unified exp1/exp2 comparison."""
    rows_cmp = []
    lines = [
        "---",
        "",
        "# 附錄：Stage1 路由、混淆矩陣與端到端偵測（實驗一／二對照）",
        "",
        "**偵測任務**：label=1 為對抗圖、label=0 為乾淨圖；`flagged=True` 表示系統攔截（判對抗）。",
        "",
        "**最終管線（兩實驗同一套，便於對照）**：",
        "",
        "```",
        "輸入 (2000 列：1000 對抗 + 1000 乾淨)",
        "    ↓",
        "Stage 1：TC-VAD 風險分數 → low 放行 | high 攔截 | mid 進 Stage 2",
        "    ↓",
        "Stage 2（偵測）：僅 mid — 多 σ 高斯探針，任一 σ 上 VQA 答案翻轉 → 攔截",
        "              （flip_count ≥ 1，與 detector_comparison 最佳 J 一致）",
        "    ↓",
        "Final：統計 TP / FP / TN / FN → TPR、FPR、Precision、Recall、F1、J",
        "```",
        "",
        "## ① Stage 1 風險分段（Low / Mid / High）",
        "",
        "| 子集 | 實驗一 Low | 實驗一 Mid | 實驗一 High | 實驗二 Low | 實驗二 Mid | 實驗二 High |",
        "|------|-----------|-----------|------------|-----------|-----------|------------|",
    ]

    routings = {}
    finals = {}
    stage1_only = {}
    for root, tag in [(EXP1, "實驗一"), (EXP2, "實驗二")]:
        s1 = load_stage1_table(root)
        routings[tag] = stage1_routing_counts(s1)
        mid_flag = flip_count_mid_flags(root)
        y, fl = end_to_end_flagged(s1, mid_flag)
        finals[tag] = confusion_from_flags(y, fl)
        y1, fl1 = end_to_end_flagged(s1, {})
        stage1_only[tag] = confusion_from_flags(y1, fl1)

    r1, r2 = routings["實驗一"], routings["實驗二"]
    lines.append(
        f"| Attack 1000 | {r1['attack']['low']} | {r1['attack']['mid']} | {r1['attack']['high']} "
        f"| {r2['attack']['low']} | {r2['attack']['mid']} | {r2['attack']['high']} |"
    )
    lines.append(
        f"| Clean 1000 | {r1['clean']['low']} | {r1['clean']['mid']} | {r1['clean']['high']} "
        f"| {r2['clean']['low']} | {r2['clean']['mid']} | {r2['clean']['high']} |"
    )
    lines += [
        "",
        "（實驗一 Attack：320 / 391 / 289；Clean：548 / 395 / 57。實驗二 Attack：219 / 350 / 431；Clean：533 / 326 / 141。）",
        "",
        "## ② 混淆矩陣（實驗一，最終管線 flip_count≥1）",
        "",
        "正類 = **對抗圖**；預測正類 = **系統攔截**。",
        "",
    ]
    m1 = finals["實驗一"]
    m2 = finals["實驗二"]
    lines += [
        "| | 預測：對抗（攔截） | 預測：乾淨（放行） |",
        "|---|-------------------|-------------------|",
        f"| **實際：對抗** | TP = {m1['TP']} | FN = {m1['FN']} |",
        f"| **實際：乾淨** | FP = {m1['FP']} | TN = {m1['TN']} |",
        "",
        "## ②-b 混淆矩陣（實驗二，同一最終管線）",
        "",
        "| | 預測：對抗（攔截） | 預測：乾淨（放行） |",
        "|---|-------------------|-------------------|",
        f"| **實際：對抗** | TP = {m2['TP']} | FN = {m2['FN']} |",
        f"| **實際：乾淨** | FP = {m2['FP']} | TN = {m2['TN']} |",
        "",
        "## ③ Stage 2 最終指標（與實驗二同一張表）",
        "",
        "| 指標 | 實驗一 | 實驗二 | 說明 |",
        "|------|--------|--------|------|",
    ]
    metric_rows = [
        ("TP", "TP", "真對抗且攔截"),
        ("FP", "FP", "真乾淨卻攔截"),
        ("TN", "TN", "真乾淨且放行"),
        ("FN", "FN", "真對抗卻放行"),
        ("TPR (Recall)", "TPR", "對抗抓到的比例"),
        ("FPR", "FPR", "乾淨被誤殺比例"),
        ("Precision", "Precision", "攔截裡真是對抗的比例"),
        ("F1", "F1", "Precision 與 Recall 調和平均"),
        ("J (Youden)", "J", "TPR − FPR"),
    ]
    for label, key, note in metric_rows:
        v1 = m1[key]
        v2 = m2[key]
        if key in ("TP", "FP", "TN", "FN"):
            s1, s2 = str(v1), str(v2)
        else:
            s1, s2 = f"{v1:.4f}", f"{v2:.4f}"
        lines.append(f"| {label} | {s1} | {s2} | {note} |")
        rows_cmp.append({"metric": label, "exp1": s1, "exp2": s2, "note": note})

    lines += [
        "",
        "### 僅 Stage 1（high 攔截、無 Stage2）對照",
        "",
        "| 指標 | 實驗一 | 實驗二 |",
        "|------|--------|--------|",
    ]
    s1o, s2o = stage1_only["實驗一"], stage1_only["實驗二"]
    for k in ("TP", "FP", "TN", "FN", "TPR", "FPR", "J"):
        v1, v2 = s1o[k], s2o[k]
        if k in ("TP", "FP", "TN", "FN"):
            lines.append(f"| {k} | {v1} | {v2} |")
        else:
            lines.append(f"| {k} | {v1:.4f} | {v2:.4f} |")

    lines += [
        "",
        "## ④ 實驗一 vs 實驗二（同表比較，已寫入 CSV）",
        "",
        "檔案：`data_supplement/exp1_exp2_final_detection_comparison.csv`",
        "",
    ]

    sigma = 1.5
    if (EXP1 / "theta_det.json").is_file():
        with open(EXP1 / "theta_det.json", encoding="utf-8") as f:
            sigma = float(json.load(f).get("sigma", sigma))
    lines.extend(export_case_studies(EXP1, sigma_det=sigma))

    cmp_df = pd.DataFrame(rows_cmp)
    return lines, cmp_df


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "（無資料）"
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = []
    for _, row in df.iterrows():
        body.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join([header, sep] + body)


def write_readable_csvs(
    exp_key: str,
    stage1_sum: pd.DataFrame,
    stage2_sum: pd.DataFrame,
    clean_harm: pd.DataFrame,
    mitigation: pd.DataFrame,
    detection: pd.DataFrame,
) -> None:
    tag = "实验一" if "exp1" in exp_key else "实验二"
    stage1_wide_table(stage1_sum, exp_key).to_csv(
        OUT_DIR / f"{tag}_Stage1四维_角距后.csv", index=False, encoding="utf-8-sig"
    )
    if not stage2_sum.empty and exp_key in stage2_sum["experiment"].values:
        s2, _, _, _ = stage2_wide_table(stage2_sum, exp_key)
        s2.to_csv(OUT_DIR / f"{tag}_Stage2四维_Mid池_糊后角距.csv", index=False, encoding="utf-8-sig")
    if not clean_harm.empty:
        ch = clean_harm[clean_harm["experiment"] == exp_key].copy()
        ch.to_csv(OUT_DIR / f"{tag}_乾淨圖_Mid池_糊后改答率.csv", index=False, encoding="utf-8-sig")
    if not mitigation.empty:
        mitigation[mitigation["experiment"] == exp_key].to_csv(
            OUT_DIR / f"{tag}_緩解ASR_完整.csv", index=False, encoding="utf-8-sig"
        )
    if not detection.empty:
        det = detection[detection["experiment"] == exp_key].copy()
        det.to_csv(OUT_DIR / f"{tag}_偵測端到端.csv", index=False, encoding="utf-8-sig")


def write_markdown(
    path: Path,
    stage1_sum: pd.DataFrame,
    delta_mu: pd.DataFrame,
    stage2_sum: pd.DataFrame | None,
    clean_harm: pd.DataFrame | None,
    mitigation: pd.DataFrame | None,
    detection: pd.DataFrame | None,
    attack_asr: pd.DataFrame,
    gaps: pd.DataFrame,
) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    stage2_sum = stage2_sum if stage2_sum is not None else pd.DataFrame()
    clean_harm = clean_harm if clean_harm is not None else pd.DataFrame()
    mitigation = mitigation if mitigation is not None else pd.DataFrame()
    detection = detection if detection is not None else pd.DataFrame()

    lines = [
        "# 實驗數據一覽（易讀版）",
        "",
        "> 看不懂舊版長表時：**先看本文**，要用 Excel 再開 `data_supplement/实验一_*.csv`、`实验二_*.csv`（UTF-8 BOM）。",
        "",
        f"更新時間：{ts}",
        "",
        "## 30 秒讀懂",
        "",
        "1. **四維特徵**在程式裡都是「角距後」：`d_center, d_iqr, d_tail, d_q95`（沒有另外再存「角距前」11 條序列）。",
        "2. **Stage2 四維**只有 **中風險 Mid** 子集（不是 2000 列全有），且是在 **θ*=0.8 模糊後**重算同一套四維。",
        "3. **乾淨圖沒有攻擊 ASR**；只能報「模糊後答案是否變了」。",
        "4. **Stage2 分兩種**：**緩解 ASR**（糊了還能不能騙模型）vs **偵測**（答案翻轉，報 TPR/FPR/J）。",
        "5. **論文主線必備數據已齊**；僅「角距前 raw」與「Low/High 乾淨糊後改答」是可選。",
        "",
    ]
    for exp_key in EXP_META:
        lines.extend(
            write_experiment_section(
                exp_key, stage1_sum, delta_mu, stage2_sum, clean_harm, mitigation, detection, attack_asr
            )
        )

    appendix_lines, cmp_df = build_detection_appendix()
    lines.extend(appendix_lines)

    lines += [
        "## 還需要補什麼？（必備 vs 可選）",
        "",
        "| 類別 | 內容 | 要不要現在跑 |",
        "|------|------|----------------|",
        "| ✅ 必備 | Stage1/2 角距後四維、緩解 ASR、偵測、Mid 乾淨改答 | **不用**，已在各實驗目錄 |",
        "| ⭕ 可選 | 每張圖 11 探針「角距前」序列 | 要改程式 + GPU 重跑 Stage1 |",
        "| ⭕ 可選 | Low/High 乾淨圖糊後改答 | 額外 VQA 實驗 |",
        "",
        "技術細節與英文長表：`data_supplement/stage1_4d_summary.csv` 等（給程式用）。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return cmp_df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    stage1_parts = []
    delta_parts = []
    stage2_parts = []
    clean_parts = []
    mit_parts = []
    det_parts = []
    asr_parts = []

    for root, name in [(EXP1, "exp1_steps100_alpha0.002_run1000"), (EXP2, "exp2_success1000")]:
        s1_path = root / "tcvad_stage1_4_academic_dataset.csv"
        if not s1_path.is_file():
            print(f"[skip] missing {s1_path}")
            continue
        s1 = pd.read_csv(s1_path)
        s1 = s1[s1["method_type"] == "rotate_5"].copy()
        stage1_parts.append(stage1_feature_summary(s1, name))
        delta_parts.append(stage1_delta_mu(s1, name))

        s2 = stage2_feature_summary(root / "stage2_gaussian_results.csv", name)
        if s2 is not None:
            stage2_parts.append(s2)

        ch = clean_answer_change_by_sigma(root / "sss_per_theta_raw.csv", name)
        if ch is not None:
            clean_parts.append(ch)

        mit = load_mitigation_rows(root / "mitigation_summary.csv", name)
        if mit is not None:
            mit_parts.append(mit)

        det = load_detection_rows(root / "detector_comparison.csv", name)
        if det is not None:
            det_parts.append(det)

        asr_parts.append(load_attack_asr(root, name))

    stage1_sum = pd.concat(stage1_parts, ignore_index=True)
    delta_mu = pd.concat(delta_parts, ignore_index=True)
    stage2_sum = pd.concat(stage2_parts, ignore_index=True) if stage2_parts else pd.DataFrame()
    clean_harm = pd.concat(clean_parts, ignore_index=True) if clean_parts else pd.DataFrame()
    mitigation = pd.concat(mit_parts, ignore_index=True) if mit_parts else pd.DataFrame()
    detection = pd.concat(det_parts, ignore_index=True) if det_parts else pd.DataFrame()
    attack_asr = pd.concat(asr_parts, ignore_index=True)
    gaps = gap_checklist()

    stage1_sum.to_csv(OUT_DIR / "stage1_4d_summary.csv", index=False, encoding="utf-8")
    delta_mu.to_csv(OUT_DIR / "stage1_delta_mu_attack_minus_clean.csv", index=False, encoding="utf-8")
    if not stage2_sum.empty:
        stage2_sum.to_csv(OUT_DIR / "stage2_4d_summary.csv", index=False, encoding="utf-8")
    if not clean_harm.empty:
        clean_harm.to_csv(OUT_DIR / "clean_mid_answer_change_by_sigma.csv", index=False, encoding="utf-8")
    if not mitigation.empty:
        mitigation.to_csv(OUT_DIR / "mitigation_asr_summary.csv", index=False, encoding="utf-8")
    if not detection.empty:
        detection.to_csv(OUT_DIR / "stage2_detection_end2end.csv", index=False, encoding="utf-8")
    attack_asr.to_csv(OUT_DIR / "attack_asr_overview.csv", index=False, encoding="utf-8")
    gaps.to_csv(OUT_DIR / "data_gap_checklist.csv", index=False, encoding="utf-8")

    cmp_df = write_markdown(
        VQA_ROOT / "EXPERIMENT_DATA_SUPPLEMENT.md",
        stage1_sum,
        delta_mu,
        stage2_sum,
        clean_harm,
        mitigation,
        detection,
        attack_asr,
        gaps,
    )
    if cmp_df is not None and not cmp_df.empty:
        cmp_df.to_csv(OUT_DIR / "exp1_exp2_final_detection_comparison.csv", index=False, encoding="utf-8-sig")
    for exp_key in EXP_META:
        write_readable_csvs(exp_key, stage1_sum, stage2_sum, clean_harm, mitigation, detection)
    print(f"Wrote {OUT_DIR} and {VQA_ROOT / 'EXPERIMENT_DATA_SUPPLEMENT.md'}")


if __name__ == "__main__":
    main()
