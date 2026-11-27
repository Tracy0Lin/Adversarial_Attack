"""Prepare checklist analytics: sample flow, pipeline verification, scenario comparison.

Reads existing experiment CSVs only (no GPU). Writes NEW files under:
  BLIP_attack/output/VQA/data_supplement/checklist_prepared/

Does NOT overwrite data_supplement/*, EXPERIMENT_DATA_SUPPLEMENT.md, or experiment dirs.

Run from repo root:
  python BLIP_attack/output/VQA/build_checklist_analytics.py
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

VQA_ROOT = Path(__file__).resolve().parent
OUT_DIR = VQA_ROOT / "data_supplement" / "checklist_prepared"

EXP1_ROOT = VQA_ROOT / "steps100_alpha0.002_run1000"
EXP2_ROOT = VQA_ROOT / "exp2_success1000"
EXP1_MANIFEST = VQA_ROOT / "exp1_mixed1000" / "dataset_manifest.csv"
EXP2_MANIFEST = EXP2_ROOT / "dataset_manifest.csv"

EXP_META = {
    "exp1": {
        "name": "實驗一（混合流量／自然分布）",
        "root": EXP1_ROOT,
        "manifest": EXP1_MANIFEST,
        "scenario": "natural_mixed",
    },
    "exp2": {
        "name": "實驗二（僅攻擊成功／壓力測試）",
        "root": EXP2_ROOT,
        "manifest": EXP2_MANIFEST,
        "scenario": "stress_all_success",
    },
}


def load_stage1(root: Path) -> pd.DataFrame:
    df = pd.read_csv(root / "tcvad_stage1_4_academic_dataset.csv")
    df["image_id"] = df["image_id"].astype(str)
    return df[df["method_type"] == "rotate_5"].copy()


def load_manifest(path: Path) -> pd.DataFrame:
    m = pd.read_csv(path)
    m["image_id"] = m["question_id"].astype(str)
    m["attack_success"] = m["attack_success"].astype(bool)
    m["ori_correct"] = m["original_prediction"].astype(str) == m["ground_truth"].astype(str)
    m["adv_correct"] = m["adversarial_prediction"].astype(str) == m["ground_truth"].astype(str)
    return m


def load_theta_star(root: Path) -> float:
    with open(root / "sss_best_theta.json", encoding="utf-8") as f:
        return float(json.load(f)["sigma"])


def load_flip_threshold(root: Path) -> int:
    p = root / "theta_det.json"
    if not p.is_file():
        return 1
    with open(p, encoding="utf-8") as f:
        return int(json.load(f).get("multi_sigma", {}).get("threshold", 1))


def flip_count_flags(root: Path, threshold: int) -> dict[tuple[str, int], bool]:
    raw = pd.read_csv(root / "sss_per_theta_raw.csv")
    raw["image_id"] = raw["image_id"].astype(str)
    raw["flip"] = (raw["pred_denoised"].astype(str) != raw["pred_ref"].astype(str)).astype(int)
    per = raw.groupby(["image_id", "label"], as_index=False)["flip"].sum()
    return {
        (str(r.image_id), int(r.label)): int(r.flip) >= threshold for r in per.itertuples(index=False)
    }


def flip_flags_sigma(root: Path, sigma: float) -> dict[tuple[str, int], bool]:
    raw = pd.read_csv(root / "sss_per_theta_raw.csv")
    raw["image_id"] = raw["image_id"].astype(str)
    sub = raw[raw["sigma"] == sigma]
    return {
        (str(r.image_id), int(r.label)): str(r.pred_denoised) != str(r.pred_ref)
        for r in sub.itertuples(index=False)
    }


def svm_stage2_flags(root: Path) -> dict[tuple[str, int], bool]:
    p = root / "stage2_gaussian_results.csv"
    if not p.is_file():
        return {}
    s2 = pd.read_csv(p)
    s2["image_id"] = s2["image_id"].astype(str)
    return {(str(r.image_id), int(r.label)): r.bin_stage2 == "high" for r in s2.itertuples(index=False)}


def end_to_end_flagged(stage1: pd.DataFrame, mid_flag: dict[tuple[str, int], bool]) -> np.ndarray:
    flagged = []
    for row in stage1.itertuples(index=False):
        if row.risk_bin == "high":
            flagged.append(True)
        elif row.risk_bin == "low":
            flagged.append(False)
        else:
            flagged.append(bool(mid_flag.get((str(row.image_id), int(row.label)), False)))
    return np.array(flagged, dtype=bool)


def rates(y_true: np.ndarray, flagged: np.ndarray) -> dict:
    y_true = y_true.astype(bool)
    flagged = flagged.astype(bool)
    tp = int((flagged & y_true).sum())
    fn = int((~flagged & y_true).sum())
    fp = int((flagged & ~y_true).sum())
    tn = int((~flagged & ~y_true).sum())
    tpr = tp / (tp + fn) if (tp + fn) else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "TPR": tpr,
        "FPR": fpr,
        "intercept_attack_n": tp,
        "intercept_attack_rate": tpr,
        "false_block_clean_n": fp,
        "false_block_clean_rate": fpr,
        "youden_j": tpr - fpr,
    }


def mitigation_routed_asr(root: Path, manifest: pd.DataFrame, sigma: float) -> pd.Series:
    """Per attack image: still fooled after routed mitigation @ sigma."""
    stage1 = load_stage1(root)
    s1_atk = stage1[stage1["label"] == 1].merge(
        manifest[["image_id", "original_prediction", "attack_success"]],
        on="image_id",
        how="inner",
    )
    raw = pd.read_csv(root / "sss_per_theta_raw.csv")
    raw["image_id"] = raw["image_id"].astype(str)
    sub = raw[(raw["sigma"] == sigma) & (raw["label"] == 1) & (raw["kind"] == "attack")]
    denoised = sub.set_index("image_id")["pred_denoised"]
    pred_ref = sub.set_index("image_id")["pred_ref"]

    still = []
    for r in s1_atk.itertuples(index=False):
        ori = str(r.original_prediction)
        if r.risk_bin == "high":
            still.append(False)
        elif r.risk_bin == "low":
            still.append(bool(r.attack_success))
        else:
            qid = r.image_id
            if qid in denoised.index:
                still.append(str(denoised.loc[qid]) != ori)
            elif qid in pred_ref.index:
                still.append(str(pred_ref.loc[qid]) != ori)
            else:
                still.append(bool(r.attack_success))
    s1_atk = s1_atk.copy()
    s1_atk["residual_fool"] = still
    return s1_atk


def build_routing_breakdown(stage1: pd.DataFrame, manifest: pd.DataFrame, kind: str) -> pd.DataFrame:
    """kind: attack | clean"""
    label = 1 if kind == "attack" else 0
    sub = stage1[stage1["label"] == label].copy()
    rows = []
    for bin_name in ("low", "mid", "high"):
        g = sub[sub["risk_bin"] == bin_name]
        n = len(g)
        if kind == "attack":
            m = manifest.set_index("image_id")
            succ = g["image_id"].map(lambda q: bool(m.loc[q, "attack_success"]) if q in m.index else False)
            fooled_n = int(succ.sum())
            gt_wrong_adv = g["image_id"].map(
                lambda q: not bool(m.loc[q, "adv_correct"]) if q in m.index else False
            )
            rows.append(
                {
                    "image_kind": kind,
                    "stage": "S1_TCVAD_routing",
                    "risk_bin": bin_name,
                    "n": n,
                    "attack_success_n": fooled_n,
                    "attack_success_rate": fooled_n / n if n else float("nan"),
                    "adv_wrong_vs_gt_n": int(gt_wrong_adv.sum()),
                    "adv_wrong_vs_gt_rate": float(gt_wrong_adv.mean()) if n else float("nan"),
                    "ori_correct_n": np.nan,
                    "ori_correct_rate": np.nan,
                }
            )
        else:
            m = manifest.set_index("image_id")
            ori_ok = g["image_id"].map(lambda q: bool(m.loc[q, "ori_correct"]) if q in m.index else False)
            rows.append(
                {
                    "image_kind": kind,
                    "stage": "S1_TCVAD_routing",
                    "risk_bin": bin_name,
                    "n": n,
                    "attack_success_n": np.nan,
                    "attack_success_rate": np.nan,
                    "adv_wrong_vs_gt_n": np.nan,
                    "adv_wrong_vs_gt_rate": np.nan,
                    "ori_correct_n": int(ori_ok.sum()),
                    "ori_correct_rate": float(ori_ok.mean()) if n else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def build_stage_summary_rows(
    exp_key: str, stage1: pd.DataFrame, manifest: pd.DataFrame, root: Path
) -> list[dict]:
    sigma_mit = load_theta_star(root)
    thr = load_flip_threshold(root)
    flip_multi = flip_count_flags(root, thr)

    atk_m = manifest
    n_atk = len(atk_m)
    n_clean = n_atk

    fooled_before = int(atk_m["attack_success"].sum())
    asr0 = fooled_before / n_atk

    # Clean baseline
    clean_ori_ok = int(manifest["ori_correct"].sum())

    rows = [
        {
            "experiment": exp_key,
            "image_kind": "attack",
            "stage_id": "S0_baseline_adv_vqa",
            "stage_label": "基線：對抗圖 VQA（無防禦）",
            "n": n_atk,
            "metric_primary": "ASR",
            "success_or_fooled_n": fooled_before,
            "rate": asr0,
            "retention_vs_prev": 1.0,
            "notes": "success ≡ VQA(adv)≠VQA(ori)",
        },
        {
            "experiment": exp_key,
            "image_kind": "clean",
            "stage_id": "S0_baseline_ori_vqa",
            "stage_label": "基線：原圖 VQA（無防禦）",
            "n": n_clean,
            "metric_primary": "原圖答對率(vs GT)",
            "success_or_fooled_n": clean_ori_ok,
            "rate": clean_ori_ok / n_clean,
            "retention_vs_prev": 1.0,
            "notes": "非 ASR",
        },
    ]

    mit = mitigation_routed_asr(root, manifest, sigma_mit)
    residual_n = int(mit["residual_fool"].sum())
    rows.append(
        {
            "experiment": exp_key,
            "image_kind": "attack",
            "stage_id": "S3_mitigation_routed",
            "stage_label": f"緩解管線：Low 放行 / Mid 高斯 σ={sigma_mit:g} / High 阻擋",
            "n": n_atk,
            "metric_primary": "殘存 ASR",
            "success_or_fooled_n": residual_n,
            "rate": residual_n / n_atk,
            "retention_vs_prev": (residual_n / fooled_before) if fooled_before else float("nan"),
            "notes": "θ* 緩解；High 視為攻擊失效",
        }
    )

    s1_atk = stage1[stage1["label"] == 1].merge(
        manifest[["image_id", "attack_success"]], on="image_id", how="inner"
    )
    s1_block = s1_atk["risk_bin"] == "high"
    blocked_s1 = int(s1_block.sum())
    fooled_in_high = int(s1_atk.loc[s1_block, "attack_success"].sum())
    residual_after_high_only = fooled_before - fooled_in_high
    rows.append(
        {
            "experiment": exp_key,
            "image_kind": "attack",
            "stage_id": "S1_high_block_only",
            "stage_label": "僅 Stage1：High 攔截",
            "n": n_atk,
            "metric_primary": "攔截攻擊數(High)",
            "success_or_fooled_n": fooled_in_high,
            "rate": fooled_in_high / n_atk,
            "retention_vs_prev": residual_after_high_only / fooled_before if fooled_before else float("nan"),
            "notes": f"High 路由 n={blocked_s1}；其中攻擊成功 {fooled_in_high}；殘存可騙 {residual_after_high_only}",
        }
    )

    flagged_det = end_to_end_flagged(stage1, flip_multi)
    y = stage1["label"].values == 1
    det = rates(y, flagged_det)
    fn = det["FN"]
    s1_atk_flag = s1_atk.copy()
    s1_atk_flag["blocked"] = end_to_end_flagged(s1_atk_flag, flip_multi)
    fooled_fn = int(s1_atk_flag.loc[~s1_atk_flag["blocked"], "attack_success"].sum())
    rows.append(
        {
            "experiment": exp_key,
            "image_kind": "attack",
            "stage_id": "S2_detection_e2e",
            "stage_label": f"偵測管線 E2E：Stage1 + Mid flip_count≥{thr}",
            "n": n_atk,
            "metric_primary": "攻擊攔截率(TPR)",
            "success_or_fooled_n": det["intercept_attack_n"],
            "rate": det["intercept_attack_rate"],
            "retention_vs_prev": fooled_fn / fooled_before if fooled_before else float("nan"),
            "notes": f"FN={fn}；FN中仍攻擊成功={fooled_fn}（殘存ASR={fooled_fn/n_atk:.3f}）；FPR見 clean 列",
        }
    )

    # Clean: false block under detection
    fp = det["FP"]
    raw = pd.read_csv(root / "sss_per_theta_raw.csv")
    raw["image_id"] = raw["image_id"].astype(str)
    ch = raw[(raw["sigma"] == sigma_mit) & (raw["label"] == 0) & (raw["kind"] == "clean")]
    harm_n = int((ch["pred_denoised"].astype(str) != ch["pred_ref"].astype(str)).sum())
    rows.append(
        {
            "experiment": exp_key,
            "image_kind": "clean",
            "stage_id": "S2_detection_e2e",
            "stage_label": f"偵測管線 E2E（flip_count≥{thr}）",
            "n": n_clean,
            "metric_primary": "誤攔(FPR)",
            "success_or_fooled_n": fp,
            "rate": fp / n_clean,
            "retention_vs_prev": float("nan"),
            "notes": f"Mid 池糊後改答 n={harm_n}（σ={sigma_mit:g}，副作用）",
        }
    )

    return rows


def recompute_detector_comparison(root: Path) -> pd.DataFrame:
    raw = pd.read_csv(root / "sss_per_theta_raw.csv")
    raw["image_id"] = raw["image_id"].astype(str)
    stage1 = load_stage1(root)
    thr = load_flip_threshold(root)

    per = raw.copy()
    per["flip"] = (per["pred_denoised"].astype(str) != per["pred_ref"].astype(str)).astype(int)
    counts = per.groupby(["image_id", "label"], as_index=False)["flip"].sum()
    flag_multi = {
        (str(r.image_id), int(r.label)): int(r.flip) >= thr for r in counts.itertuples(index=False)
    }

    comparisons = [rates(stage1["label"].values == 1, end_to_end_flagged(stage1, {}))]
    comp_rows = [
        {
            "pipeline": "stage1 only (high bin)",
            "n": len(stage1),
            **{k: comparisons[0][k] for k in ("TPR", "FPR", "youden_j", "TP", "FP", "TN", "FN")},
        }
    ]

    svm = svm_stage2_flags(root)
    if svm:
        r = rates(stage1["label"].values == 1, end_to_end_flagged(stage1, svm))
        comp_rows.append({"pipeline": "stage1 + caption-SVM stage2", "n": len(stage1), **r})

    for sigma, name in [
        (1.5, "stage1 + answer-flip (sigma=1.5)"),
    ]:
        fl = flip_flags_sigma(root, sigma)
        r = rates(stage1["label"].values == 1, end_to_end_flagged(stage1, fl))
        comp_rows.append({"pipeline": name, "n": len(stage1), **r})

    sigma_mit = load_theta_star(root)
    if sigma_mit != 1.5:
        fl = flip_flags_sigma(root, sigma_mit)
        r = rates(stage1["label"].values == 1, end_to_end_flagged(stage1, fl))
        comp_rows.append(
            {
                "pipeline": f"stage1 + answer-flip (sigma={sigma_mit}, theta*)",
                "n": len(stage1),
                **r,
            }
        )

    r = rates(stage1["label"].values == 1, end_to_end_flagged(stage1, flag_multi))
    comp_rows.append({"pipeline": f"stage1 + flip_count >= {thr}", "n": len(stage1), **r})

    return pd.DataFrame(comp_rows)


def mid_only_filter_metrics(root: Path) -> pd.DataFrame:
    """Stage-2 filters evaluated on Mid pool only vs routed E2E (1000+1000)."""
    stage1 = load_stage1(root)
    mid = stage1[stage1["risk_bin"] == "mid"]
    y = mid["label"].values == 1
    thr = load_flip_threshold(root)

    rows = []
    for name, mid_flag in [
        ("flip_count>=thr (Mid only)", flip_count_flags(root, thr)),
        ("answer_flip sigma=1.5 (Mid only)", flip_flags_sigma(root, 1.5)),
        ("caption-SVM high (Mid only)", svm_stage2_flags(root)),
    ]:
        flagged = np.array(
            [bool(mid_flag.get((str(r.image_id), int(r.label)), False)) for r in mid.itertuples(index=False)]
        )
        r = rates(y, flagged)
        rows.append(
            {
                "scope": "mid_pool_only",
                "method": name,
                "n_eval": len(mid),
                "n_attack": int(y.sum()),
                "n_clean": int((~y).sum()),
                **{k: r[k] for k in ("TPR", "FPR", "youden_j", "TP", "FP", "TN", "FN")},
            }
        )

    flip_multi = flip_count_flags(root, thr)
    flagged_e2e = end_to_end_flagged(stage1, flip_multi)
    r_e2e = rates(stage1["label"].values == 1, flagged_e2e)
    rows.append(
        {
            "scope": "e2e_routed_2000",
            "method": f"flip_count>=thr (Stage1 route + Mid)",
            "n_eval": len(stage1),
            "n_attack": 1000,
            "n_clean": 1000,
            **{k: r_e2e[k] for k in ("TPR", "FPR", "youden_j", "TP", "FP", "TN", "FN")},
        }
    )
    return pd.DataFrame(rows)


def mitigation_denominator_comparison(root: Path, manifest: pd.DataFrame) -> pd.DataFrame:
    """Contrast legacy 462 success-only vs all-1000 attack cohort @ mid denoise only."""
    sigma = load_theta_star(root)
    raw = pd.read_csv(root / "sss_per_theta_raw.csv")
    raw["image_id"] = raw["image_id"].astype(str)
    sub = raw[(raw["sigma"] == sigma) & (raw["label"] == 1) & (raw["kind"] == "attack")].copy()
    sub = sub.merge(manifest[["image_id", "original_prediction", "attack_success"]], on="image_id")
    sub["residual"] = sub["pred_denoised"].astype(str) != sub["original_prediction"].astype(str)

    cohorts = [
        ("mid_pool_attack_n391", sub),
        ("attack_success_only_in_mid", sub[sub["attack_success"]]),
        ("all_1000_attack_mid_denoise_if_in_mid_else_unchanged", None),
    ]
    rows = []
    for label, frame in cohorts[:2]:
        before = frame["attack_success"] if label != "mid_pool_attack_n391" else frame["attack_success"]
        if label == "mid_pool_attack_n391":
            before = frame["attack_success"]
        rows.append(
            {
                "cohort": label,
                "n": len(frame),
                "asr_before": float(before.mean()),
                "asr_after_mid_blur_only": float(frame["residual"].mean()),
                "retention_rate": float(frame["residual"].mean() / before.mean()) if before.mean() else float("nan"),
            }
        )

    mit = pd.read_csv(root / "mitigation_summary.csv")
    for _, r in mit[mit["cohort"].str.contains("all_1000|462|1000", regex=True)].iterrows():
        rows.append(
            {
                "cohort": r["cohort"],
                "n": int(r["n"]),
                "asr_before": float(r["asr_before"]) if pd.notna(r["asr_before"]) else float("nan"),
                "asr_after_mid_blur_only": float(r["asr_after"]) if pd.notna(r["asr_after"]) else float("nan"),
                "retention_rate": (
                    float(r["asr_after"] / r["asr_before"]) if pd.notna(r["asr_before"]) and r["asr_before"] else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def scenario_decomposition(exp1_root: Path, exp2_root: Path) -> pd.DataFrame:
    s1_1 = load_stage1(exp1_root)
    s1_2 = load_stage1(exp2_root)
    thr1 = load_flip_threshold(exp1_root)
    thr2 = load_flip_threshold(exp2_root)
    fl1 = flip_count_flags(exp1_root, thr1)
    fl2 = flip_count_flags(exp2_root, thr2)

    def bundle(tag, root, s1, fl, thr):
        y = s1["label"].values == 1
        s1o = rates(y, end_to_end_flagged(s1, {}))
        e2e = rates(y, end_to_end_flagged(s1, fl))
        mit = pd.read_csv(root / "mitigation_summary.csv")
        routed = mit[mit["cohort"].str.startswith("routed_low_pass")].iloc[0]
        atk = s1[s1["label"] == 1]
        vc = atk["risk_bin"].value_counts()
        return {
            "experiment": tag,
            "attack_high_n": int(vc.get("high", 0)),
            "attack_mid_n": int(vc.get("mid", 0)),
            "attack_low_n": int(vc.get("low", 0)),
            "stage1_only_TPR": s1o["TPR"],
            "stage1_only_FPR": s1o["FPR"],
            "e2e_flip_TPR": e2e["TPR"],
            "e2e_flip_FPR": e2e["FPR"],
            "delta_TPR_stage2": e2e["TPR"] - s1o["TPR"],
            "delta_FPR_stage2": e2e["FPR"] - s1o["FPR"],
            "residual_ASR_routed_mitigation": float(routed["asr_after"]),
            "mitigation_ASR_drop_pp": float(routed["asr_drop_pp"]),
            "flip_threshold": thr,
        }

    r1 = bundle("exp1_natural", exp1_root, s1_1, fl1, thr1)
    r2 = bundle("exp2_stress", exp2_root, s1_2, fl2, thr2)
    df = pd.DataFrame([r1, r2])
    df["high_bin_share_delta_exp2_minus_exp1"] = (r2["attack_high_n"] - r1["attack_high_n"]) / 1000
    df["stage1_TPR_gap_exp2_minus_exp1"] = r2["stage1_only_TPR"] - r1["stage1_only_TPR"]
    df["e2e_TPR_gap_exp2_minus_exp1"] = r2["e2e_flip_TPR"] - r1["e2e_flip_TPR"]
    return df


def write_markdown_report(
    path: Path,
    flow_summary: pd.DataFrame,
    routing_all: pd.DataFrame,
    verify: pd.DataFrame,
    scenario: pd.DataFrame,
    mid_cmp: dict[str, pd.DataFrame],
) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# 清單用數據包（不覆蓋原版）",
        "",
        f"產生時間：{ts}  ",
        "輸出目錄：`data_supplement/checklist_prepared/`",
        "",
        "## ① 樣本流向表",
        "",
        "- `exp1_sample_flow_summary.csv` / `exp2_sample_flow_summary.csv`：逐階段 ASR／殘存率／維持率",
        "- `exp1_routing_breakdown.csv` / `exp2_routing_breakdown.csv`：S1 各風險段路由 + 正誤答",
        "- `sample_flow_routing_combined.csv`：兩實驗 Attack/Clean 路由長表",
        "",
        "## ② 完整流程串接驗證",
        "",
        "- `pipeline_recompute_vs_saved.csv`：由 CSV 重算 E2E 偵測，與原 `detector_comparison.csv` 對照",
        "- `filter_method_mid_vs_e2e.csv`：濾波方法 **Mid 池** vs **路由後 2000 列**",
        "- `mitigation_denominator_exp1.csv`：**1000 張** vs 舊 **462 成功子集** 口徑對照",
        "",
        "## ③ 情境對照（自然 vs 壓力）",
        "",
        "- `scenario_natural_vs_stress.csv`：攔截率、殘存 ASR、Stage 貢獻分解",
        "",
        "### 重點數字（偵測 flip_count≥1）",
        "",
    ]
    for _, r in scenario.iterrows():
        lines.append(
            f"- **{r['experiment']}**：Stage1 TPR={r['stage1_only_TPR']:.1%} → E2E TPR={r['e2e_flip_TPR']:.1%} "
            f"(+{r['delta_TPR_stage2']:.1%})；FPR {r['stage1_only_FPR']:.1%}→{r['e2e_flip_FPR']:.1%}；"
            f"路由緩解殘存 ASR={r['residual_ASR_routed_mitigation']:.1%}"
        )
    lines += [
        "",
        "### 差異主要來自哪一階段？",
        "",
        "見 `scenario_stage_attribution.csv`（High 路由差、Stage1 TPR 差、Stage2 增量差）。",
        "",
        "## 限制（需 GPU 才補）",
        "",
        "Mid 探針 `sss_per_theta_raw.csv` 僅含 **中風險列**；對 Low/High 的攻擊圖 **沒有** 糊後 VQA。",
        "因此「濾波方法獨立測試」若要求 **1000 張攻擊圖皆跑 flip**，需另跑 VQA+blur；",
        "本包已用 **1000 張口徑的緩解 ASR**（`all_1000_attack_*` / `routed_*`）與 **E2E 路由偵測** 補齊可算部分。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_flow_matrix(exp_key: str, flow_df: pd.DataFrame, routing: pd.DataFrame) -> pd.DataFrame:
    """Wide table for reports: attack + clean side by side."""
    atk = flow_df[flow_df["image_kind"] == "attack"]
    cl = flow_df[flow_df["image_kind"] == "clean"]
    r_atk = routing[(routing["experiment"] == exp_key) & (routing["image_kind"] == "attack")]
    r_cl = routing[(routing["experiment"] == exp_key) & (routing["image_kind"] == "clean")]

    def pick(df, stage_id, col="rate"):
        sub = df[df["stage_id"] == stage_id]
        return float(sub[col].iloc[0]) if len(sub) else float("nan")

    rows = [
        {
            "experiment": exp_key,
            "cohort": "attack_1000",
            "S0_ASR_or_baseline": pick(atk, "S0_baseline_adv_vqa"),
            "S0_success_n": int(atk[atk["stage_id"] == "S0_baseline_adv_vqa"]["success_or_fooled_n"].iloc[0]),
            "S1_route_low_n": int(r_atk[r_atk["risk_bin"] == "low"]["n"].iloc[0]),
            "S1_route_mid_n": int(r_atk[r_atk["risk_bin"] == "mid"]["n"].iloc[0]),
            "S1_route_high_n": int(r_atk[r_atk["risk_bin"] == "high"]["n"].iloc[0]),
            "S1_ASR_in_low": float(r_atk[r_atk["risk_bin"] == "low"]["attack_success_rate"].iloc[0]),
            "S1_ASR_in_mid": float(r_atk[r_atk["risk_bin"] == "mid"]["attack_success_rate"].iloc[0]),
            "S1_ASR_in_high": float(r_atk[r_atk["risk_bin"] == "high"]["attack_success_rate"].iloc[0]),
            "S3_routed_mitigation_ASR": pick(atk, "S3_mitigation_routed"),
            "S3_retention_vs_S0": pick(atk, "S3_mitigation_routed", "retention_vs_prev"),
            "S2_detection_intercept_rate": pick(atk, "S2_detection_e2e"),
            "S2_residual_ASR_after_detection": float(
                atk[atk["stage_id"] == "S2_detection_e2e"]["retention_vs_prev"].iloc[0]
            )
            * pick(atk, "S0_baseline_adv_vqa"),
        },
        {
            "experiment": exp_key,
            "cohort": "clean_1000",
            "S0_ASR_or_baseline": pick(cl, "S0_baseline_ori_vqa"),
            "S0_success_n": int(cl[cl["stage_id"] == "S0_baseline_ori_vqa"]["success_or_fooled_n"].iloc[0]),
            "S1_route_low_n": int(r_cl[r_cl["risk_bin"] == "low"]["n"].iloc[0]),
            "S1_route_mid_n": int(r_cl[r_cl["risk_bin"] == "mid"]["n"].iloc[0]),
            "S1_route_high_n": int(r_cl[r_cl["risk_bin"] == "high"]["n"].iloc[0]),
            "S1_ASR_in_low": float("nan"),
            "S1_ASR_in_mid": float("nan"),
            "S1_ASR_in_high": float("nan"),
            "S3_routed_mitigation_ASR": float("nan"),
            "S3_retention_vs_S0": float("nan"),
            "S2_detection_intercept_rate": float("nan"),
            "S2_residual_ASR_after_detection": pick(cl, "S2_detection_e2e"),
        },
    ]
    return pd.DataFrame(rows)


def flow_matrix_to_markdown(df: pd.DataFrame, title: str) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = []
    for _, row in df.iterrows():
        body.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join([f"## {title}", "", header, sep] + body)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    flow_parts = []
    routing_parts = []
    verify_parts = []
    mid_parts = {}
    mit_den_parts = []

    for key, meta in EXP_META.items():
        root = meta["root"]
        manifest = load_manifest(meta["manifest"])
        stage1 = load_stage1(root)

        summary_rows = build_stage_summary_rows(key, stage1, manifest, root)
        flow_df = pd.DataFrame(summary_rows)
        flow_df.to_csv(OUT_DIR / f"{key}_sample_flow_summary.csv", index=False, encoding="utf-8-sig")
        flow_parts.append(flow_df)

        r_atk = build_routing_breakdown(stage1, manifest, "attack")
        r_clean = build_routing_breakdown(stage1, manifest, "clean")
        r_atk["experiment"] = key
        r_clean["experiment"] = key
        routing = pd.concat([r_atk, r_clean], ignore_index=True)
        routing.to_csv(OUT_DIR / f"{key}_routing_breakdown.csv", index=False, encoding="utf-8-sig")
        routing_parts.append(routing)

        matrix = build_flow_matrix(key, flow_df, routing)
        matrix.to_csv(OUT_DIR / f"{key}_sample_flow_matrix.csv", index=False, encoding="utf-8-sig")
        (OUT_DIR / f"{key}_sample_flow_table.md").write_text(
            flow_matrix_to_markdown(matrix, f"{meta['name']} 樣本流向矩陣") + "\n",
            encoding="utf-8",
        )

        recomputed = recompute_detector_comparison(root)
        saved = pd.read_csv(root / "detector_comparison.csv")
        merged = recomputed.merge(saved, on="pipeline", suffixes=("_recomputed", "_saved"), how="outer")
        merged["experiment"] = key
        merged["j_delta"] = merged["youden_j_recomputed"] - merged["youden_j_saved"]
        merged.to_csv(OUT_DIR / f"{key}_pipeline_recompute_vs_saved.csv", index=False, encoding="utf-8-sig")
        verify_parts.append(merged)

        mid_df = mid_only_filter_metrics(root)
        mid_df["experiment"] = key
        mid_df.to_csv(OUT_DIR / f"{key}_filter_method_mid_vs_e2e.csv", index=False, encoding="utf-8-sig")
        mid_parts[key] = mid_df

        if key == "exp1":
            mit_den = mitigation_denominator_comparison(root, manifest)
            mit_den.to_csv(OUT_DIR / f"{key}_mitigation_denominator_comparison.csv", index=False, encoding="utf-8-sig")
            mit_den_parts.append(mit_den)

    flow_all = pd.concat(flow_parts, ignore_index=True)
    flow_all.to_csv(OUT_DIR / "sample_flow_summary_combined.csv", index=False, encoding="utf-8-sig")

    routing_all = pd.concat(routing_parts, ignore_index=True)
    routing_all.to_csv(OUT_DIR / "sample_flow_routing_combined.csv", index=False, encoding="utf-8-sig")

    verify_all = pd.concat(verify_parts, ignore_index=True)
    verify_all.to_csv(OUT_DIR / "pipeline_recompute_vs_saved_combined.csv", index=False, encoding="utf-8-sig")

    scenario = scenario_decomposition(EXP1_ROOT, EXP2_ROOT)
    scenario.to_csv(OUT_DIR / "scenario_natural_vs_stress.csv", index=False, encoding="utf-8-sig")

    s1 = scenario.iloc[0]
    s2 = scenario.iloc[1]
    attrib = pd.DataFrame(
        [
            {
                "factor": "Stage1_high_routing_on_attacks",
                "exp1": s1["attack_high_n"],
                "exp2": s2["attack_high_n"],
                "delta_exp2_minus_exp1": s2["attack_high_n"] - s1["attack_high_n"],
                "interpretation": "High 直接攔截；實驗二成功攻擊更常進 High",
            },
            {
                "factor": "stage1_only_TPR",
                "exp1": s1["stage1_only_TPR"],
                "exp2": s2["stage1_only_TPR"],
                "delta_exp2_minus_exp1": s2["stage1_only_TPR"] - s1["stage1_only_TPR"],
                "interpretation": "Stage1 單獨貢獻的攻擊攔截率",
            },
            {
                "factor": "stage2_flip_increment_TPR",
                "exp1": s1["delta_TPR_stage2"],
                "exp2": s2["delta_TPR_stage2"],
                "delta_exp2_minus_exp1": s2["delta_TPR_stage2"] - s1["delta_TPR_stage2"],
                "interpretation": "Mid 翻轉探針在路由後追加的 TPR",
            },
            {
                "factor": "residual_ASR_after_routed_mitigation",
                "exp1": s1["residual_ASR_routed_mitigation"],
                "exp2": s2["residual_ASR_routed_mitigation"],
                "delta_exp2_minus_exp1": s2["residual_ASR_routed_mitigation"] - s1["residual_ASR_routed_mitigation"],
                "interpretation": "緩解管線殘存 ASR（非偵測）",
            },
            {
                "factor": "clean_false_block_FPR_e2e",
                "exp1": s1["e2e_flip_FPR"],
                "exp2": s2["e2e_flip_FPR"],
                "delta_exp2_minus_exp1": s2["e2e_flip_FPR"] - s1["e2e_flip_FPR"],
                "interpretation": "乾淨圖誤攔率",
            },
        ]
    )
    attrib.to_csv(OUT_DIR / "scenario_stage_attribution.csv", index=False, encoding="utf-8-sig")

    write_markdown_report(
        OUT_DIR / "README_CHECKLIST_DATA.md",
        flow_all,
        routing_all,
        verify_all,
        scenario,
        mid_parts,
    )
    print(f"Wrote checklist data to {OUT_DIR}")


if __name__ == "__main__":
    main()
