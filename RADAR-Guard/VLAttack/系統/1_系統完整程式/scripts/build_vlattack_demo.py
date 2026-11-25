"""Build a small VLAttack demo pack (images + catalog) from VLAttack_zip."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

VLATTACK_ROOT = Path(r"C:\Users\tracy_aboikg6\OneDrive\Desktop\program\VLAttack_zip")
EXP = VLATTACK_ROOT / "BLIP_attack" / "output" / "VQA" / "exp1_mixed1000"
SOURCE_RUN = VLATTACK_ROOT / "BLIP_attack" / "output" / "VQA" / "steps100_alpha0.002_run1000"
WEB_ROOT = Path(__file__).resolve().parents[1]
OUT_STATIC = WEB_ROOT / "static" / "vlattack_demo"
OUT_DATA = WEB_ROOT / "data" / "vlattack_demo"

TEACHER_SLOTS = [
    ("A_correct_clean", "393225001", "clean", "乾淨圖、低風險：Stage 1 正確放行"),
    ("B_correct_attack", "524361001", "attack", "攻擊成功、高風險：Stage 1 正確攔截"),
    ("C_false_alarm", "393254002", "clean", "乾淨圖被判高風險：假警報"),
    ("D_miss_attack", "262522002", "attack", "攻擊成功卻低風險：Stage 1 漏抓"),
    ("E_clean_mid", "393777005", "clean", "乾淨圖中風險：應送 Stage 2"),
    ("F_boundary", "131089003", "attack", "攻擊失敗但高風險：邊界案例"),
    ("G_attack_mid", "262148000", "attack", "攻擊成功、中風險：完整走 Stage 2 翻轉 + SSS 去噪"),
]


def load_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def index_csv(path: Path, key="image_id"):
    rows = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.setdefault(str(row[key]), []).append(row)
    return rows


def first_row(rows, **pred):
    for row in rows:
        ok = True
        for k, v in pred.items():
            if str(row.get(k, "")).lower() != str(v).lower():
                ok = False
                break
        if ok:
            return row
    return rows[0] if rows else {}


def copy_img(src: Path, dst: Path) -> bool:
    if not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def main():
    OUT_STATIC.mkdir(parents=True, exist_ok=True)
    OUT_DATA.mkdir(parents=True, exist_ok=True)

    log_by_qid = {str(item["question_id"]): item for item in load_json(EXP / "attack_dual_defense_log.json")}
    stage1 = index_csv(EXP / "tcvad_stage1_4_academic_dataset.csv")
    sss_raw = index_csv(SOURCE_RUN / "sss_per_theta_raw.csv")
    rate = load_json(SOURCE_RUN / "attack_rate_summary.json")
    sss_best = load_json(SOURCE_RUN / "sss_best_theta.json")
    theta_det = load_json(SOURCE_RUN / "theta_det.json")
    visual_exp = EXP / "visual_results"
    visual_src = SOURCE_RUN / "visual_results"

    samples = []
    for slot, qid, feed, why in TEACHER_SLOTS:
        log = log_by_qid.get(qid)
        if not log:
            print(f"skip {qid}: not in attack log")
            continue
        s1_rows = stage1.get(qid, [])
        s1_clean = first_row(s1_rows, attack_type="clean")
        s1_atk = first_row(s1_rows, attack_type="VLAttack")
        s1_feed = s1_clean if feed == "clean" else s1_atk

        dest = OUT_STATIC / qid
        dest.mkdir(parents=True, exist_ok=True)
        copied = {
            "ori": copy_img(visual_exp / f"{qid}_1_ori.png", dest / "ori.png"),
            "atk": copy_img(visual_exp / f"{qid}_2_atk.png", dest / "atk.png"),
            "gs_s": copy_img(visual_src / f"{qid}_4a_gs_s.png", dest / "gaussian_strong.png"),
            "gs_m": copy_img(visual_src / f"{qid}_4a_gs_m.png", dest / "gaussian_mild.png"),
        }
        print(qid, copied)

        flips = []
        for row in sss_raw.get(qid, []):
            if str(row.get("kind", "")).lower() != ("clean" if feed == "clean" else "attack"):
                continue
            pred_ref = str(row.get("pred_ref", ""))
            pred_dn = str(row.get("pred_denoised", ""))
            flips.append(
                {
                    "sigma": float(row["sigma"]),
                    "kernel": int(float(row["kernel"])),
                    "pred_ref": pred_ref,
                    "pred_denoised": pred_dn,
                    "flipped": pred_ref != pred_dn,
                    "rescued": str(row.get("rescued", "")).lower() == "true",
                    "fooled_before": str(row.get("fooled_before", "")).lower() == "true",
                    "caption_ref": row.get("caption_ref", ""),
                    "caption_denoised": row.get("caption_denoised", ""),
                    "ssim": float(row.get("ssim") or 0),
                    "clip_sim": float(row.get("clip_sim") or 0),
                }
            )
        flips.sort(key=lambda x: x["sigma"])
        flip_sigma15 = next((x for x in flips if abs(x["sigma"] - 1.5) < 1e-6), None)
        sss_sigma08 = next((x for x in flips if abs(x["sigma"] - 0.8) < 1e-6), None)

        risk = float(s1_feed.get("risk_score") or 0)
        risk_bin = s1_feed.get("risk_bin") or "mid"
        attack_success = bool(log.get("attack_success"))

        if risk_bin == "high":
            stage1_action = "攔截"
            e2e_flagged = True
        elif risk_bin == "low":
            stage1_action = "放行"
            e2e_flagged = False
        else:
            stage1_action = "送 Stage 2"
            e2e_flagged = bool(flip_sigma15 and flip_sigma15["flipped"]) if flip_sigma15 else False

        samples.append(
            {
                "id": slot,
                "question_id": qid,
                "slot": slot,
                "why": why,
                "feed": feed,
                "question": log.get("question_original", ""),
                "ground_truth": log.get("ground_truth", ""),
                "original_prediction": log.get("original_prediction", ""),
                "adversarial_prediction": log.get("adversarial_prediction", ""),
                "attack_success": attack_success,
                "attack_type": log.get("attack_type", ""),
                "ssim": (log.get("ssim") or {}).get("original_vs_attack"),
                "lpips": (log.get("lpips") or {}).get("original_vs_attack"),
                "clip": (log.get("clip") or {}).get("original_vs_attack"),
                "defenses": {
                    "gaussian_mild": log.get("gaussian_mild") or {},
                    "gaussian_strong": log.get("gaussian_strong") or {},
                    "mild_resize": log.get("mild_defense") or {},
                    "strong_resize": log.get("strong_defense") or {},
                    "resize_gaussian": log.get("resize_gaussian") or {},
                },
                "stage1": {
                    "method": "rotate_5",
                    "k": 10,
                    "risk_score": risk,
                    "risk_bin": risk_bin,
                    "action": stage1_action,
                    "d_center": float(s1_feed.get("d_center") or 0),
                    "d_iqr": float(s1_feed.get("d_iqr") or 0),
                    "d_tail": float(s1_feed.get("d_tail") or 0),
                    "d_q95": float(s1_feed.get("d_q95") or 0),
                    "clean": {
                        "risk_score": float(s1_clean.get("risk_score") or 0),
                        "risk_bin": s1_clean.get("risk_bin"),
                    },
                    "attack": {
                        "risk_score": float(s1_atk.get("risk_score") or 0),
                        "risk_bin": s1_atk.get("risk_bin"),
                    },
                },
                "stage2": {
                    "detector": "answer_flip",
                    "theta_det": {"kernel": 5, "sigma": 1.5},
                    "theta_star": {"kernel": 5, "sigma": 0.8},
                    "flip_sigma15": flip_sigma15,
                    "sss_sigma08": sss_sigma08,
                    "flips": flips,
                    "end_to_end_flagged": e2e_flagged,
                },
                "images": {
                    "ori": f"/static/vlattack_demo/{qid}/ori.png",
                    "atk": f"/static/vlattack_demo/{qid}/atk.png",
                    "gaussian_strong": f"/static/vlattack_demo/{qid}/gaussian_strong.png",
                    "gaussian_mild": f"/static/vlattack_demo/{qid}/gaussian_mild.png",
                },
            }
        )

    catalog = {
        "source": str(EXP),
        "attack": {
            "method": "VLAttack BSA",
            "whitebox": "BLIP pretrain model_base.pth",
            "blackbox": "BLIP VQA model_vqa.pth",
            "steps": 100,
            "alpha": 0.002,
            "linf": 0.125,
            "image_size": 480,
            "asr": rate.get("attack_success_rate"),
            "asr_count": f"{rate.get('attack_success_count')}/{rate.get('total_samples')}",
            "mean_ssim": rate.get("mean_ssim_original_vs_attack"),
        },
        "pipeline": [
            "VQAv2 原圖 + 問題（文字 prompt 保持乾淨）",
            "VLAttack PGD：steps=100、α=0.002、L∞=0.125，白盒梯度來自 BLIP pretrain",
            "黑盒 BLIP-VQA 推論：攻擊成功 ⇔ 對抗圖答案 ≠ 原圖答案",
            "TC-VAD Stage 1：rotate_5 × k=10 → BLIP caption → 4-D 角距離 → SVM 風險 r",
            "分流：r<0.3 放行、r≥0.7 攔截、其餘送 Stage 2",
            "Stage 2 偵測：答案翻轉 VQA(blur 5×5 σ=1.5) ≠ VQA(x)",
            "緩解 SSS：gaussian 5×5 σ=0.8 去噪後再問，尋找 Sweet Spot",
        ],
        "metrics": {
            "stage1_auroc": 0.6655,
            "residual_asr": rate.get("defense_residual_asr"),
            "sss_best": sss_best,
            "theta_det": theta_det,
            "end_to_end": [
                {"name": "Stage 1 僅 high", "tpr": 0.289, "fpr": 0.057, "j": 0.232},
                {"name": "caption-SVM Stage 2", "tpr": 0.328, "fpr": 0.111, "j": 0.217},
                {"name": "答案翻轉 σ=1.5", "tpr": 0.448, "fpr": 0.078, "j": 0.370},
                {"name": "flip_count ≥ 1", "tpr": 0.474, "fpr": 0.087, "j": 0.387},
            ],
            "sss_grid": [
                {"sigma": 0.3, "delta_asr": 0.018, "sss": 0.253},
                {"sigma": 0.5, "delta_asr": 0.073, "sss": 0.372},
                {"sigma": 0.8, "delta_asr": 0.479, "sss": 0.583},
                {"sigma": 1.0, "delta_asr": 0.570, "sss": 0.573},
                {"sigma": 1.5, "delta_asr": 0.697, "sss": 0.558},
            ],
        },
        "samples": samples,
    }
    out = OUT_DATA / "catalog.json"
    out.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out} ({len(samples)} samples)")


if __name__ == "__main__":
    main()
