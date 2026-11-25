"""RADAR-Guard (VLAttack) Command-Line Interface.

Supports:
  --status                       Show CUDA/CPU device status and weight resolution
  --demo-case <case_id>          Inspect a curated pre-computed demo case (e.g. G_attack_mid)
  --list-cases                   List the 7 curated demo cases
  --image <path> --question <q>  Run live Stage 1 detection and/or Stage 2 defense (or BSA attack)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vlattack_bridge import get_sample, list_samples
from vlattack_engine import (
    ANSWER_LIST,
    CAPTION_WEIGHTS,
    DEFAULT_ALPHA,
    DEFAULT_EPS,
    DEFAULT_STEPS,
    PRETRAIN_WEIGHTS,
    STAGE1_CSV,
    VQA_WEIGHTS,
    engine_status,
    get_engine,
)
from vlattack_live import DET_SIGMA, SSS_SIGMA

LOW_RISK_MAX = 0.30
HIGH_RISK_MIN = 0.70


def print_json(obj: dict | list):
    print(json.dumps(obj, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description="RADAR-Guard (VLAttack) CLI")
    parser.add_argument("--status", action="store_true", help="Print engine and checkpoint status")
    parser.add_argument("--list-cases", action="store_true", help="List precomputed 7 demo cases")
    parser.add_argument("--demo-case", type=str, help="Inspect one precomputed demo case by ID")
    parser.add_argument("--image", type=Path, help="Input image path for live inference")
    parser.add_argument("--question", type=str, help="VQA question string")
    parser.add_argument(
        "--mode",
        choices=["detect", "defend", "attack_and_defend"],
        default="defend",
        help="Pipeline mode when --image is provided",
    )
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    args = parser.parse_args()

    if args.status:
        st = engine_status()
        st["paths"] = {
            "pretrain_weights": str(PRETRAIN_WEIGHTS),
            "pretrain_exists": PRETRAIN_WEIGHTS.is_file(),
            "vqa_weights": str(VQA_WEIGHTS),
            "vqa_exists": VQA_WEIGHTS.is_file(),
            "caption_weights": str(CAPTION_WEIGHTS),
            "caption_exists": CAPTION_WEIGHTS.is_file(),
            "answer_list": str(ANSWER_LIST),
            "answer_list_exists": ANSWER_LIST.is_file(),
            "stage1_csv": str(STAGE1_CSV),
            "stage1_csv_exists": STAGE1_CSV.is_file(),
        }
        print_json(st)
        return

    if args.list_cases:
        print_json([{"id": s["id"], "label": s.get("label"), "question": s.get("question")} for s in list_samples()])
        return

    if args.demo_case:
        sample = get_sample(args.demo_case)
        if sample is None:
            raise SystemExit(f"Case not found: {args.demo_case}")
        print_json(sample)
        return

    if not args.image or not args.question:
        parser.print_help()
        return

    if not args.image.is_file():
        raise SystemExit(f"Image not found: {args.image}")

    pil = Image.open(args.image).convert("RGB")
    engine = get_engine()
    engine.load()

    if args.mode == "attack_and_defend":
        atk = engine.attack(pil, args.question, eps=args.eps, alpha=args.alpha, steps=args.steps)
        target_pil = atk["adv_pil"]
        attack_summary = {
            "original_answer": atk["original_answer"],
            "adversarial_answer": atk["adversarial_answer"],
            "attack_success": atk["attack_success"],
            "ssim": atk["ssim"],
        }
    else:
        target_pil = pil
        attack_summary = None

    det = engine.score(target_pil, args.image.stem)
    risk_bin = det["risk_bin"]
    out = {
        "image": str(args.image),
        "question": args.question,
        "attack": attack_summary,
        "stage1": det,
        "routing_thresholds": {"low_max": LOW_RISK_MAX, "high_min": HIGH_RISK_MIN},
    }

    if args.mode in {"defend", "attack_and_defend"}:
        ans_raw = engine.answer_pil(target_pil, args.question)
        if risk_bin == "low":
            out["stage2"] = {"executed": False, "decision": "放行（Low Risk）", "final_answer": ans_raw}
        elif risk_bin == "high":
            out["stage2"] = {"executed": False, "decision": "攔截（High Risk）", "final_answer": None}
        else:
            repaired = engine.blur(target_pil, SSS_SIGMA)
            probed = engine.blur(target_pil, DET_SIGMA)
            ans_repair = engine.answer_pil(repaired, args.question)
            ans_probe = engine.answer_pil(probed, args.question)
            out["stage2"] = {
                "executed": True,
                "decision": "高斯修復放行（Mid Risk）",
                "repair_sigma": SSS_SIGMA,
                "probe_sigma": DET_SIGMA,
                "answer_before_repair": ans_raw,
                "final_answer_after_repair": ans_repair,
                "probe_answer": ans_probe,
                "answer_flipped_on_probe": ans_probe != ans_raw,
            }

    print_json(out)


if __name__ == "__main__":
    main()
