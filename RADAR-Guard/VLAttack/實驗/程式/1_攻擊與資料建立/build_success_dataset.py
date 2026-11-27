"""Curate the experiment-2 dataset: N attack-successful samples plus their clean pairs.

Experiment 1 attacked 1000 samples but only 462 fooled the model, which dilutes any
detector whose signal depends on the answer actually having been flipped. This script
filters a source run down to samples where attack_success is true and both the clean
and adversarial PNGs exist, producing a directory laid out exactly like a normal run
so run_full_pipeline.py can consume it unchanged.

Images are hardlinked rather than copied, so the curated set costs no extra disk as
long as source and destination sit on the same volume.
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE = THIS_DIR.parent / "steps100_alpha0.002_run1000"
NEEDED_SUFFIXES = ("_1_ori.png", "_2_atk.png")


def link_or_copy(src, dst):
    # Copy rather than hardlink: OneDrive on Windows can make hardlinked PNGs
    # unreadable (OSError 22 Invalid argument) even though the file exists.
    if dst.exists():
        try:
            dst.unlink()
        except OSError:
            pass
    shutil.copy2(src, dst)
    return "copy"


def build(source_dir, out_dir, target_n, require_success=True):
    source_dir = Path(source_dir).resolve()
    out_dir = Path(out_dir).resolve()
    src_visual = source_dir / "visual_results"
    src_log = source_dir / "attack_dual_defense_log.json"
    for p in (src_visual, src_log):
        if not p.exists():
            raise FileNotFoundError(f"missing source: {p}")

    with open(src_log, "r", encoding="utf-8") as f:
        logs = json.load(f)
    print(f"source log: {len(logs)} entries from {src_log.parent.name}")

    picked, skipped_fail, skipped_img = [], 0, 0
    for item in logs:
        if require_success and not item.get("attack_success"):
            skipped_fail += 1
            continue
        qid = str(item["question_id"])
        paths = [src_visual / f"{qid}{sfx}" for sfx in NEEDED_SUFFIXES]
        if not all(p.is_file() for p in paths):
            skipped_img += 1
            continue
        picked.append(item)
        if target_n and len(picked) >= target_n:
            break

    print(f"eligible picked: {len(picked)}  (skipped: {skipped_fail} not-success, {skipped_img} missing-image)")
    if target_n and len(picked) < target_n:
        print(f"WARNING: only {len(picked)} of {target_n} available -- run more attack samples first")

    out_dir.mkdir(parents=True, exist_ok=True)
    dst_visual = out_dir / "visual_results"
    dst_visual.mkdir(exist_ok=True)

    modes = {}
    for item in picked:
        qid = str(item["question_id"])
        for sfx in NEEDED_SUFFIXES:
            mode = link_or_copy(src_visual / f"{qid}{sfx}", dst_visual / f"{qid}{sfx}")
            modes[mode] = modes.get(mode, 0) + 1
    print(f"images: {modes}")

    with open(out_dir / "attack_dual_defense_log.json", "w", encoding="utf-8") as f:
        json.dump(picked, f, ensure_ascii=False, indent=2)

    manifest = pd.DataFrame(
        [
            {
                "question_id": it["question_id"],
                "question": it.get("question_original", ""),
                "ground_truth": it.get("ground_truth", ""),
                "original_prediction": it.get("original_prediction", ""),
                "adversarial_prediction": it.get("adversarial_prediction", ""),
                "attack_success": it.get("attack_success"),
                "attack_type": it.get("attack_type", ""),
                "ssim": (it.get("ssim") or {}).get("original_vs_attack"),
                "lpips": (it.get("lpips") or {}).get("original_vs_attack"),
                "clip": (it.get("clip") or {}).get("original_vs_attack"),
            }
            for it in picked
        ]
    )
    manifest.to_csv(out_dir / "dataset_manifest.csv", index=False, encoding="utf-8")

    info = [
        f"source_run={source_dir}",
        f"require_success={require_success}",
        f"target_n={target_n}",
        f"selected={len(picked)}",
        f"clean_rows={len(picked)}  adversarial_rows={len(picked)}  stage1_rows={2 * len(picked)}",
        f"mean_ssim={manifest['ssim'].mean():.4f}" if len(manifest) else "mean_ssim=nan",
    ]
    (out_dir / "dataset_info.txt").write_text("\n".join(info) + "\n", encoding="utf-8")

    print("\n".join(info))
    print(f"\nSaved {out_dir / 'attack_dual_defense_log.json'}")
    print(f"Saved {out_dir / 'dataset_manifest.csv'}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build attack-successful subset for experiment 2")
    parser.add_argument("--source_dir", default=str(DEFAULT_SOURCE))
    parser.add_argument("--out_dir", default=str(THIS_DIR))
    parser.add_argument("--target_n", type=int, default=1000)
    parser.add_argument("--include_failed", action="store_true", help="do not filter on attack_success")
    args = parser.parse_args()
    build(args.source_dir, args.out_dir, args.target_n, require_success=not args.include_failed)
