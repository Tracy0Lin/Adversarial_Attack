"""Experiment 3: 1000 Florence attack-success images, photo-disjoint from exp1 & exp2."""
import json
import shutil
from pathlib import Path

import pandas as pd

from common import DATA_ROOT, EXP1_DIR, EXP2_DIR, EXP3_DIR, read_json, write_json


def photo_id(qid):
    return int(qid) // 1000


def eligible_count(target=1000, source_root=None):
    source_root = Path(source_root or DATA_ROOT / "master_attack")
    src_log = source_root / "attack_dual_defense_log.json"
    if not src_log.is_file():
        return 0
    logs = read_json(src_log)
    exp1_q = {str(x["question_id"]) for x in read_json(EXP1_DIR / "attack_dual_defense_log.json")}
    exp2_q = {str(x["question_id"]) for x in read_json(EXP2_DIR / "attack_dual_defense_log.json")}
    banned_photos = {photo_id(q) for q in exp1_q | exp2_q}
    exclude_path = DATA_ROOT / "exp3_exclude_qids.json"
    excluded = set()
    if exclude_path.is_file():
        raw = json.loads(exclude_path.read_text(encoding="utf-8"))
        excluded = {str(x) for x in (raw.get("question_ids", raw) if isinstance(raw, dict) else raw)}
    n = 0
    seen = set()
    src_visual = source_root / "visual_results"
    for item in logs:
        qid = str(item["question_id"])
        if qid in seen:
            continue
        seen.add(qid)
        if qid in excluded or not item.get("attack_success"):
            continue
        if qid in exp1_q or qid in exp2_q:
            continue
        if photo_id(qid) in banned_photos:
            continue
        if not (src_visual / f"{qid}_2_atk.png").is_file():
            continue
        n += 1
        if n >= target:
            return target
    return n


def build(target=1000, source_root=None):
    source_root = Path(source_root or DATA_ROOT / "master_attack")
    src_log = source_root / "attack_dual_defense_log.json"
    src_visual = source_root / "visual_results"
    logs = read_json(src_log)

    exp1_q = {str(x["question_id"]) for x in read_json(EXP1_DIR / "attack_dual_defense_log.json")}
    exp2_q = {str(x["question_id"]) for x in read_json(EXP2_DIR / "attack_dual_defense_log.json")}
    banned_photos = {photo_id(q) for q in exp1_q | exp2_q}

    exclude_path = DATA_ROOT / "exp3_exclude_qids.json"
    excluded = set()
    if exclude_path.is_file():
        raw = json.loads(exclude_path.read_text(encoding="utf-8"))
        excluded = {str(x) for x in (raw.get("question_ids", raw) if isinstance(raw, dict) else raw)}
    picked, reasons = [], {"not_success": 0, "in_exp12": 0, "shared_photo": 0, "missing_image": 0, "excluded": 0}
    seen_q = set()
    for item in logs:
        qid = str(item["question_id"])
        if qid in seen_q:
            continue
        seen_q.add(qid)
        if qid in excluded:
            reasons["excluded"] += 1
            continue
        if not item.get("attack_success"):
            reasons["not_success"] += 1
            continue
        if qid in exp1_q or qid in exp2_q:
            reasons["in_exp12"] += 1
            continue
        if photo_id(qid) in banned_photos:
            reasons["shared_photo"] += 1
            continue
        if not (src_visual / f"{qid}_2_atk.png").is_file():
            reasons["missing_image"] += 1
            continue
        picked.append(item)
        if len(picked) >= target:
            break

    print(f"master log={len(logs)} picked={len(picked)} skipped={reasons}", flush=True)
    if len(picked) < target:
        raise SystemExit(f"need {target} exp3 rows, only {len(picked)}")

    dst = EXP3_DIR / "visual_results"
    dst.mkdir(parents=True, exist_ok=True)
    for item in picked:
        qid = str(item["question_id"])
        for sfx in ("_1_ori.png", "_2_atk.png"):
            shutil.copy2(src_visual / f"{qid}{sfx}", dst / f"{qid}{sfx}")

    write_json(EXP3_DIR / "attack_dual_defense_log.json", picked)
    pd.DataFrame([{"question_id": it["question_id"], "photo_id": photo_id(it["question_id"])} for it in picked]).to_csv(
        EXP3_DIR / "dataset_manifest.csv", index=False, encoding="utf-8",
    )
    (EXP3_DIR / "dataset_info.txt").write_text(
        f"source={source_root}\nselected={len(picked)}\nskipped={reasons}\n", encoding="utf-8",
    )


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=1000)
    ap.add_argument("--source_root", default=None)
    args = ap.parse_args()
    build(args.target, args.source_root)
