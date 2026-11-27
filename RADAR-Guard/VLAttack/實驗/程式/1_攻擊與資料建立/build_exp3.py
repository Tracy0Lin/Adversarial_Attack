"""Build experiment 3: 1000 successful attacks sharing no photo with experiments 1 or 2."""
import argparse
import json
import shutil

import pandas as pd

from common import EXP2_DIR, EXP3_DIR, RUN1000, SNAPSHOT, photo_id, read_json


def build(target=1000):
    logs = read_json(RUN1000 / "attack_dual_defense_log.json")
    exp1 = {str(x["question_id"]) for x in read_json(SNAPSHOT)[:1000]}
    exp2 = {str(x["question_id"]) for x in read_json(EXP2_DIR / "attack_dual_defense_log.json")}
    banned_photos = {photo_id(q) for q in exp1 | exp2}
    src = RUN1000 / "visual_results"

    picked, reasons = [], {"not_success": 0, "in_exp1_or_exp2": 0, "shared_photo": 0, "missing_image": 0}
    for item in logs:
        qid = str(item["question_id"])
        if not item.get("attack_success"):
            reasons["not_success"] += 1
        elif qid in exp1 or qid in exp2:
            reasons["in_exp1_or_exp2"] += 1
        elif photo_id(qid) in banned_photos:
            reasons["shared_photo"] += 1
        elif not (src / f"{qid}_2_atk.png").is_file() or not (src / f"{qid}_1_ori.png").is_file():
            reasons["missing_image"] += 1
        else:
            picked.append(item)
            if len(picked) >= target:
                break
    print(f"source log={len(logs)} picked={len(picked)} skipped={reasons}")
    if len(picked) < target:
        raise SystemExit(f"only {len(picked)} eligible; need {target}")

    dst = EXP3_DIR / "visual_results"
    dst.mkdir(parents=True, exist_ok=True)
    for item in picked:
        qid = str(item["question_id"])
        for sfx in ("_1_ori.png", "_2_atk.png"):
            shutil.copy2(src / f"{qid}{sfx}", dst / f"{qid}{sfx}")
    with open(EXP3_DIR / "attack_dual_defense_log.json", "w", encoding="utf-8") as f:
        json.dump(picked, f, ensure_ascii=False, indent=2)

    manifest = pd.DataFrame([{
        "question_id": it["question_id"],
        "photo_id": photo_id(it["question_id"]),
        "question": it.get("question_original", ""),
        "original_prediction": it.get("original_prediction", ""),
        "adversarial_prediction": it.get("adversarial_prediction", ""),
        "ssim": (it.get("ssim") or {}).get("original_vs_attack"),
    } for it in picked])
    manifest.to_csv(EXP3_DIR / "dataset_manifest.csv", index=False, encoding="utf-8")
    info = [
        f"source_log_entries={len(logs)}",
        f"selected={len(picked)} photos={manifest['photo_id'].nunique()}",
        f"skipped={reasons}",
        f"mean_ssim={manifest['ssim'].mean():.4f}",
        "rule: attack_success, qid not in exp1/exp2, photo not in exp1/exp2, both PNGs present",
    ]
    (EXP3_DIR / "dataset_info.txt").write_text("\n".join(info) + "\n", encoding="utf-8")
    print("\n".join(info))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--target", type=int, default=1000)
    build(p.parse_args().target)
