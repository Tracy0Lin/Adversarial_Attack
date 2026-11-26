"""Build exp2: 1000 attack-success samples (+ clean pairs) from exp1 source."""
import json
import shutil
from pathlib import Path

import pandas as pd

from common import EXP1_DIR, EXP2_DIR, write_json


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1000)
    args = p.parse_args()

    src_log = EXP1_DIR / "attack_dual_defense_log.json"
    src_vis = EXP1_DIR / "visual_results"
    logs = json.load(open(src_log, "r", encoding="utf-8"))

    picked = []
    for item in logs:
        if not item.get("attack_success"):
            continue
        qid = str(item["question_id"])
        if not (src_vis / f"{qid}_2_atk.png").is_file():
            continue
        picked.append(item)
        if len(picked) >= args.n:
            break
    if len(picked) < args.n:
        raise SystemExit(f"only {len(picked)} successes in exp1; need {args.n}")

    vis2 = EXP2_DIR / "visual_results"
    vis2.mkdir(parents=True, exist_ok=True)
    for item in picked:
        qid = str(item["question_id"])
        for suffix in ("_1_ori.png", "_2_atk.png"):
            shutil.copy2(src_vis / f"{qid}{suffix}", vis2 / f"{qid}{suffix}")

    write_json(EXP2_DIR / "attack_dual_defense_log.json", picked)
    pd.DataFrame([{"question_id": it["question_id"], "attack_success": True} for it in picked]).to_csv(
        EXP2_DIR / "dataset_manifest.csv", index=False, encoding="utf-8",
    )
    print(f"exp2: {len(picked)} rows", flush=True)


if __name__ == "__main__":
    main()
