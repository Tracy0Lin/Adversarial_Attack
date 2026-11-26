"""Build exp1 / exp2 / exp3 from the Florence feature-distance attack log.

exp1: first 1000 attacked questions, failures kept.
exp2: successes taken from exp1, then later successes until 1000.
exp3: 1000 later successes whose photos are outside exp1 and exp2.
"""
import json
import shutil
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "VQA" / "florence_vlattack"
MASTER = DATA / "feature_attack"


def photo_id(qid):
    return int(qid) // 1000


def load():
    rows = json.loads((MASTER / "attack_dual_defense_log.json").read_text(encoding="utf-8"))
    if any("adversarial_prediction" not in row for row in rows):
        raise SystemExit("master log still has unscored rows")
    return rows


def export(name, rows, include_clean):
    dst = DATA / name
    vis = dst / "visual_results"
    vis.mkdir(parents=True, exist_ok=True)
    src = MASTER / "visual_results"
    for row in rows:
        qid = row["question_id"]
        for sfx in ("_1_ori.png", "_2_atk.png"):
            shutil.copy2(src / f"{qid}{sfx}", vis / f"{qid}{sfx}")
    (dst / "attack_dual_defense_log.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    pd.DataFrame(
        [{"question_id": r["question_id"], "photo_id": photo_id(r["question_id"])} for r in rows]
    ).to_csv(dst / "dataset_manifest.csv", index=False, encoding="utf-8")
    n_ok = sum(1 for r in rows if r.get("attack_success"))
    (dst / "dataset_info.txt").write_text(
        f"source={MASTER}\nselected={len(rows)}\nattack_success={n_ok}\ninclude_clean={include_clean}\n",
        encoding="utf-8",
    )
    print(f"{name}: rows={len(rows)} success={n_ok}", flush=True)


def split(rows):
    if len(rows) < 1000:
        print(f"NEED_MORE exp1 {len(rows)}/1000", flush=True)
        raise SystemExit(2)
    exp1 = rows[:1000]
    exp2 = [row for row in exp1 if row.get("attack_success")]
    cursor = 1000
    if len(exp2) > 1000:
        exp2 = exp2[:1000]
    elif len(exp2) < 1000:
        for i, row in enumerate(rows[1000:], start=1000):
            cursor = i + 1
            if row.get("attack_success"):
                exp2.append(row)
            if len(exp2) == 1000:
                break
    if len(exp2) < 1000:
        print(f"NEED_MORE exp2 {len(exp2)}/1000", flush=True)
        raise SystemExit(2)
    banned = {photo_id(r["question_id"]) for r in exp1 + exp2}
    used = {int(r["question_id"]) for r in exp1 + exp2}
    exp3 = []
    for row in rows[cursor:]:
        qid = int(row["question_id"])
        if qid in used or not row.get("attack_success"):
            continue
        if photo_id(qid) in banned:
            continue
        exp3.append(row)
        if len(exp3) == 1000:
            break
    if len(exp3) < 1000:
        print(f"NEED_MORE exp3 {len(exp3)}/1000", flush=True)
        raise SystemExit(2)
    return exp1, exp2, exp3


def main():
    parts = split(load())
    if parts is None:
        raise SystemExit("not enough successes yet")
    exp1, exp2, exp3 = parts
    export("exp1_mixed1000", exp1, True)
    export("exp2_success1000", exp2, True)
    export("exp3_success1000", exp3, False)


if __name__ == "__main__":
    main()
