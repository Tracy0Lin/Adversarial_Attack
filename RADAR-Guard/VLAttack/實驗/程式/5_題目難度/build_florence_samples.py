"""Unique Florence victim and VLAttack questions, judged on the clean image.

Questions already labeled for BLIP are copied. Only the rest are batched.
"""
import os
import json
from pathlib import Path

import pandas as pd

VQA = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "florence"
BLIP_LABELS = Path(__file__).resolve().parent / "labels.csv"
SCREEN = VQA / "florence_victim" / "screening_log.jsonl"
IMG_ROOT = Path(os.environ.get("VLATTACK_IMG_ROOT", r"C:\Users\tracy_aboikg6\OneDrive\Desktop\program\VLAttack_Data\vqa_images"))
STUDIES = {
    "victim": {
        "exp1": VQA / "florence_victim" / "exp1_mixed1000",
        "exp2": VQA / "florence_victim" / "exp2_success1000",
        "exp3": VQA / "florence_victim" / "exp3_success1000",
    },
    "vlattack": {
        "exp1": VQA / "florence_vlattack" / "exp1_mixed1000",
        "exp2": VQA / "florence_vlattack" / "exp2_success1000",
        "exp3": VQA / "florence_vlattack" / "exp3_success1000",
    },
}


def load_screen():
    out = {}
    if not SCREEN.is_file():
        return out
    for line in SCREEN.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        out[str(rec["question_id"])] = rec
    return out


def clean_image(qid, screen):
    name = f"{qid}_1_ori.png"
    for study in STUDIES.values():
        for folder in study.values():
            path = folder / "visual_results" / name
            if path.is_file():
                return str(path)
    rec = screen.get(qid) or {}
    image = rec.get("image") or ""
    if image:
        path = IMG_ROOT / image
        if path.is_file():
            return str(path)
    return ""


def main():
    screen = load_screen()
    rows = []
    for study, exps in STUDIES.items():
        for exp, folder in exps.items():
            log = json.loads((folder / "attack_dual_defense_log.json").read_text(encoding="utf-8"))
            for rec in log:
                qid = str(rec["question_id"])
                raw = (screen.get(qid) or {}).get("question") or rec.get("question_original") or ""
                rows.append(
                    {
                        "question_id": qid,
                        "question": str(raw).strip(),
                        "study_exp": f"{study}:{exp}",
                    }
                )
    all_rows = pd.DataFrame(rows)
    grouped = (
        all_rows.groupby("question_id", as_index=False)
        .agg(
            question=("question", "first"),
            n_question_texts=("question", "nunique"),
            experiments=("study_exp", lambda s: ";".join(sorted(set(s)))),
        )
    )
    grouped["image_path"] = grouped["question_id"].map(lambda q: clean_image(q, screen))
    grouped["image_ok"] = grouped["image_path"].ne("")
    grouped = grouped.sort_values("question_id")

    blip = pd.read_csv(BLIP_LABELS, dtype={"question_id": str})
    blip["question_id"] = blip["question_id"].str.replace(r"\.0$", "", regex=True)
    known = blip.set_index("question_id")[["difficulty_3", "reason"]]
    grouped["difficulty_3"] = grouped["question_id"].map(known["difficulty_3"])
    grouped["reason"] = grouped["question_id"].map(known["reason"])
    grouped["from_blip"] = grouped["difficulty_3"].notna()

    OUT.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(OUT / "samples.csv", index=False, encoding="utf-8")
    ready = grouped[grouped["image_ok"] & ~grouped["from_blip"]]
    batch_dir = OUT / "batches"
    batch_dir.mkdir(exist_ok=True)
    for old in batch_dir.glob("batch_*.json"):
        old.unlink()
    label_dir = OUT / "labels"
    label_dir.mkdir(exist_ok=True)
    size = 20
    n_batches = 0
    records = ready.to_dict(orient="records")
    for i in range(0, len(records), size):
        chunk = [
            {
                "question_id": row["question_id"],
                "question": row["question"],
                "image_path": row["image_path"],
            }
            for row in records[i : i + size]
        ]
        (batch_dir / f"batch_{n_batches:03d}.json").write_text(
            json.dumps(chunk, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        n_batches += 1
    missing_image = grouped[~grouped["image_ok"]]
    print(
        json.dumps(
            {
                "log_rows": int(len(all_rows)),
                "unique_questions": int(len(grouped)),
                "already_labeled_by_blip": int(grouped["from_blip"].sum()),
                "need_judge": int(len(ready)),
                "images_missing": int((~grouped["image_ok"]).sum()),
                "missing_and_unlabeled": int((~grouped["image_ok"] & ~grouped["from_blip"]).sum()),
                "question_text_conflicts": int((grouped["n_question_texts"] > 1).sum()),
                "batches": n_batches,
                "by_membership": grouped["experiments"].value_counts().head(20).to_dict(),
                "missing_ids": missing_image["question_id"].head(10).tolist(),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
