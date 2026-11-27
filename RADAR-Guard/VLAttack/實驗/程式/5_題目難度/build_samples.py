"""Unique BLIP exp1/exp2/exp3 questions, judged later on the clean image only."""
import json
from pathlib import Path

import pandas as pd

VQA = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
VIS = {
    "run": VQA / "steps100_alpha0.002_run1000" / "visual_results",
    "exp2": VQA / "exp2_success1000" / "visual_results",
    "exp3": VQA / "multi_captioner" / "exp3_success1000" / "visual_results",
}
MANIFESTS = {
    "exp1": VQA / "exp1_mixed1000" / "dataset_manifest.csv",
    "exp2": VQA / "exp2_success1000" / "dataset_manifest.csv",
    "exp3": VQA / "multi_captioner" / "exp3_success1000" / "dataset_manifest.csv",
}


def clean_image(qid: str) -> str:
    name = f"{qid}_1_ori.png"
    for key in ("run", "exp2", "exp3"):
        path = VIS[key] / name
        if path.is_file():
            return str(path)
    return ""


def main():
    frames = []
    for exp, path in MANIFESTS.items():
        df = pd.read_csv(path, dtype={"question_id": str})
        df["question_id"] = df["question_id"].str.replace(r"\.0$", "", regex=True)
        df["exp"] = exp
        frames.append(df[["question_id", "question", "exp"]])
    all_rows = pd.concat(frames, ignore_index=True)
    grouped = (
        all_rows.groupby("question_id", as_index=False)
        .agg(
            question=("question", "first"),
            n_question_texts=("question", "nunique"),
            experiments=("exp", lambda s: ";".join(sorted(set(s)))),
        )
    )
    grouped["image_path"] = grouped["question_id"].map(clean_image)
    grouped["image_ok"] = grouped["image_path"].ne("")
    grouped = grouped.sort_values("question_id")

    OUT.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(OUT / "samples.csv", index=False, encoding="utf-8")
    ready = grouped[grouped["image_ok"]]
    batch_dir = OUT / "batches"
    batch_dir.mkdir(exist_ok=True)
    for old in batch_dir.glob("batch_*.json"):
        old.unlink()
    size = 20
    n_batches = 0
    records = ready.to_dict(orient="records")
    for i in range(0, len(records), size):
        chunk = []
        for row in records[i : i + size]:
            chunk.append(
                {
                    "question_id": row["question_id"],
                    "question": row["question"],
                    "image_path": row["image_path"],
                }
            )
        name = f"batch_{n_batches:03d}.json"
        (batch_dir / name).write_text(
            json.dumps(chunk, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        n_batches += 1
    print(
        json.dumps(
            {
                "rows_in_manifests": int(len(all_rows)),
                "unique_questions": int(len(grouped)),
                "images_found": int(ready.shape[0]),
                "images_missing": int((~grouped["image_ok"]).sum()),
                "question_text_conflicts": int((grouped["n_question_texts"] > 1).sum()),
                "batches": n_batches,
                "by_exp_membership": grouped["experiments"].value_counts().to_dict(),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
