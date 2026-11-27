"""Join Florence labels, reusing BLIP judgments for shared questions."""
import json
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent / "florence"
BLIP_LABELS = Path(__file__).resolve().parent / "labels.csv"


def binary(value):
    if value == "hard":
        return "hard"
    if value in {"easy", "medium"}:
        return "not_hard"
    return value


def main():
    rows = []
    for path in sorted((OUT / "labels").glob("batch_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    fresh = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["question_id", "difficulty_3", "reason"])
    if not fresh.empty:
        fresh["question_id"] = fresh["question_id"].astype(str)
    dup = int(fresh["question_id"].duplicated().sum()) if not fresh.empty else 0
    samples = pd.read_csv(OUT / "samples.csv", dtype={"question_id": str})
    blip = pd.read_csv(BLIP_LABELS, dtype={"question_id": str})[["question_id", "difficulty_3", "reason"]]
    blip["question_id"] = blip["question_id"].str.replace(r"\.0$", "", regex=True)
    base = samples.drop(columns=[c for c in ("difficulty_3", "reason", "from_blip") if c in samples.columns])
    merged = base.merge(blip, on="question_id", how="left")
    if not fresh.empty:
        fresh = fresh.drop_duplicates("question_id", keep="last")
        merged = merged.merge(fresh, on="question_id", how="left", suffixes=("", "_new"))
        take = merged["difficulty_3_new"].notna()
        merged.loc[take, "difficulty_3"] = merged.loc[take, "difficulty_3_new"]
        merged.loc[take, "reason"] = merged.loc[take, "reason_new"]
        merged = merged.drop(columns=["difficulty_3_new", "reason_new"])
    merged["difficulty_binary"] = merged["difficulty_3"].map(binary)
    merged.to_csv(OUT / "labels.csv", index=False, encoding="utf-8")
    done = merged["difficulty_3"].notna()
    print(
        json.dumps(
            {
                "labeled": int(done.sum()),
                "missing": int((~done).sum()),
                "duplicate_ids_in_jsonl": dup,
                "difficulty_3": merged.loc[done, "difficulty_3"].value_counts().to_dict(),
                "difficulty_binary": merged.loc[done, "difficulty_binary"].value_counts().to_dict(),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
