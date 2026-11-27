"""Join batch JSONL labels onto the sample list and add the binary cut."""
import json
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent


def main():
    rows = []
    for path in sorted((OUT / "labels").glob("batch_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    labels = pd.DataFrame(rows)
    labels["question_id"] = labels["question_id"].astype(str)
    dup = int(labels["question_id"].duplicated().sum())
    samples = pd.read_csv(OUT / "samples.csv", dtype={"question_id": str})
    merged = samples.merge(labels, on="question_id", how="left")
    merged["difficulty_binary"] = merged["difficulty_3"].map(
        lambda x: "hard" if x == "hard" else ("not_hard" if x in {"easy", "medium"} else x)
    )
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
