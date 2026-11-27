"""Cosine distance of VQA answers inside the existing C1/C2/C4/C5 buckets.

BLIP and Florence white-box PGD, experiments 1–3. Embeddings are
all-MiniLM-L6-v2. The median is computed only on pairs whose normalized
strings differ, so identical answers do not pad C4 up to 1.0.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

DEEP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DEEP))
from build_report import VQA, load_rows  # noqa: E402

OUT = Path(__file__).resolve().parent
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
KEEP = ["C1", "C2", "C4", "C5"]
TYPES = ["Yes/No", "Number", "Other", "All"]
NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six",
    "seven", "eight", "nine", "ten", "none",
}

JOBS = [
    ("blip", "exp1", VQA / "multi_captioner" / "exp1" / "blip" / "rows_scored.csv"),
    ("blip", "exp2", VQA / "multi_captioner" / "exp2" / "blip" / "rows_scored.csv"),
    ("blip", "exp3", VQA / "multi_captioner" / "exp3" / "blip" / "rows_scored.csv"),
    ("florence", "exp1", VQA / "florence_victim" / "exp1" / "florence2" / "vqa_florence2" / "rows_scored.csv"),
    ("florence", "exp2", VQA / "florence_victim" / "exp2" / "florence2" / "vqa_florence2" / "rows_scored.csv"),
    ("florence", "exp3", VQA / "florence_victim" / "exp3" / "florence2" / "vqa_florence2" / "rows_scored.csv"),
]


def answer_type(ans: str) -> str:
    ans = norm(ans)
    if ans in {"yes", "no"}:
        return "Yes/No"
    if ans.isdigit() or ans in NUMBER_WORDS:
        return "Number"
    return "Other"


def norm(text) -> str:
    return str(text).lower().strip()


def mean_pool(last_hidden, mask):
    mask = mask.unsqueeze(-1).expand(last_hidden.size()).float()
    summed = torch.sum(last_hidden * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def embed_texts(texts: list[str]) -> dict[str, np.ndarray]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to(device)
    model.eval()
    vectors = {}
    batch = 64
    with torch.no_grad():
        for start in range(0, len(texts), batch):
            chunk = texts[start:start + batch]
            tokens = tokenizer(
                chunk, padding=True, truncation=True, max_length=64, return_tensors="pt"
            ).to(device)
            hidden = model(**tokens).last_hidden_state
            pooled = mean_pool(hidden, tokens["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            for text, vec in zip(chunk, pooled.cpu().numpy()):
                vectors[text] = vec
    return vectors


def pair_stats(frame: pd.DataFrame, left: str, right: str, vectors: dict[str, np.ndarray]):
    a = frame[left].map(norm)
    b = frame[right].map(norm)
    same = a.eq(b)
    n = int(len(frame))
    n_same = int(same.sum())
    differ = ~same
    if differ.any():
        sims = [
            float(np.dot(vectors[x], vectors[y]))
            for x, y in zip(a[differ], b[differ])
        ]
        median = float(np.median(sims))
        n_median = int(differ.sum())
    else:
        median = np.nan
        n_median = 0
    return n, (n_same / n if n else np.nan), median, n_median


def summarize(frame: pd.DataFrame, vectors: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for category in KEEP:
        cat = frame[frame["category"] == category]
        for kind in TYPES:
            sub = cat if kind == "All" else cat[cat["answer_type"] == kind]
            n, same_ref, med_ref, n_ref = pair_stats(sub, "original_prediction", "pred_ref", vectors)
            row = {
                "category": category,
                "answer_type": kind,
                "n": n,
                "same_ref_rate": same_ref,
                "sim_ref_median": med_ref,
                "sim_ref_n": n_ref,
                "same_blur_rate": np.nan,
                "sim_blur_median": np.nan,
                "sim_blur_n": 0,
            }
            if category in {"C2", "C4"}:
                _, same_b, med_b, n_b = pair_stats(sub, "original_prediction", "pred_0.8", vectors)
                row["same_blur_rate"] = same_b
                row["sim_blur_median"] = med_b
                row["sim_blur_n"] = n_b
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    frames = []
    texts = set()
    for study, exp, path in JOBS:
        df = load_rows(path, exp)
        df = df[df["category"].isin(KEEP)].copy()
        df["answer_type"] = df["original_prediction"].map(answer_type)
        df["study"] = study
        frames.append(df)
        for col in ("original_prediction", "pred_ref", "pred_0.8"):
            texts.update(df[col].map(norm))
    print(f"unique strings {len(texts)}", flush=True)
    vectors = embed_texts(sorted(texts))
    OUT.mkdir(parents=True, exist_ok=True)
    pieces = []
    for df in frames:
        study = df["study"].iloc[0]
        exp = df["exp"].iloc[0]
        table = summarize(df, vectors)
        table.insert(0, "experiment", exp)
        table.insert(0, "study", study)
        dest = OUT / f"{study}_{exp}_by_category_type.csv"
        table.to_csv(dest, index=False)
        pieces.append(table)
        print(study, exp, "kept", len(df), "->", dest.name, flush=True)
    pd.concat(pieces, ignore_index=True).to_csv(OUT / "all_by_category_type.csv", index=False)


if __name__ == "__main__":
    main()
