"""Cosine similarity of BLIP clean-image answers, experiments 1 and 2.

Experiment 3 has no clean images. Embeddings match build_answer_distance.py.
The median is computed only on pairs whose normalized strings differ.
pred_0.8 is the probe after sigma 0.8. The pipeline blurs only the mid bin.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_answer_distance as dist  # noqa: E402

VQA = dist.VQA
OUT = HERE
BINS = ["low", "mid", "high", "All"]
TYPES = ["Yes/No", "Number", "Other", "All"]
JOBS = [
    ("exp1", VQA / "multi_captioner" / "exp1" / "blip" / "rows_scored.csv"),
    ("exp2", VQA / "multi_captioner" / "exp2" / "blip" / "rows_scored.csv"),
]


def cosine(vectors, left, right) -> float:
    return float(np.dot(vectors[left], vectors[right]))


def summarize(frame: pd.DataFrame, vectors) -> pd.DataFrame:
    rows = []
    for risk in BINS:
        pool = frame if risk == "All" else frame[frame["bin"] == risk]
        for kind in TYPES:
            sub = pool if kind == "All" else pool[pool["answer_type"] == kind]
            n, same_ref, med_ref, n_ref = dist.pair_stats(
                sub, "original_prediction", "pred_ref", vectors
            )
            _, same_blur, med_blur, n_blur = dist.pair_stats(
                sub, "original_prediction", "pred_0.8", vectors
            )
            rows.append({
                "study": "blip",
                "experiment": frame["exp"].iloc[0] if len(frame) else "",
                "bin": risk,
                "answer_type": kind,
                "n": n,
                "same_ref_rate": same_ref,
                "sim_ref_median": med_ref,
                "sim_ref_n": n_ref,
                "same_blur_rate": same_blur,
                "sim_blur_median": med_blur,
                "sim_blur_n": n_blur,
            })
    return pd.DataFrame(rows)


def changed_rows(frame: pd.DataFrame, vectors) -> pd.DataFrame:
    rows = []
    for _, row in frame.iterrows():
        ori = dist.norm(row["original_prediction"])
        ref = dist.norm(row["pred_ref"])
        blur = dist.norm(row["pred_0.8"])
        if ori == ref and ori == blur:
            continue
        rows.append({
            "experiment": row["exp"],
            "image_id": row["image_id"],
            "bin": row["bin"],
            "answer_type": row["answer_type"],
            "original_prediction": ori,
            "pred_ref": ref,
            "pred_0.8": blur,
            "served_full": dist.norm(row["served_full"]),
            "blocked_full": row["blocked_full"],
            "sim_ref": 1.0 if ori == ref else cosine(vectors, ori, ref),
            "sim_blur": 1.0 if ori == blur else cosine(vectors, ori, blur),
        })
    return pd.DataFrame(rows)


def main():
    frames = []
    texts = set()
    for exp, path in JOBS:
        df = pd.read_csv(path, dtype={"image_id": str}, keep_default_na=False)
        df = df[df["label"].astype(int) == 0].copy()
        df["exp"] = exp
        df["answer_type"] = df["original_prediction"].map(dist.answer_type)
        frames.append(df)
        for col in ("original_prediction", "pred_ref", "pred_0.8", "served_full"):
            texts.update(df[col].map(dist.norm))
    print(f"unique strings {len(texts)}", flush=True)
    vectors = dist.embed_texts(sorted(texts))
    OUT.mkdir(parents=True, exist_ok=True)
    pieces = []
    changed = []
    for df in frames:
        exp = df["exp"].iloc[0]
        table = summarize(df, vectors)
        table["experiment"] = exp
        dest = OUT / f"blip_{exp}_clean_by_bin_type.csv"
        table.to_csv(dest, index=False)
        pieces.append(table)
        changed.append(changed_rows(df, vectors))
        print(exp, "clean", len(df), "->", dest.name, flush=True)
        show = table[table["answer_type"].isin(["Other", "All"])]
        print(show.to_string(index=False), flush=True)
    pd.concat(pieces, ignore_index=True).to_csv(OUT / "blip_clean_all_by_bin_type.csv", index=False)
    detail = pd.concat(changed, ignore_index=True)
    detail.to_csv(OUT / "blip_clean_changed.csv", index=False)
    print("changed rows", len(detail), flush=True)


if __name__ == "__main__":
    main()
