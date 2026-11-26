"""Finish the exp3 one-row replacement: reuse cached probes and captions for the other 999."""
import os
import subprocess
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = HERE.parent / "VQA" / "florence_victim"
PY = ROOT / "vlattack_gpu" / "Scripts" / "python.exe"
MC = HERE.parent / "multi_captioner"
NEW_ID = "132288000"
DROP_ID = "263163002"
COLS = ["image_id", "label", "sigma", "pred_ref", "pred_denoised", "ssim", "lpips", "brisque",
        "lap_ratio", "clip_sim", "iqp"]

ENV = os.environ.copy()
ENV["PYTHONPATH"] = str(Path(os.environ.get("LOCALAPPDATA", "")) / "vlattack_florence_pkgs")
ENV["FLORENCE_MODEL_ID"] = "microsoft/Florence-2-large-ft"
ENV["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
ENV["PYTHONUNBUFFERED"] = "1"
ENV["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
ENV["VQA_STUDY"] = "florence_victim"


def seed_probe_cache():
    frames = []
    for exp in ("exp1", "exp2", "exp3"):
        path = DATA / exp / "probe_gaussian_florence2.csv"
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        frames.append(df[COLS])
    seeded = pd.concat(frames, ignore_index=True).drop_duplicates(["image_id", "label", "sigma"], keep="last")
    out = DATA / "probe_cache_florence2.csv"
    seeded.to_csv(out, index=False, encoding="utf-8")
    print(f"seeded {out} rows={len(seeded)}", flush=True)


def merge_captions():
    folder = DATA / "exp3" / "florence2"
    old = pd.read_csv(folder / "caption_features.csv", dtype={"image_id": str}, keep_default_na=False)
    partial = pd.read_csv(folder / "caption_features_partial.csv", dtype={"image_id": str}, keep_default_na=False)
    if set(partial["image_id"]) != {NEW_ID}:
        raise SystemExit(f"partial captions are {sorted(partial['image_id'].unique())}")
    kept = old[~old["image_id"].isin([DROP_ID, NEW_ID])]
    merged = pd.concat([kept, partial], ignore_index=True)
    if len(merged) != 1000 or DROP_ID in set(merged["image_id"]) or NEW_ID not in set(merged["image_id"]):
        raise SystemExit(f"merged captions n={len(merged)}")
    merged.to_csv(folder / "caption_features.csv", index=False, encoding="utf-8")
    print(f"merged captions n={len(merged)}", flush=True)


def run(cmd, cwd):
    print("RUN", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=ENV, check=True)


def main():
    seed_probe_cache()
    py = str(PY)
    run([py, "-u", "run_probe.py", "--exp", "exp3", "--vqa-backend", "florence2"], MC)
    run([py, "-u", "run_captions.py", "--exp", "exp3", "--captioner", "florence2", "--image-id", NEW_ID], MC)
    merge_captions()
    run([py, "-u", "analyze.py", "--exp", "exp3", "--captioner", "florence2", "--vqa-backend", "florence2"], MC)
    run([py, "-u", "analyze.py", "--exp", "exp3", "--captioner", "florence2", "--vqa-backend", "florence2",
         "--output-tag", "sigma08_fixed", "--fix-theta-star", "0.8", "--fix-theta-det", "0.8"], MC)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
