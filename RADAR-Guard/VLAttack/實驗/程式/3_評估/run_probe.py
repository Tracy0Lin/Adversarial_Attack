"""Gaussian probe: VQA answers before/after blur (+ IQP / CLIP).

Default victim is BLIP VQA. Use --vqa-backend gemma3 to answer with LLM-API instead.
"""
import argparse
import csv
import time
from pathlib import Path

import pandas as pd

from common import DATA_ROOT, EXP2_DIR, KERNEL, RUN1000, SIGMAS, exp_dir, experiment_rows, gaussian, open_rgb
from vqa_backends import build_vqa

COLS = ["image_id", "label", "sigma", "pred_ref", "pred_denoised", "ssim", "lpips", "brisque",
        "lap_ratio", "clip_sim", "iqp"]


def cache_path(vqa_backend):
    return DATA_ROOT / ("probe_cache.csv" if vqa_backend == "blip" else f"probe_cache_{vqa_backend}.csv")


def probe_out_path(exp, vqa_backend):
    name = "probe_gaussian.csv" if vqa_backend == "blip" else f"probe_gaussian_{vqa_backend}.csv"
    return exp_dir(exp) / name


def seed_cache(cache):
    if cache.is_file() or cache.name != "probe_cache.csv":
        return
    frames = []
    for d in (RUN1000, EXP2_DIR):
        raw = pd.read_csv(d / "sss_per_theta_raw.csv", dtype=str, keep_default_na=False)
        frames.append(raw[COLS])
    seeded = pd.concat(frames).drop_duplicates(["image_id", "label", "sigma"])
    seeded.to_csv(cache, index=False, encoding="utf-8")
    print(f"seeded {cache} with {len(seeded)} rows from old SSS raw files")


def load_cache(cache):
    if not cache.is_file():
        return pd.DataFrame(columns=COLS), set()
    df = pd.read_csv(cache, dtype=str, keep_default_na=False)
    for c in COLS[5:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["label"] = df["label"].astype(int)
    df["sigma"] = df["sigma"].astype(float)
    df = df.drop_duplicates(["image_id", "label", "sigma"], keep="last")
    counts = df.groupby(["image_id", "label"])["sigma"].nunique()
    done = set(counts[counts == len(SIGMAS)].index)
    df = df[[k in done for k in zip(df["image_id"], df["label"])]]
    return df, done


def run(exp, device="cuda", limit=None, vqa_backend="blip"):
    cache = cache_path(vqa_backend)
    seed_cache(cache)
    _, done = load_cache(cache)
    rows = experiment_rows(exp)
    if limit:
        rows = rows[:limit]
    todo = [r for r in rows if (r["image_id"], r["label"]) not in done]
    print(f"[probe {exp} vqa={vqa_backend}] rows={len(rows)} cached={len(rows) - len(todo)} todo={len(todo)}",
          flush=True)

    if todo:
        from sss_denoise import QualityScorer

        vqa = build_vqa(vqa_backend, device)
        quality = QualityScorer(device)
        t0 = time.time()
        write_header = not cache.is_file() or cache.stat().st_size == 0
        with open(cache, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=COLS)
            if write_header:
                writer.writeheader()
            for i, r in enumerate(todo, 1):
                ref = open_rgb(r["path"])
                pred_ref = vqa.predict(ref, r["question"])
                for s in SIGMAS:
                    den = gaussian(ref, s, KERNEL)
                    m = quality.metrics(den, ref)
                    writer.writerow({
                        "image_id": r["image_id"], "label": r["label"], "sigma": s,
                        "pred_ref": pred_ref, "pred_denoised": vqa.predict(den, r["question"]),
                        "ssim": m["ssim"], "lpips": m["lpips"], "brisque": m["brisque"],
                        "lap_ratio": m["lap_ratio"], "clip_sim": m["clip_sim"],
                        "iqp": QualityScorer.iqp(m),
                    })
                fh.flush()
                if i % 5 == 0 or i == len(todo):
                    rate = (time.time() - t0) / i
                    print(f"[probe {exp}] {i}/{len(todo)} {rate:.2f}s/row eta {rate * (len(todo) - i) / 60:.1f} min",
                          flush=True)

    cache_df, _ = load_cache(cache)
    keys = pd.DataFrame([{"image_id": r["image_id"], "label": r["label"]} for r in rows])
    out = keys.merge(cache_df, on=["image_id", "label"], how="left")
    missing = out["pred_ref"].isna().sum()
    if missing:
        raise RuntimeError(f"{missing} probe rows missing for {exp} ({vqa_backend})")
    path = probe_out_path(exp, vqa_backend)
    out.to_csv(path, index=False, encoding="utf-8")
    print(f"Saved {path} ({len(out)} rows)", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--exp", nargs="+", required=True)
    p.add_argument("--vqa-backend", default="blip", choices=("blip", "git", "gemma3", "florence2"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    for e in args.exp:
        run(e, device=args.device, limit=args.limit, vqa_backend=args.vqa_backend)
