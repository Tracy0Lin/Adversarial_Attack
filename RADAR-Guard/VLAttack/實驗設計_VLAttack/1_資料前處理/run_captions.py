"""Captioner-dependent features for one (experiment, captioner).

Stage 1: original + 10 rotations (each -5 or +5 degrees, same RNG as the original
stage 1) on the 384x384 image -> MiniLM angular distances -> four statistics.

Stage 2 semantic term: caption of the native-size image vs its 5x5 Gaussian blur at
every sigma (MSS, used in SP = geo-mean(CLIP, MSS) when picking theta*).

Writes <exp>/<captioner>/caption_features.csv. Resumable through the caption cache.
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch

from captioners import SentenceEncoder, build_captioner
from common import (
    SIGMAS, STAGE1_SIZE, exp_dir, experiment_rows, gaussian, open_rgb, rotate, rotation_angles,
)

FEATURES = ["d_center", "d_iqr", "d_tail", "d_q95"]


def four_stats(emb):
    e_bar = torch.nn.functional.normalize(emb.mean(0, keepdim=True), dim=-1)
    cos = np.clip((emb @ e_bar.T).squeeze(-1).numpy(), -1.0, 1.0)
    delta = np.arccos(cos) / np.pi
    return {
        "d_center": float(np.mean(delta)),
        "d_iqr": float(np.percentile(delta, 75) - np.percentile(delta, 25)),
        "d_tail": float(np.percentile(delta, 90) - np.percentile(delta, 50)),
        "d_q95": float(np.percentile(delta, 95)),
    }


def views_for(row):
    native = open_rgb(row["path"])
    small = native.resize((STAGE1_SIZE, STAGE1_SIZE))
    angles = rotation_angles(row["image_id"], row["label"])
    rot = {a: rotate(small, a) for a in set(angles)}
    blurred = [gaussian(native, s) for s in SIGMAS]
    return small, angles, rot, native, blurred


def run(exp, captioner_name, device="cuda", chunk=24, limit=None, image_ids=None):
    rows = experiment_rows(exp)
    if image_ids:
        want = {str(x) for x in image_ids}
        rows = [r for r in rows if r["image_id"] in want]
        missing = want - {r["image_id"] for r in rows}
        if missing:
            raise SystemExit(f"image ids not in {exp}: {sorted(missing)}")
    if limit:
        rows = rows[:limit]
    out_dir = exp_dir(exp, captioner_name)
    cap = build_captioner(captioner_name, device)
    enc = SentenceEncoder(device)
    print(f"[{exp}/{captioner_name}] rows={len(rows)}", flush=True)

    records = []
    t0 = time.time()
    for start in range(0, len(rows), chunk):
        batch = rows[start:start + chunk]
        prepared = [views_for(r) for r in batch]
        images = []
        for small, angles, rot, native, blurred in prepared:
            images += [small, rot[-5.0] if -5.0 in rot else small, rot[5.0] if 5.0 in rot else small,
                       native, *blurred]
        caps = cap.caption(images)
        width = 4 + len(SIGMAS)
        for i, (r, (small, angles, rot, native, blurred)) in enumerate(zip(batch, prepared)):
            c = caps[i * width:(i + 1) * width]
            c0, c_neg, c_pos, c_native, c_blur = c[0], c[1], c[2], c[3], c[4:]
            seq = [c0] + [c_neg if a < 0 else c_pos for a in angles]
            emb = enc.encode(seq)
            rec = {
                "image_id": r["image_id"],
                "photo_id": r["photo_id"],
                "label": r["label"],
                **four_stats(emb),
                "rewrite_rate": float(np.mean([s != c0 for s in seq[1:]])),
                "n_minus5": int(sum(a < 0 for a in angles)),
                "caption_orig": c0,
                "caption_rot_minus5": c_neg,
                "caption_rot_plus5": c_pos,
                "caption_native": c_native,
            }
            e_ref = enc.encode([c_native])
            e_blur = enc.encode(list(c_blur))
            mss = (e_blur @ e_ref.T).squeeze(-1).clamp(0.0, 1.0).numpy()
            for s, cb, m in zip(SIGMAS, c_blur, mss):
                rec[f"caption_blur_{s}"] = cb
                rec[f"mss_{s}"] = float(m)
            records.append(rec)
        done = start + len(batch)
        rate = (time.time() - t0) / done
        print(f"[{exp}/{captioner_name}] {done}/{len(rows)}  {rate:.2f}s/row  eta {rate * (len(rows) - done) / 60:.1f} min",
              flush=True)

    df = pd.DataFrame(records)
    if image_ids:
        name = "caption_features_partial.csv"
    elif limit:
        name = f"caption_features_limit{limit}.csv"
    else:
        name = "caption_features.csv"
    df.to_csv(out_dir / name, index=False, encoding="utf-8")
    print(f"Saved {out_dir / name}", flush=True)
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--exp", required=True, nargs="+")
    p.add_argument("--captioner", required=True, nargs="+")
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--image-id", nargs="*", default=None,
                   help="Caption only these ids and write caption_features_partial.csv")
    args = p.parse_args()
    for e in args.exp:
        for c in args.captioner:
            run(e, c, device=args.device, limit=args.limit, image_ids=args.image_id)
