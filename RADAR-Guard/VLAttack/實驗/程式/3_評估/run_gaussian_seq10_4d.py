"""Ten cumulative Gaussian blurs, then four caption-consistency statistics.

Each image is captioned, then blurred 10 times in sequence (each pass blurs the
previous result). The original plus those 10 frames give 11 sentence embeddings.
d_center, d_iqr, d_tail and d_q95 are computed twice: once on cosine-to-centroid,
once on angular distance. Experiment 1 and experiment 2 each train their own
4-D RBF SVM. There is no +/-5 degree probe and no stage split.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

THIS_FILE = Path(__file__).resolve()
BLIP_OUTPUT = THIS_FILE.parent
sys.path.insert(0, str(BLIP_OUTPUT))

from evaluate_best_stage1_new import (  # noqa: E402
    BLIP_ROOT,
    CaptionEmbedder,
    load_caption_model,
    load_sentence_encoder,
    margin_to_risk_score,
    resolve_device,
    set_seed,
)
from run_gaussian_caption_shift import (  # noqa: E402
    EXPERIMENTS,
    KERNEL,
    N_SPLITS,
    SIGMA,
    THRESHOLDS,
    gaussian_blur,
    image_key,
    index_pairs,
    load_cached_embedding,
    norm_qid,
    official_qids,
    open_rgb,
    photo_id,
    separated_rates,
    store_embedding,
)

OUT_DIR = BLIP_OUTPUT / "VQA" / "gaussian_seq10_4d"
OLD_CACHE = BLIP_OUTPUT / "VQA" / "gaussian_caption_shift" / "embed_cache"
BLUR_STEPS = 10
COS_COLS = ["d_center_cos", "d_iqr_cos", "d_tail_cos", "d_q95_cos"]
ANG_COLS = ["d_center_ang", "d_iqr_ang", "d_tail_ang", "d_q95_ang"]
FEATURE_SETS = (("cosine_4d", COS_COLS), ("angular_4d", ANG_COLS))
SEED = 42


def four_stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "d_center": float(np.mean(values)),
        "d_iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
        "d_tail": float(np.percentile(values, 90) - np.percentile(values, 50)),
        "d_q95": float(np.percentile(values, 95)),
    }


def blur_chain(raw, steps=BLUR_STEPS):
    frames = [raw]
    current = raw
    for _ in range(steps):
        current = gaussian_blur(current, KERNEL, SIGMA)
        frames.append(current)
    return frames


def lookup_embedding(cache_dir, key):
    cached = load_cached_embedding(cache_dir, key)
    if cached is not None:
        return cached
    if OLD_CACHE.is_dir():
        return load_cached_embedding(OLD_CACHE, key)
    return None


def caption_images(embedder, images):
    batch = torch.stack([embedder.processor(img.convert("RGB")) for img in images]).to(embedder.device)
    with torch.no_grad():
        texts = embedder.caption_model.generate(batch, sample=False, num_beams=3)
    if len(texts) != len(images):
        raise RuntimeError(f"caption batch returned {len(texts)} texts for {len(images)} images")
    return list(texts)


def encode_texts(embedder, texts):
    encoded = embedder.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(embedder.device)
    with torch.no_grad():
        output = embedder.sent_model(**encoded)
        token_embeddings = output.last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
        pooled = torch.sum(token_embeddings * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
    return pooled.detach().float().cpu().numpy()


def embed_frames(embedder, frames, cache_dir):
    vectors = [None] * len(frames)
    captions = [None] * len(frames)
    keys = [image_key(img) for img in frames]
    missing = []
    for index, key in enumerate(keys):
        cached = lookup_embedding(cache_dir, key)
        if cached is None:
            missing.append(index)
            continue
        vectors[index], captions[index] = cached
    if missing:
        try:
            texts = caption_images(embedder, [frames[index] for index in missing])
        except RuntimeError:
            texts = [embedder.caption(frames[index]) for index in missing]
        embeds = encode_texts(embedder, texts)
        for index, text, vector in zip(missing, texts, embeds):
            store_embedding(cache_dir, keys[index], vector, text)
            vectors[index] = vector
            captions[index] = text
    return vectors, captions, len(missing)


def chain_features(embedder, cache_dir, image_path):
    raw = open_rgb(image_path)
    frames = blur_chain(raw, BLUR_STEPS)
    vectors, captions, misses = embed_frames(embedder, frames, cache_dir)
    stacked = np.stack(vectors).astype(np.float64)
    center = stacked.mean(axis=0)
    center = center / max(np.linalg.norm(center), 1e-12)
    cosine = np.clip(stacked @ center, -1.0, 1.0)
    angular = np.arccos(cosine) / np.pi
    cos_stats = four_stats(cosine)
    ang_stats = four_stats(angular)
    raw_arr = np.asarray(frames[0], dtype=np.float32)
    last_arr = np.asarray(frames[-1], dtype=np.float32)
    pixel_shift = float(np.mean(np.abs(raw_arr - last_arr)) / 255.0)
    row = {
        "gaussian_kernel": KERNEL,
        "gaussian_sigma": SIGMA,
        "blur_steps": BLUR_STEPS,
        "n_captions": len(captions),
        "n_unique_captions": int(len(set(captions))),
        "pixel_shift_last": pixel_shift,
        "captions": " || ".join(captions),
    }
    for name, value in cos_stats.items():
        row[f"{name}_cos"] = value
    for name, value in ang_stats.items():
        row[f"{name}_ang"] = value
    return row, misses


def extract_experiment(name, root, embedder, cache_dir, limit=None):
    qids = official_qids(root)
    pairs = index_pairs(root / "visual_results", qids)
    if limit:
        pairs = pairs[:limit]
    out_path = OUT_DIR / f"{name}_features.csv"
    rows = []
    done = set()
    if out_path.is_file():
        prev = pd.read_csv(out_path)
        if not prev.empty:
            rows = prev.to_dict("records")
            done = set(prev["image_id"].astype(str).map(norm_qid))
            print(f"[{name}] resume {len(done)} / {len(pairs)} pairs", flush=True)
    remaining = [pair for pair in pairs if pair["image_id"] not in done]
    print(f"[{name}] {len(remaining)} pairs to caption, {len(pairs)} total", flush=True)
    started = time.time()
    new_captions = 0
    for index, pair in enumerate(remaining, 1):
        for label, kind, path in (
            (0, "clean", pair["ori_path"]),
            (1, "attack", pair["atk_path"]),
        ):
            measured, misses = chain_features(embedder, cache_dir, path)
            new_captions += misses
            rows.append(
                {
                    "experiment": name,
                    "image_id": pair["image_id"],
                    "label": label,
                    "kind": kind,
                    **measured,
                }
            )
        pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8")
        elapsed = time.time() - started
        print(
            f"[{name}] pair {len(done) + index}/{len(pairs)} id={pair['image_id']} "
            f"new_captions={new_captions} elapsed={elapsed:.0f}s",
            flush=True,
        )
    return pd.read_csv(out_path) if rows else pd.DataFrame()


def score_feature_set(df, feature_name, columns):
    work = df.reset_index(drop=True)
    x = work[list(columns)].to_numpy(dtype=np.float64)
    y = work["label"].to_numpy(dtype=int)
    groups = work["image_id"].map(photo_id).to_numpy()
    n_groups = len(np.unique(groups))
    n_splits = min(N_SPLITS, n_groups)
    if n_splits < 2:
        raise RuntimeError(f"Need at least 2 photos to score {feature_name}, got {n_groups}")
    splitter = GroupKFold(n_splits=n_splits)
    margin = np.zeros(len(work), dtype=np.float64)
    prediction = np.zeros(len(work), dtype=int)
    fold_threshold = np.zeros(len(work), dtype=np.float64)
    fold_flag = np.zeros(len(work), dtype=bool)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(x, y, groups), 1):
        clf = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svm", SVC(kernel="rbf", C=1.0, class_weight="balanced")),
            ]
        )
        clf.fit(x[train_idx], y[train_idx])
        train_margin = clf.decision_function(x[train_idx])
        test_margin = clf.decision_function(x[test_idx])
        margin[test_idx] = test_margin
        prediction[test_idx] = clf.predict(x[test_idx])
        threshold, _ = pick_threshold(y[train_idx], margin_to_risk_score(train_margin))
        test_risk = margin_to_risk_score(test_margin)
        fold_threshold[test_idx] = threshold
        fold_flag[test_idx] = test_risk >= threshold
        test_rates = separated_rates(y[test_idx], test_risk >= threshold)
        print(
            f"  fold {fold}/{n_splits} t={threshold:.2f} "
            f"catch={test_rates['attack_catch_rate']:.3f} "
            f"fpr={test_rates['clean_false_alarm_rate']:.3f}",
            flush=True,
        )
    risk = margin_to_risk_score(margin)
    keep = ["experiment", "image_id", "label", "kind", "n_unique_captions", "pixel_shift_last", *columns]
    scored = work[keep].copy()
    scored["photo_id"] = groups
    scored["feature_name"] = feature_name
    scored["oof_margin"] = margin
    scored["oof_risk"] = risk
    scored["oof_pred"] = prediction
    scored["fold_threshold"] = fold_threshold
    scored["fold_flag"] = fold_flag
    headline = separated_rates(y, fold_flag)
    default_rates = separated_rates(y, prediction == 1)
    summary = {
        "experiment": work["experiment"].iloc[0],
        "feature": feature_name,
        "n_questions": int(work["image_id"].nunique()),
        "n_photos": int(n_groups),
        "n_splits": n_splits,
        "auroc_svm_margin": float(roc_auc_score(y, margin)),
        "fold_threshold_mean": float(np.mean(fold_threshold)),
        "mean_unique_captions_clean": float(work.loc[work["label"] == 0, "n_unique_captions"].mean()),
        "mean_unique_captions_attack": float(work.loc[work["label"] == 1, "n_unique_captions"].mean()),
        "mean_pixel_shift": float(work["pixel_shift_last"].mean()),
        **{f"fold_{key}": value for key, value in headline.items()},
        **{f"svm_default_{key}": value for key, value in default_rates.items()},
    }
    sweep_rows = []
    for threshold in THRESHOLDS:
        rates = separated_rates(y, risk >= threshold)
        sweep_rows.append(
            {
                "experiment": summary["experiment"],
                "feature": feature_name,
                "threshold": float(threshold),
                "source": "pooled_oof_risk",
                **rates,
            }
        )
    return scored, summary, sweep_rows


def pick_threshold(labels, risk):
    best = None
    for threshold in THRESHOLDS:
        rates = separated_rates(labels, risk >= threshold)
        youden = rates["attack_catch_rate"] - rates["clean_false_alarm_rate"]
        key = (youden, -rates["clean_false_alarm_rate"], float(threshold))
        if best is None or key > best[0]:
            best = (key, float(threshold), rates)
    return best[1], best[2]


def write_summary(summaries):
    lines = [
        "連糊十次後的四維 caption 一致性。5×5，σ=0.8，每一輪糊在上一輪的結果上。",
        "每張圖 11 句：原圖 + 10 輪。四個統計是這 11 個向量到中心的距離。",
        "cosine_4d 把距離留在 cosine；angular_4d 用 arccos(cosine)/pi。",
        "SVM 是 RBF、C=1、class_weight=balanced。GroupKFold 按照片分組（question_id // 1000）。",
        "攔截率只看攻擊題，誤攔率只看乾淨題。主結果用 SVM 預設切點（決策值 >= 0）。",
        "",
    ]
    for row in summaries:
        lines.append(
            f"{row['experiment']} | {row['feature']} | "
            f"題數={row['n_questions']} 照片={row['n_photos']} | "
            f"AUROC={row['auroc_svm_margin']:.4f} | "
            f"SVM攔截={row['svm_default_attack_flagged']}/{row['svm_default_attack_n']} "
            f"({row['svm_default_attack_catch_rate']:.1%}) | "
            f"SVM誤攔={row['svm_default_clean_flagged']}/{row['svm_default_clean_n']} "
            f"({row['svm_default_clean_false_alarm_rate']:.1%}) | "
            f"fold攔截={row['fold_attack_catch_rate']:.1%} "
            f"fold誤攔={row['fold_clean_false_alarm_rate']:.1%} | "
            f"獨特caption 乾淨={row['mean_unique_captions_clean']:.2f} "
            f"攻擊={row['mean_unique_captions_attack']:.2f}"
        )
    text = "\n".join(lines) + "\n"
    (OUT_DIR / "summary.txt").write_text(text, encoding="utf-8")
    print("\n" + text, flush=True)


def evaluate_all(frames):
    scored_frames = []
    summaries = []
    sweep_rows = []
    for frame in frames:
        experiment = frame["experiment"].iloc[0]
        for feature_name, columns in FEATURE_SETS:
            print(f"\n=== {experiment} / {feature_name} ===", flush=True)
            scored, summary, sweep = score_feature_set(frame, feature_name, columns)
            scored_frames.append(scored)
            summaries.append(summary)
            sweep_rows.extend(sweep)
    pd.concat(scored_frames, ignore_index=True).to_csv(OUT_DIR / "oof_scores.csv", index=False, encoding="utf-8")
    pd.DataFrame(summaries).to_csv(OUT_DIR / "operating_point.csv", index=False, encoding="utf-8")
    pd.DataFrame(sweep_rows).to_csv(OUT_DIR / "threshold_sweep.csv", index=False, encoding="utf-8")
    write_summary(summaries)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Sequential Gaussian x10 four-stat SVM")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--experiments", nargs="+", default=["exp1", "exp2"], choices=list(EXPERIMENTS))
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cache_dir = OUT_DIR / "embed_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    frames = []
    if args.eval_only:
        for name in args.experiments:
            frames.append(pd.read_csv(OUT_DIR / f"{name}_features.csv"))
    else:
        device = resolve_device(args.device)
        caption_ckpt = BLIP_ROOT / "checkpoints" / "model_base_caption_capfilt_large.pth"
        print(f"device={device} blur_steps={BLUR_STEPS} kernel={KERNEL} sigma={SIGMA}", flush=True)
        caption_model, processor = load_caption_model(str(caption_ckpt.resolve()), device)
        tokenizer, sent_model = load_sentence_encoder(device)
        embedder = CaptionEmbedder(caption_model, processor, tokenizer, sent_model, device)
        for name in args.experiments:
            frame = extract_experiment(name, EXPERIMENTS[name].resolve(), embedder, cache_dir.resolve(), args.limit)
            frames.append(frame)
    evaluate_all(frames)


if __name__ == "__main__":
    main()
