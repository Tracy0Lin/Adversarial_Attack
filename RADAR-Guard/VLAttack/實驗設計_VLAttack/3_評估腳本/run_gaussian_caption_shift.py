"""Gaussian-once caption shift for experiment 1 and experiment 2.

Each image is captioned once, blurred with a fixed 5x5 Gaussian, then captioned
again. The only feature is the distance between those two sentence embeddings.
Cosine and angular distance are strictly rank-reversed, so both are scored.
There is no +/-5 degree probe and no stage-1 / stage-2 split.

Samples are the image ids already used in each experiment's stage-1 table
(1000 pairs each), not every row of the attack log.
"""
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

THIS_FILE = Path(__file__).resolve()
BLIP_OUTPUT = THIS_FILE.parent
VQA_ROOT = BLIP_OUTPUT / "VQA"
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

EXPERIMENTS = {
    "exp1": VQA_ROOT / "steps100_alpha0.002_run1000",
    "exp2": VQA_ROOT / "exp2_success1000",
}
OUT_DIR = VQA_ROOT / "gaussian_caption_shift"
FEATURES = ("cosine", "angular")
KERNEL = 5
SIGMA = 0.8
N_SPLITS = 5
THRESHOLDS = np.round(np.arange(0.02, 0.99, 0.02), 2)
SEED = 42


def gaussian_blur(img, kernel=KERNEL, sigma=SIGMA):
    tensor = TF.to_tensor(img)
    blurred = TF.gaussian_blur(tensor, kernel_size=[kernel, kernel], sigma=[sigma, sigma])
    return TF.to_pil_image(blurred.clamp(0.0, 1.0))


def image_key(img):
    digest = hashlib.sha1()
    digest.update(f"{img.size[0]}x{img.size[1]}".encode())
    digest.update(img.tobytes())
    return digest.hexdigest()


def norm_qid(value):
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def photo_id(qid):
    """VQA question ids share one photo: question_id // 1000."""
    return str(int(norm_qid(qid)) // 1000)


def official_qids(experiment_root):
    table = pd.read_csv(experiment_root / "tcvad_stage1_4_academic_dataset.csv")
    if "method_type" in table.columns:
        table = table[table["method_type"] == "rotate_5"]
    qids = []
    seen = set()
    for value in table["image_id"].tolist():
        qid = norm_qid(value)
        if qid not in seen:
            seen.add(qid)
            qids.append(qid)
    return qids


def index_pairs(visual_dir, qids):
    names = os.listdir(visual_dir)
    ori = {}
    atk = {}
    for name in names:
        qid = name.split("_", 1)[0]
        if qid not in ori and "ori" in name:
            ori[qid] = name
        if qid not in atk and ("_2_atk.png" in name or name.endswith("_atk.png")):
            atk[qid] = name
    pairs = []
    missing = []
    for qid in qids:
        if qid not in ori or qid not in atk:
            missing.append(qid)
            continue
        pairs.append(
            {
                "image_id": qid,
                "ori_path": str((visual_dir / ori[qid]).resolve()),
                "atk_path": str((visual_dir / atk[qid]).resolve()),
            }
        )
    if missing:
        raise FileNotFoundError(f"Missing ori/atk for {len(missing)} ids, e.g. {missing[:5]}")
    return pairs


def load_cached_embedding(cache_dir, key):
    npy_path = cache_dir / f"{key}.npy"
    txt_path = cache_dir / f"{key}.txt"
    if not npy_path.is_file() or not txt_path.is_file():
        return None
    embedding = np.load(npy_path).astype(np.float32)
    caption = txt_path.read_text(encoding="utf-8")
    return embedding, caption


def store_embedding(cache_dir, key, embedding, caption):
    npy_path = cache_dir / f"{key}.npy"
    txt_path = cache_dir / f"{key}.txt"
    temporary = cache_dir / f"{key}.tmp.npy"
    np.save(temporary, np.asarray(embedding, dtype=np.float32))
    temporary.replace(npy_path)
    txt_path.write_text(caption, encoding="utf-8")


def embed_image(embedder, img, cache_dir, key):
    cached = load_cached_embedding(cache_dir, key)
    if cached is not None:
        return cached
    embedding, caption = embedder.embed_image(img)
    vector = embedding.squeeze(0).detach().float().cpu().numpy()
    store_embedding(cache_dir, key, vector, caption)
    return vector, caption


def open_rgb(path):
    with Image.open(path) as img:
        return img.convert("RGB").resize((384, 384))


def shift_row(embedder, cache_dir, image_path):
    raw = open_rgb(image_path)
    blurred = gaussian_blur(raw, KERNEL, SIGMA)
    raw_emb, raw_cap = embed_image(embedder, raw, cache_dir, image_key(raw))
    blur_emb, blur_cap = embed_image(embedder, blurred, cache_dir, image_key(blurred))
    cosine = float(np.dot(raw_emb, blur_emb))
    cosine = float(np.clip(cosine, -1.0, 1.0))
    angular = float(np.arccos(cosine) / np.pi)
    return {
        "cosine": cosine,
        "angular": angular,
        "caption_raw": raw_cap,
        "caption_blur": blur_cap,
        "gaussian_kernel": KERNEL,
        "gaussian_sigma": SIGMA,
    }


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
    for index, pair in enumerate(remaining, 1):
        for label, kind, path in (
            (0, "clean", pair["ori_path"]),
            (1, "attack", pair["atk_path"]),
        ):
            measured = shift_row(embedder, cache_dir, path)
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
        print(
            f"[{name}] pair {len(done) + index}/{len(pairs)} id={pair['image_id']}",
            flush=True,
        )
    return pd.read_csv(out_path) if rows else pd.DataFrame()


def separated_rates(labels, flagged):
    labels = np.asarray(labels)
    flagged = np.asarray(flagged).astype(bool)
    attack = labels == 1
    clean = labels == 0
    attack_n = int(attack.sum())
    clean_n = int(clean.sum())
    attack_flagged = int(flagged[attack].sum()) if attack_n else 0
    clean_flagged = int(flagged[clean].sum()) if clean_n else 0
    return {
        "attack_n": attack_n,
        "attack_flagged": attack_flagged,
        "attack_catch_rate": attack_flagged / attack_n if attack_n else np.nan,
        "clean_n": clean_n,
        "clean_flagged": clean_flagged,
        "clean_false_alarm_rate": clean_flagged / clean_n if clean_n else np.nan,
    }


def pick_threshold(labels, risk):
    best = None
    for threshold in THRESHOLDS:
        rates = separated_rates(labels, risk >= threshold)
        youden = rates["attack_catch_rate"] - rates["clean_false_alarm_rate"]
        key = (youden, -rates["clean_false_alarm_rate"], float(threshold))
        if best is None or key > best[0]:
            best = (key, float(threshold), rates)
    return best[1], best[2]


def score_feature(df, feature):
    work = df.reset_index(drop=True)
    x = work[[feature]].to_numpy(dtype=np.float64)
    y = work["label"].to_numpy(dtype=int)
    groups = work["image_id"].map(photo_id).to_numpy()
    scored_groups = groups
    n_groups = len(np.unique(groups))
    n_splits = min(N_SPLITS, n_groups)
    if n_splits < 2:
        raise RuntimeError(f"Need at least 2 image ids to score {feature}, got {n_groups}")
    splitter = GroupKFold(n_splits=n_splits)
    margin = np.zeros(len(work), dtype=np.float64)
    prediction = np.zeros(len(work), dtype=int)
    fold_threshold = np.zeros(len(work), dtype=np.float64)
    fold_flag = np.zeros(len(work), dtype=bool)
    chosen = []
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
        chosen.append({"fold": fold, "threshold": threshold, **test_rates})
        print(
            f"  fold {fold}/{n_splits} t={threshold:.2f} "
            f"catch={test_rates['attack_catch_rate']:.3f} "
            f"fpr={test_rates['clean_false_alarm_rate']:.3f}",
            flush=True,
        )
    risk = margin_to_risk_score(margin)
    scored = work.copy()
    scored["photo_id"] = scored_groups
    scored["feature_name"] = feature
    scored["oof_margin"] = margin
    scored["oof_risk"] = risk
    scored["oof_pred"] = prediction
    scored["fold_threshold"] = fold_threshold
    scored["fold_flag"] = fold_flag
    auroc_margin = float(roc_auc_score(y, margin))
    auroc_raw = float(roc_auc_score(y, work[feature].to_numpy(dtype=np.float64)))
    headline = separated_rates(y, fold_flag)
    default_rates = separated_rates(y, prediction == 1)
    summary = {
        "experiment": work["experiment"].iloc[0],
        "feature": feature,
        "n_questions": int(work["image_id"].nunique()),
        "n_photos": int(len(np.unique(groups))),
        "n_splits": n_splits,
        "auroc_svm_margin": auroc_margin,
        "auroc_raw_feature": auroc_raw,
        "fold_threshold_mean": float(np.mean(fold_threshold)),
        **{f"fold_{key}": value for key, value in headline.items()},
        **{f"svm_default_{key}": value for key, value in default_rates.items()},
    }
    sweep_rows = []
    for threshold in THRESHOLDS:
        rates = separated_rates(y, risk >= threshold)
        sweep_rows.append(
            {
                "experiment": summary["experiment"],
                "feature": feature,
                "threshold": float(threshold),
                "source": "pooled_oof_risk",
                **rates,
            }
        )
    return scored, summary, sweep_rows, chosen


def write_summary(summaries):
    lines = [
        "高斯一次後的 caption 位移。每張圖糊前一句、糊後一句（5×5，σ=0.8）。",
        "沒有 ±5° 探測，也不分第一階段 / 第二階段。",
        "SVM 是 RBF、C=1、class_weight=balanced。GroupKFold 按照片分組（question_id // 1000），同一張圖的多個問題只進同一折。",
        "攔截率只看攻擊圖，誤攔率只看乾淨圖。",
        "fold 門檻在該折訓練集上取 攔截率−誤攔率 最大，再套到測試折。",
        "cosine 與 angular = arccos(cosine)/pi 排名相反，所以兩者的原始 AUROC 相加為 1。",
        "門檻掃描是把各折的 out-of-fold 風險分放在一起看，用來畫曲線；主結果用 fold 門檻那一列。",
        "",
    ]
    for row in summaries:
        lines.append(
            f"{row['experiment']} | {row['feature']} | "
            f"題數={row['n_questions']} 照片={row['n_photos']} | "
            f"AUROC(SVM)={row['auroc_svm_margin']:.4f} | "
            f"AUROC(raw)={row['auroc_raw_feature']:.4f} | "
            f"fold攔截={row['fold_attack_flagged']}/{row['fold_attack_n']} "
            f"({row['fold_attack_catch_rate']:.1%}) | "
            f"fold誤攔={row['fold_clean_flagged']}/{row['fold_clean_n']} "
            f"({row['fold_clean_false_alarm_rate']:.1%}) | "
            f"SVM預設攔截={row['svm_default_attack_catch_rate']:.1%} "
            f"誤攔={row['svm_default_clean_false_alarm_rate']:.1%}"
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
        for feature in FEATURES:
            print(f"\n=== {experiment} / {feature} ===", flush=True)
            scored, summary, sweep, _ = score_feature(frame, feature)
            scored_frames.append(scored)
            summaries.append(summary)
            sweep_rows.extend(sweep)
    pd.concat(scored_frames, ignore_index=True).to_csv(
        OUT_DIR / "oof_scores.csv", index=False, encoding="utf-8"
    )
    pd.DataFrame(summaries).to_csv(OUT_DIR / "operating_point.csv", index=False, encoding="utf-8")
    pd.DataFrame(sweep_rows).to_csv(OUT_DIR / "threshold_sweep.csv", index=False, encoding="utf-8")
    write_summary(summaries)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Gaussian-once caption shift SVM")
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
            path = OUT_DIR / f"{name}_features.csv"
            frames.append(pd.read_csv(path))
    else:
        device = resolve_device(args.device)
        caption_ckpt = BLIP_ROOT / "checkpoints" / "model_base_caption_capfilt_large.pth"
        print(f"device={device}", flush=True)
        caption_model, processor = load_caption_model(str(caption_ckpt.resolve()), device)
        tokenizer, sent_model = load_sentence_encoder(device)
        embedder = CaptionEmbedder(caption_model, processor, tokenizer, sent_model, device)
        for name in args.experiments:
            root = EXPERIMENTS[name].resolve()
            frame = extract_experiment(name, root, embedder, cache_dir.resolve(), limit=args.limit)
            frames.append(frame)
    evaluate_all(frames)


if __name__ == "__main__":
    main()
