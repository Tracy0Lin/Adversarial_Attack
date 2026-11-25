"""TC-VAD stage-1 for dual-defense 200-question log.

Pairs come from attack_dual_defense_log.json (not filename-sorted first 200).
Captions use the local BLIP decoder checkpoint. Four-d angular features are
scored with GroupKFold P1 SVM; out-of-fold risk_score is written to CSV.
"""
import json
import os
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

warnings.filterwarnings("ignore", category=UserWarning)

THIS_FILE = Path(__file__).resolve()
EXPERIMENT_ROOT = THIS_FILE.parent
BLIP_ROOT = THIS_FILE.parents[3]
sys.path.insert(0, str(BLIP_ROOT))

METHODS = ["rotate_5", "scale_crop_mild"]
FEATURE_COLS = ["d_center", "d_iqr", "d_tail", "d_q95"]
CAPTION_MEAN = (0.48145466, 0.4578275, 0.40821073)
CAPTION_STD = (0.26862954, 0.26130258, 0.27577711)
K_DEFAULT = 10
SEED_DEFAULT = 42
LOW_RISK = 0.3
HIGH_RISK = 0.7
PARTIAL_CSV = "tcvad_stage1_partial.csv"
OUTPUT_CSV = "tcvad_stage1_4_academic_dataset.csv"
METRICS_CSV = "tcvad_stage1_metrics.csv"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested="cuda"):
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        return "cpu"
    return requested


def margin_to_risk_score(decision_margin, T=1.0):
    x = np.clip(np.asarray(decision_margin, dtype=np.float64) / T, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def risk_bin(score):
    if score <= LOW_RISK:
        return "low"
    if score >= HIGH_RISK:
        return "high"
    return "mid"


def load_log_qids(log_path):
    with open(log_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    qids = []
    seen = set()
    for item in data:
        qid = str(item["question_id"])
        if qid not in seen:
            qids.append(qid)
            seen.add(qid)
    return qids


def collect_pairs_from_log(visual_dir, log_path):
    qids = load_log_qids(log_path)
    names = os.listdir(visual_dir)
    pairs = []
    missing = []
    for qid in qids:
        ori = [f for f in names if f.startswith(qid + "_") and "ori" in f]
        atk = [f for f in names if f.startswith(qid + "_") and ("_2_atk.png" in f or f.endswith("_atk.png"))]
        if not ori or not atk:
            missing.append(qid)
            continue
        pairs.append(
            {
                "image_id": qid,
                "ori_path": os.path.join(visual_dir, ori[0]),
                "atk_path": os.path.join(visual_dir, atk[0]),
            }
        )
    if missing:
        raise FileNotFoundError(f"Missing ori/atk pairs for {len(missing)} log qids, e.g. {missing[:5]}")
    print(f"Aligned {len(pairs)} pairs to {os.path.basename(log_path)}")
    return pairs


def apply_transform(pil_img, method_type, rng):
    if method_type == "rotate_5":
        angle = rng.choice([-5.0, 5.0])
        return TF.rotate(pil_img, angle, interpolation=InterpolationMode.BILINEAR, fill=0)
    if method_type == "scale_crop_mild":
        scale = rng.uniform(0.95, 1.0)
        w, h = pil_img.size
        new_w = max(10, int(w * scale))
        new_h = max(10, int(h * scale))
        cropped = TF.center_crop(pil_img, [new_h, new_w])
        return cropped.resize((w, h), Image.BILINEAR)
    return pil_img.copy()


def load_caption_model(ckpt_path, device):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"BLIP caption checkpoint missing: {ckpt_path}")
    os.chdir(BLIP_ROOT)
    from models.blip import blip_decoder

    model = blip_decoder(pretrained=ckpt_path, image_size=384, vit="base")
    model = model.to(device)
    model.eval()
    processor = transforms.Compose(
        [
            transforms.Resize((384, 384), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(CAPTION_MEAN, CAPTION_STD),
        ]
    )
    print(f"Loaded BLIP caption decoder from {ckpt_path}")
    return model, processor


def load_sentence_encoder(device):
    name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name).to(device)
    model.eval()
    print(f"Loaded sentence encoder {name}")
    return tokenizer, model


def encode_sentence(text, tokenizer, model, device):
    encoded = tokenizer([text], padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model(**encoded)
        token_embeddings = output.last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
        pooled = torch.sum(token_embeddings * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
        return F.normalize(pooled, p=2, dim=-1)


class CaptionEmbedder:
    def __init__(self, caption_model, processor, tokenizer, sent_model, device):
        self.caption_model = caption_model
        self.processor = processor
        self.tokenizer = tokenizer
        self.sent_model = sent_model
        self.device = device

    def caption(self, pil_img):
        image = self.processor(pil_img.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            texts = self.caption_model.generate(image, sample=False, num_beams=3)
        return texts[0] if texts else ""

    def embed_image(self, pil_img):
        text = self.caption(pil_img)
        return encode_sentence(text, self.tokenizer, self.sent_model, self.device), text


def extract_four_dimensional_features(embedder, pil_img, method_type, k, seed, image_id, channel_tag):
    embeddings = []
    captions = []
    e0, cap0 = embedder.embed_image(pil_img)
    embeddings.append(e0)
    captions.append(cap0)
    for i in range(k):
        rng = random.Random(f"{seed}|{image_id}|{channel_tag}|{i}")
        aug = apply_transform(pil_img, method_type, rng)
        ei, capi = embedder.embed_image(aug)
        embeddings.append(ei)
        captions.append(capi)
    e_tensor = torch.cat(embeddings, dim=0)
    e_bar = F.normalize(torch.mean(e_tensor, dim=0, keepdim=True), dim=-1)
    cosine_values = torch.mm(e_tensor, e_bar.t()).squeeze(-1).cpu().numpy()
    safe_cosine = np.clip(cosine_values, -1.0, 1.0)
    delta = np.arccos(safe_cosine) / np.pi
    return {
        "d_center": float(np.mean(delta)),
        "d_iqr": float(np.percentile(delta, 75) - np.percentile(delta, 25)),
        "d_tail": float(np.percentile(delta, 90) - np.percentile(delta, 50)),
        "d_q95": float(np.percentile(delta, 95)),
        "n_captions": len(captions),
        "captions": captions,
    }


def attach_oof_risk_scores(df):
    df = df.copy()
    df["risk_score"] = np.nan
    df["risk_bin"] = ""
    for method in METHODS:
        mask = df["method_type"] == method
        sub = df.loc[mask]
        if sub.empty:
            continue
        X = sub[FEATURE_COLS].values
        y = sub["label"].values
        groups = sub["image_id"].astype(str).values
        n_splits = min(3, len(np.unique(groups)))
        gkf = GroupKFold(n_splits=n_splits)
        scores = np.zeros(len(sub), dtype=np.float64)
        for train_idx, test_idx in gkf.split(X, y, groups=groups):
            clf = Pipeline(
                [("scaler", StandardScaler()), ("svm", SVC(kernel="rbf", C=1.0, class_weight="balanced"))]
            )
            clf.fit(X[train_idx], y[train_idx])
            margins = clf.decision_function(X[test_idx])
            scores[test_idx] = margin_to_risk_score(margins)
        df.loc[sub.index, "risk_score"] = scores
        df.loc[sub.index, "risk_bin"] = [risk_bin(s) for s in scores]
    return df


def evaluate_metrics(df):
    rows = []
    print("\n" + "=" * 110)
    print("【第一階段數據：同模型可分性與基準消融對比 (Stage 1)】")
    print("=" * 110)
    print(f"{'手法':<18} | {'配置':<16} | {'模型':<18} | {'AUROC':<8} | {'Recall':<8} | {'Precision':<8} | {'F1':<8} | {'中風險率'}")
    print("-" * 110)
    configs = [
        {"id": "B2", "features": ["d_center"], "model": "LogisticRegression"},
        {"id": "P1", "features": FEATURE_COLS, "model": "SVC_RBF"},
    ]
    for method in METHODS:
        df_sub = df[df["method_type"] == method].reset_index(drop=True)
        y = df_sub["label"].values
        groups = df_sub["image_id"].astype(str).values
        for cfg in configs:
            X = df_sub[cfg["features"]].values
            n_splits = min(3, len(np.unique(groups)))
            gkf = GroupKFold(n_splits=n_splits)
            fold_auroc, fold_recall, fold_prec, fold_f1, fold_med = [], [], [], [], []
            for train_idx, test_idx in gkf.split(X, y, groups=groups):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                if cfg["model"] == "LogisticRegression":
                    clf = Pipeline(
                        [("scaler", StandardScaler()), ("lr", LogisticRegression(class_weight="balanced", max_iter=1000))]
                    )
                    clf.fit(X_train, y_train)
                    probs = clf.predict_proba(X_test)[:, 1]
                    preds = clf.predict(X_test)
                    scores = probs
                else:
                    clf = Pipeline(
                        [("scaler", StandardScaler()), ("svm", SVC(kernel="rbf", C=1.0, class_weight="balanced"))]
                    )
                    clf.fit(X_train, y_train)
                    margins = clf.decision_function(X_test)
                    preds = clf.predict(X_test)
                    scores = margin_to_risk_score(margins)
                    probs = scores
                if len(np.unique(y_test)) > 1:
                    fold_auroc.append(roc_auc_score(y_test, probs))
                fold_recall.append(recall_score(y_test, preds, zero_division=0))
                fold_prec.append(precision_score(y_test, preds, zero_division=0))
                fold_f1.append(f1_score(y_test, preds, zero_division=0))
                fold_med.append(np.mean((scores > LOW_RISK) & (scores < HIGH_RISK)))
            row = {
                "method_type": method,
                "config": cfg["id"],
                "auroc": float(np.mean(fold_auroc)) if fold_auroc else float("nan"),
                "recall": float(np.mean(fold_recall)),
                "precision": float(np.mean(fold_prec)),
                "f1": float(np.mean(fold_f1)),
                "medium_rate": float(np.mean(fold_med)),
            }
            rows.append(row)
            print(
                f"{method:<18} | {cfg['id']:<16} | {cfg['model']:<18} | "
                f"{row['auroc']:.4f}   | {row['recall']:.4f}   | {row['precision']:.4f}   | "
                f"{row['f1']:.4f}   | {row['medium_rate']:.2%}"
            )
    print("-" * 110)
    return pd.DataFrame(rows)


def generate_plots(df, save_dir):
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(save_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    df_plot = df[df["method_type"] == "rotate_5"].copy()
    df_plot["類別"] = df_plot["label"].map({0: "Clean (乾淨影像)", 1: "VLAttack (攻擊影像)"})
    palette = {"Clean (乾淨影像)": "#2ecc71", "VLAttack (攻擊影像)": "#e74c3c"}
    features_info = [
        ("d_center", "圖一：中心不一致度 (D_center)", "D_center", "fig1_d_center.png"),
        ("d_iqr", "圖二：語意離散程度 (D_IQR)", "D_IQR", "fig2_d_iqr.png"),
        ("d_tail", "圖三：尾端轉換敏感度 (D_tail)", "D_tail", "fig3_d_tail.png"),
        ("d_q95", "圖四：高尾端角距離 (D_q95)", "D_q95", "fig4_d_q95.png"),
    ]
    for col, title, xlabel, filename in features_info:
        plt.figure(figsize=(7, 5), dpi=200)
        sns.kdeplot(data=df_plot, x=col, hue="類別", palette=palette, fill=True, common_norm=False, alpha=0.4, linewidth=2)
        plt.title(title, fontsize=13, fontweight="bold")
        plt.xlabel(xlabel)
        plt.ylabel("密度")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, filename))
        plt.close()

    plt.figure(figsize=(8, 5), dpi=200)
    sns.kdeplot(data=df_plot, x="d_center", hue="類別", palette=palette, fill=True, common_norm=False, alpha=0.4)
    plt.title("圖五：d_center 分佈（真實數據）", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fig5_raw_angular_distance_distribution.png"))
    plt.close()

    plt.figure(figsize=(10, 6), dpi=200)
    melted = pd.melt(df_plot, id_vars=["類別"], value_vars=FEATURE_COLS, var_name="特徵維度", value_name="特徵數值")
    sns.violinplot(data=melted, x="特徵維度", y="特徵數值", hue="類別", palette=palette, split=True, inner="quart")
    plt.title("圖六：四維角距離特徵提琴圖（真實數據）", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fig6_four_dimensional_features_violin.png"))
    plt.close()

    if "risk_score" in df_plot.columns:
        plt.figure(figsize=(9, 5), dpi=200)
        for lab, color, name in [(0, "#2ecc71", "Clean"), (1, "#e74c3c", "Attack")]:
            vals = df_plot.loc[df_plot["label"] == lab, "risk_score"].dropna()
            plt.hist(vals, bins=25, alpha=0.55, color=color, label=f"{name} risk")
        plt.axvline(LOW_RISK, color="black", linestyle="--", label="low=0.3")
        plt.axvline(HIGH_RISK, color="black", linestyle="-.", label="high=0.7")
        plt.title("圖七：P1 風險分數分佈（真實 OOF 分數）", fontsize=13, fontweight="bold")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "fig7_risk_score_and_routing_distribution.png"))
        plt.close()

    shift_vectors = {}
    for method in METHODS:
        sub = df[df["method_type"] == method]
        clean_mu = sub[sub["label"] == 0][FEATURE_COLS].mean().values
        adv_mu = sub[sub["label"] == 1][FEATURE_COLS].mean().values
        delta = adv_mu - clean_mu
        norm = np.linalg.norm(delta) + 1e-12
        shift_vectors[method] = delta / norm
    v1 = shift_vectors["rotate_5"]
    v2 = shift_vectors["scale_crop_mild"]
    cos_sim = float(np.dot(v1, v2))
    plt.figure(figsize=(6, 5), dpi=200)
    import seaborn as sns

    mat = np.array([[1.0, cos_sim], [cos_sim, 1.0]])
    sns.heatmap(mat, annot=True, fmt=".4f", cmap="YlGnBu", xticklabels=METHODS, yticklabels=METHODS, vmin=0.0, vmax=1.0)
    plt.title("圖八：特徵位移方向餘弦相似度（真實數據）", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fig8_shift_direction_similarity_heatmap.png"))
    plt.close()
    print(f"Plots saved to {save_dir}")
    return cos_sim


def extract_all_pairs(pairs, embedder, k, seed, partial_path):
    done = set()
    rows = []
    if os.path.isfile(partial_path):
        prev = pd.read_csv(partial_path)
        if not prev.empty and "image_id" in prev.columns:
            done = set(prev["image_id"].astype(str).unique())
            rows = prev.to_dict("records")
            print(f"Resume: {len(done)} image_ids already in {partial_path}")
    remaining = [p for p in pairs if p["image_id"] not in done]
    print(f"Feature extraction: {len(remaining)} pairs remaining / {len(pairs)} total")
    for idx, task in enumerate(remaining, 1):
        clean_img = Image.open(task["ori_path"]).convert("RGB").resize((384, 384))
        adv_img = Image.open(task["atk_path"]).convert("RGB").resize((384, 384))
        for method in METHODS:
            z_clean = extract_four_dimensional_features(
                embedder, clean_img, method, k, seed, task["image_id"], f"{method}|clean"
            )
            z_adv = extract_four_dimensional_features(
                embedder, adv_img, method, k, seed, task["image_id"], f"{method}|atk"
            )
            rows.append(
                {
                    "image_id": task["image_id"],
                    "method_type": method,
                    "caption_model": "BLIP",
                    "attack_type": "clean",
                    "label": 0,
                    "n_captions": z_clean["n_captions"],
                    "d_center": z_clean["d_center"],
                    "d_iqr": z_clean["d_iqr"],
                    "d_tail": z_clean["d_tail"],
                    "d_q95": z_clean["d_q95"],
                }
            )
            rows.append(
                {
                    "image_id": task["image_id"],
                    "method_type": method,
                    "caption_model": "BLIP",
                    "attack_type": "VLAttack",
                    "label": 1,
                    "n_captions": z_adv["n_captions"],
                    "d_center": z_adv["d_center"],
                    "d_iqr": z_adv["d_iqr"],
                    "d_tail": z_adv["d_tail"],
                    "d_q95": z_adv["d_q95"],
                }
            )
        pd.DataFrame(rows).to_csv(partial_path, index=False, encoding="utf-8")
        print(f"progress pair [{len(done) + idx}/{len(pairs)}] rows={len(rows)}", flush=True)
    return pd.DataFrame(rows)


def run_stage1(
    experiment_root=None,
    visual_dir=None,
    log_path=None,
    caption_ckpt=None,
    device="cuda",
    k=K_DEFAULT,
    seed=SEED_DEFAULT,
):
    experiment_root = Path(experiment_root or EXPERIMENT_ROOT)
    visual_dir = Path(visual_dir or (experiment_root / "visual_results"))
    log_path = Path(log_path or (experiment_root / "attack_dual_defense_log.json"))
    caption_ckpt = Path(caption_ckpt or (BLIP_ROOT / "checkpoints" / "model_base_caption_capfilt_large.pth"))
    device = resolve_device(device)
    set_seed(seed)
    print(f"Using device: {device}")

    pairs = collect_pairs_from_log(str(visual_dir), str(log_path))
    caption_model, processor = load_caption_model(str(caption_ckpt), device)
    tokenizer, sent_model = load_sentence_encoder(device)
    embedder = CaptionEmbedder(caption_model, processor, tokenizer, sent_model, device)

    partial_path = experiment_root / PARTIAL_CSV
    df = extract_all_pairs(pairs, embedder, k, seed, str(partial_path))
    df = attach_oof_risk_scores(df)
    metrics = evaluate_metrics(df)

    print("\n" + "=" * 110)
    print("【第二階段數據：特徵位移方向】")
    print("=" * 110)
    shift_vectors = {}
    for method in METHODS:
        sub = df[df["method_type"] == method]
        clean_mu = sub[sub["label"] == 0][FEATURE_COLS].mean().values
        adv_mu = sub[sub["label"] == 1][FEATURE_COLS].mean().values
        delta = adv_mu - clean_mu
        shift_vectors[method] = delta / (np.linalg.norm(delta) + 1e-12)
        print(f"[{method}] Δμ = {np.round(delta, 4)}")
    cos_sim = float(np.dot(shift_vectors["rotate_5"], shift_vectors["scale_crop_mild"]))
    print(f"rotate_5 vs scale_crop_mild shift cosine: {cos_sim:.4f}")

    print("\n" + "=" * 110)
    print("【第三階段數據：跨手法泛化 rotate_5 → scale_crop_mild】")
    print("=" * 110)
    df_train = df[df["method_type"] == "rotate_5"]
    df_test = df[df["method_type"] == "scale_crop_mild"]
    svm_p1 = Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel="rbf", C=1.0, class_weight="balanced"))])
    svm_p1.fit(df_train[FEATURE_COLS].values, df_train["label"].values)
    margins_cross = svm_p1.decision_function(df_test[FEATURE_COLS].values)
    preds_cross = svm_p1.predict(df_test[FEATURE_COLS].values)
    y_test = df_test["label"].values
    print(f"cross AUROC  : {roc_auc_score(y_test, margins_cross):.4f}")
    print(f"cross Recall : {recall_score(y_test, preds_cross, zero_division=0):.4f}")

    out_csv = experiment_root / OUTPUT_CSV
    metrics_csv = experiment_root / METRICS_CSV
    df.to_csv(out_csv, index=False, encoding="utf-8")
    metrics.to_csv(metrics_csv, index=False, encoding="utf-8")
    generate_plots(df, str(experiment_root / "academic_plots"))
    print(f"Saved {out_csv}")
    print(f"Saved {metrics_csv}")
    return df, metrics

    analyze_script = BLIP_ROOT / "scripts" / "analyze_tcvad_defense.py"
    if analyze_script.is_file():
        import subprocess
        import sys

        print(f"\nRunning post-analysis: {analyze_script.name}")
        subprocess.run(
            [sys.executable, str(analyze_script), "--run_dir", run_dir],
            check=False,
        )


if __name__ == "__main__":
    run_stage1()
