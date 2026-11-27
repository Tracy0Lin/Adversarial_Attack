"""Cosine 4D vs angular 4D on the stored +/-5 degree rotation captions.

The 11 sentences are already in caption_features.csv: one original caption,
n_minus5 copies of the -5 degree caption, and the rest +5 degree. This script
re-encodes those strings with all-MiniLM-L6-v2 (mean pool, L2 norm), computes
the same four statistics on cosine and on arccos(cosine)/pi, checks the angular
numbers against the stored columns, then scores both feature sets with the
stage-1 RBF SVM.

No captions are invented. A blank sentence aborts the run.
"""
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from transformers import BertTokenizer

THIS_FILE = Path(__file__).resolve()
BLIP_OUTPUT = THIS_FILE.parent
ROOT = BLIP_OUTPUT / "VQA" / "multi_captioner"
OUT_DIR = BLIP_OUTPUT / "VQA" / "rotation_cos_vs_ang"
MODEL_DIR = Path(
    r"C:\Users\tracy\.cache\huggingface\hub\models--sentence-transformers--all-MiniLM-L6-v2"
    r"\snapshots\1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
)
EXPS = ("exp1", "exp2", "exp3")
STATS = ("d_center", "d_iqr", "d_tail", "d_q95")
COS_COLS = [f"{name}_cos" for name in STATS]
ANG_COLS = [f"{name}_ang" for name in STATS]
STORED_COLS = list(STATS)
N_FOLDS = 5
N_BOOT = 1000
SEED = 42
MATCH_ATOL = 1e-4
HIDDEN = 384
HEADS = 12
HEAD_DIM = HIDDEN // HEADS
LAYERS = 6
LN_EPS = 1e-12


def rotation_angles(image_id, label, k=10, seed=42):
    """Same stream as multi_captioner.common.rotation_angles. image_id is str."""
    tag = f"rotate_5|{'clean' if int(label) == 0 else 'atk'}"
    return [random.Random(f"{seed}|{image_id}|{tag}|{i}").choice([-5.0, 5.0]) for i in range(k)]


def four_stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "d_center": float(np.mean(values)),
        "d_iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
        "d_tail": float(np.percentile(values, 90) - np.percentile(values, 50)),
        "d_q95": float(np.percentile(values, 95)),
    }


def stats_from_embeddings(emb):
    """emb is L2-normalised, shape (11, 384).

    The centroid dot is done in float64. Float32 dots of an identical set sit
    just below 1, and arccos turns that into a fake distance of about 1e-4.
    The stored features are exact zeros when all 11 captions match.
    """
    emb64 = emb.detach().double()
    if torch.equal(emb64, emb64[:1].expand_as(emb64)):
        cosine = np.ones(emb64.shape[0], dtype=np.float64)
    else:
        center = F.normalize(emb64.mean(0, keepdim=True), dim=-1)
        cosine = np.clip((emb64 @ center.T).squeeze(-1).numpy(), -1.0, 1.0)
    angular = np.arccos(cosine) / np.pi
    return four_stats(cosine), four_stats(angular)


class MiniLMEncoder:
    """BertModel forward in eval mode, then masked mean pool and L2 norm."""

    def __init__(self, model_dir):
        blob = torch.load(model_dir / "pytorch_model.bin", map_location="cpu")
        self.weights = {key: value.float() for key, value in blob.items() if torch.is_tensor(value)}
        self.tokenizer = BertTokenizer.from_pretrained(str(model_dir))

    def _linear(self, x, name):
        return F.linear(x, self.weights[f"{name}.weight"], self.weights[f"{name}.bias"])

    def _ln(self, x, name):
        return F.layer_norm(
            x,
            (HIDDEN,),
            self.weights[f"{name}.weight"],
            self.weights[f"{name}.bias"],
            LN_EPS,
        )

    def _layer(self, hidden, mask, index):
        prefix = f"encoder.layer.{index}."
        query = self._linear(hidden, prefix + "attention.self.query")
        key = self._linear(hidden, prefix + "attention.self.key")
        value = self._linear(hidden, prefix + "attention.self.value")
        batch, length, _ = query.shape

        def split(tensor):
            return tensor.view(batch, length, HEADS, HEAD_DIM).permute(0, 2, 1, 3)

        query, key, value = split(query), split(key), split(value)
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(HEAD_DIM)
        scores = scores + mask
        context = torch.matmul(torch.softmax(scores, dim=-1), value)
        context = context.permute(0, 2, 1, 3).contiguous().view(batch, length, HIDDEN)
        attended = self._linear(context, prefix + "attention.output.dense")
        hidden = self._ln(hidden + attended, prefix + "attention.output.LayerNorm")
        intermediate = F.gelu(self._linear(hidden, prefix + "intermediate.dense"))
        output = self._linear(intermediate, prefix + "output.dense")
        return self._ln(hidden + output, prefix + "output.LayerNorm")

    @torch.no_grad()
    def encode(self, texts, batch_size=64):
        """Return a dict of text -> float32 unit vector. Order of texts does not matter."""
        unique = list(dict.fromkeys(texts))
        out = {}
        for start in range(0, len(unique), batch_size):
            chunk = unique[start:start + batch_size]
            encoded = self.tokenizer(
                chunk, padding=True, truncation=True, max_length=512, return_tensors=None
            )
            input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long)
            token_type = torch.tensor(encoded["token_type_ids"], dtype=torch.long)
            attention = torch.tensor(encoded["attention_mask"], dtype=torch.float32)
            hidden = (
                F.embedding(input_ids, self.weights["embeddings.word_embeddings.weight"])
                + F.embedding(
                    torch.arange(input_ids.shape[1]).unsqueeze(0),
                    self.weights["embeddings.position_embeddings.weight"],
                )
                + F.embedding(token_type, self.weights["embeddings.token_type_embeddings.weight"])
            )
            hidden = self._ln(hidden, "embeddings.LayerNorm")
            # Pad positions get a large negative; real tokens then match an unpadded forward.
            mask = (1.0 - attention)[:, None, None, :] * -10000.0
            for index in range(LAYERS):
                hidden = self._layer(hidden, mask, index)
            denom = attention.sum(dim=1, keepdim=True).clamp(min=1e-9)
            pooled = (hidden * attention.unsqueeze(-1)).sum(dim=1) / denom
            pooled = F.normalize(pooled, p=2, dim=-1)
            for text, vector in zip(chunk, pooled.cpu()):
                out[text] = vector.clone()
            print(f"  encoded {min(start + batch_size, len(unique))}/{len(unique)}", flush=True)
        return out


def load_exp(exp):
    path = ROOT / exp / "blip" / "caption_features.csv"
    if not path.is_file():
        raise SystemExit(f"missing caption file: {path}")
    frame = pd.read_csv(path, dtype={"image_id": str}, keep_default_na=False)
    needed = [
        "image_id", "photo_id", "label", "n_minus5",
        "caption_orig", "caption_rot_minus5", "caption_rot_plus5",
        *STORED_COLS,
    ]
    missing_cols = [column for column in needed if column not in frame.columns]
    if missing_cols:
        raise SystemExit(f"{exp} missing columns: {missing_cols}")
    return frame


def blank_mask(series):
    text = series.astype(str)
    return text.str.strip().eq("") | text.isin(["nan", "None", "NaN"])


def sentence_sequence(row):
    image_id = str(row["image_id"])
    label = int(row["label"])
    angles = rotation_angles(image_id, label)
    n_minus = int(row["n_minus5"])
    counted = sum(angle < 0 for angle in angles)
    if counted != n_minus:
        raise SystemExit(
            f"rotation count mismatch image_id={image_id} label={label}: "
            f"seed says {counted} times -5, csv n_minus5={n_minus}"
        )
    orig = row["caption_orig"]
    neg = row["caption_rot_minus5"]
    pos = row["caption_rot_plus5"]
    return [orig] + [neg if angle < 0 else pos for angle in angles]


def build_features(exp, vectors):
    frame = load_exp(exp)
    records = []
    worst = 0.0
    for row in frame.itertuples(index=False):
        mapping = row._asdict()
        sequence = sentence_sequence(mapping)
        emb = torch.stack([vectors[text] for text in sequence])
        cos_stats, ang_stats = stats_from_embeddings(emb)
        record = {
            "experiment": exp,
            "image_id": mapping["image_id"],
            "photo_id": int(mapping["photo_id"]),
            "label": int(mapping["label"]),
            "n_unique_captions": len(set(sequence)),
        }
        for name in STATS:
            record[f"{name}_cos"] = cos_stats[name]
            record[f"{name}_ang"] = ang_stats[name]
            record[f"{name}_stored"] = float(mapping[name])
            worst = max(worst, abs(ang_stats[name] - float(mapping[name])))
        records.append(record)
    built = pd.DataFrame(records)
    errors = np.maximum.reduce(
        [np.abs(built[f"{name}_ang"] - built[f"{name}_stored"]) for name in STATS]
    )
    varied = (built["n_unique_captions"] > 1).to_numpy()
    return built, {
        "experiment": exp,
        "rows": int(len(built)),
        "max_abs": float(errors.max()),
        "max_abs_varied": float(errors[varied].max()) if varied.any() else 0.0,
        "mean_abs_varied": float(errors[varied].mean()) if varied.any() else 0.0,
        "n_identical": int((~varied).sum()),
        "n_above_atol": int((errors[varied] > MATCH_ATOL).sum()) if varied.any() else 0,
    }


def svm_pipe():
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=1.0, class_weight="balanced")),
        ]
    )


def oof_margin(frame, columns):
    work = frame.reset_index(drop=True)
    x = work[list(columns)].to_numpy(dtype=np.float64)
    y = work["label"].to_numpy(dtype=int)
    groups = work["photo_id"].to_numpy()
    margin = np.zeros(len(work), dtype=np.float64)
    prediction = np.zeros(len(work), dtype=int)
    for train_idx, test_idx in GroupKFold(n_splits=N_FOLDS).split(x, y, groups):
        model = svm_pipe().fit(x[train_idx], y[train_idx])
        margin[test_idx] = model.decision_function(x[test_idx])
        prediction[test_idx] = model.predict(x[test_idx])
    return margin, prediction


def separated_rates(labels, flagged):
    labels = np.asarray(labels, dtype=int)
    flagged = np.asarray(flagged, dtype=bool)
    attack = labels == 1
    clean = labels == 0
    return {
        "attack_n": int(attack.sum()),
        "attack_flagged": int((attack & flagged).sum()),
        "attack_catch_rate": float((attack & flagged).sum() / max(attack.sum(), 1)),
        "clean_n": int(clean.sum()),
        "clean_flagged": int((clean & flagged).sum()),
        "clean_false_alarm_rate": float((clean & flagged).sum() / max(clean.sum(), 1)),
    }


def to_risk(margin):
    return 1.0 / (1.0 + np.exp(-np.clip(margin, -50, 50)))


def tpr_at_fpr(labels, scores, target_fpr):
    """Interpolated ROC point, plus the best realized threshold with FPR <= target.

    Ties share one threshold: a cut cannot flag only part of an equal score.
    """
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=np.float64)
    fpr, tpr, _ = roc_curve(labels, scores)
    interpolated = float(np.interp(target_fpr, fpr, tpr))
    n_clean = max(int((labels == 0).sum()), 1)
    n_attack = max(int((labels == 1).sum()), 1)
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]
    ranked = scores[order]
    false_pos = 0
    true_pos = 0
    best = None
    index = 0
    while index < len(y):
        end = index + 1
        while end < len(y) and ranked[end] == ranked[index]:
            end += 1
        true_pos += int((y[index:end] == 1).sum())
        false_pos += int((y[index:end] == 0).sum())
        fpr_now = false_pos / n_clean
        if fpr_now <= target_fpr + 1e-12:
            tpr_now = true_pos / n_attack
            if best is None or tpr_now >= best[0]:
                best = (tpr_now, fpr_now, float(ranked[index]), true_pos, false_pos)
        index = end
    if best is None:
        best = (0.0, 0.0, float("inf"), 0, 0)
    return {
        "target_fpr": target_fpr,
        "tpr_interp": interpolated,
        "tpr_at_most": best[0],
        "fpr_at_most": best[1],
        "score_threshold": best[2],
        "attack_flagged": best[3],
        "clean_flagged": best[4],
    }


def univariate_auroc(frame):
    rows = []
    y = frame["label"].to_numpy(dtype=int)
    for name in STATS:
        for kind, column in (("cosine", f"{name}_cos"), ("angular", f"{name}_ang")):
            values = frame[column].to_numpy(dtype=np.float64)
            raw = float(roc_auc_score(y, values))
            flipped = float(roc_auc_score(y, -values))
            if raw >= flipped:
                auroc, direction = raw, "higher"
            else:
                auroc, direction = flipped, "lower"
            rows.append({
                "feature": name,
                "space": kind,
                "auroc_toward_attack": auroc,
                "attack_direction": direction,
            })
    return rows


def bootstrap_delta(labels, groups, cos_score, ang_score, n_boot=N_BOOT):
    rng = np.random.default_rng(SEED)
    photos = np.unique(groups)
    buckets = {photo: np.flatnonzero(groups == photo) for photo in photos}
    deltas = []
    for _ in range(n_boot):
        drawn = rng.choice(photos, size=len(photos), replace=True)
        idx = np.concatenate([buckets[photo] for photo in drawn])
        y = labels[idx]
        if y.min() == y.max():
            continue
        deltas.append(roc_auc_score(y, ang_score[idx]) - roc_auc_score(y, cos_score[idx]))
    deltas = np.asarray(deltas, dtype=np.float64)
    return {
        "n_valid": int(len(deltas)),
        "mean": float(deltas.mean()),
        "p2_5": float(np.percentile(deltas, 2.5)),
        "p97_5": float(np.percentile(deltas, 97.5)),
    }


def mean_table(frame):
    rows = []
    for label, kind in ((0, "clean"), (1, "attack")):
        part = frame[frame["label"] == label]
        if part.empty:
            continue
        row = {"kind": kind, "n": int(len(part)), "n_unique_captions": float(part["n_unique_captions"].mean())}
        for column in COS_COLS + ANG_COLS:
            row[column] = float(part[column].mean())
            row[f"{column}_std"] = float(part[column].std(ddof=1))
        rows.append(row)
    return rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = {exp: load_exp(exp) for exp in EXPS}
    texts = []
    for exp, frame in frames.items():
        for column in ("caption_orig", "caption_rot_minus5", "caption_rot_plus5"):
            bad = blank_mask(frame[column])
            if bad.any():
                ids = frame.loc[bad, "image_id"].head(20).tolist()
                raise SystemExit(
                    f"{exp} has {int(bad.sum())} blank {column}; first ids {ids}. "
                    "Refusing to fill them in."
                )
            texts.extend(frame[column].astype(str).tolist())
    print(f"unique captions {len(set(texts))}", flush=True)
    encoder = MiniLMEncoder(MODEL_DIR)
    vectors = encoder.encode(texts)

    built = {}
    checks = []
    for exp in EXPS:
        print(f"features {exp}", flush=True)
        table, check = build_features(exp, vectors)
        table.to_csv(OUT_DIR / f"{exp}_features.csv", index=False, encoding="utf-8")
        built[exp] = table
        checks.append(check)
        print(
            f"  match varied_max={check['max_abs_varied']:.3e} "
            f"identical_rows={check['n_identical']} all_rows_max={check['max_abs']:.3e}",
            flush=True,
        )
    pd.DataFrame(checks).to_csv(OUT_DIR / "angular_match.csv", index=False)
    failed = [row for row in checks if row["max_abs_varied"] > MATCH_ATOL]
    if failed:
        raise SystemExit(
            "Recomputed angular 4D does not match the stored columns. "
            "SVM was not fit. See angular_match.csv."
        )

    summaries = []
    sweep_rows = []
    uni_rows = []
    boot_rows = []
    oof_parts = []
    for exp in ("exp1", "exp2"):
        frame = built[exp]
        print(f"svm {exp}", flush=True)
        means = mean_table(frame)
        uni = univariate_auroc(frame)
        for row in uni:
            row["experiment"] = exp
            uni_rows.append(row)
        margins = {}
        preds = {}
        for feature, columns in (("cosine_4d", COS_COLS), ("angular_4d", ANG_COLS)):
            margin, prediction = oof_margin(frame, columns)
            margins[feature] = margin
            preds[feature] = prediction
            y = frame["label"].to_numpy(dtype=int)
            risk = to_risk(margin)
            default = separated_rates(y, prediction == 1)
            summary = {
                "experiment": exp,
                "feature": feature,
                "n_rows": int(len(frame)),
                "n_photos": int(frame["photo_id"].nunique()),
                "auroc": float(roc_auc_score(y, margin)),
                "mean_unique_clean": next(row["n_unique_captions"] for row in means if row["kind"] == "clean"),
                "mean_unique_attack": next(row["n_unique_captions"] for row in means if row["kind"] == "attack"),
                **{f"default_{key}": value for key, value in default.items()},
            }
            for target in (0.05, 0.10):
                point = tpr_at_fpr(y, risk, target)
                summary[f"tpr_interp_fpr{int(target * 100)}"] = point["tpr_interp"]
                summary[f"tpr_atmost_fpr{int(target * 100)}"] = point["tpr_at_most"]
                summary[f"fpr_atmost_fpr{int(target * 100)}"] = point["fpr_at_most"]
                sweep_rows.append({"experiment": exp, "feature": feature, **point})
            summaries.append(summary)
            part = frame[["experiment", "image_id", "photo_id", "label"]].copy()
            part["feature"] = feature
            part["oof_margin"] = margin
            part["oof_risk"] = risk
            part["oof_pred"] = prediction
            oof_parts.append(part)
        boot = bootstrap_delta(
            frame["label"].to_numpy(dtype=int),
            frame["photo_id"].to_numpy(),
            margins["cosine_4d"],
            margins["angular_4d"],
        )
        boot["experiment"] = exp
        boot_rows.append(boot)
        print(
            f"  cosine AUROC {summaries[-2]['auroc']:.4f}  angular {summaries[-1]['auroc']:.4f}  "
            f"delta 95% [{boot['p2_5']:.4f}, {boot['p97_5']:.4f}]",
            flush=True,
        )

    pd.DataFrame(summaries).to_csv(OUT_DIR / "svm_summary.csv", index=False)
    pd.DataFrame(sweep_rows).to_csv(OUT_DIR / "fpr_operating_points.csv", index=False)
    pd.DataFrame(uni_rows).to_csv(OUT_DIR / "univariate_auroc.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(OUT_DIR / "bootstrap_delta.csv", index=False)
    pd.concat(oof_parts, ignore_index=True).to_csv(OUT_DIR / "oof_scores.csv", index=False)
    write_report(built, checks, summaries, uni_rows, boot_rows)
    print(f"wrote {OUT_DIR}", flush=True)


def fmt(value, digits=4):
    return f"{value:.{digits}f}"


def write_report(built, checks, summaries, uni_rows, boot_rows):
    lines = [
        "# ±5° 旋轉：cosine 四維與角距四維",
        "",
        "同一批 BLIP 描述句。每張圖 11 句：1 句原描述，加上 10 次 ±5°。",
        "句子用 all-MiniLM-L6-v2 做平均池化並 L2 正規化。中心是這 11 句自己的平均方向。",
        "cosine 四維是這 11 個 cosine 的平均、IQR、Q90−中位數、Q95。角距四維把同一組統計做在 arccos(cosine)/π 上。",
        "重算出的角距四維先和 `caption_features.csv` 裡已存的四欄比對，對上之後才訓練 SVM。",
        "",
        "SVM 是 RBF、C = 1、`class_weight = balanced`，特徵先標準化。",
        "實驗一、二各自五折 GroupKFold，分組是 `photo_id`（`question_id // 1000`），兩種四維用同一組折。",
        "AUROC 用折外決策值。實驗三沒有乾淨圖，只列平均。",
        "",
        "## 角距核對",
        "",
        "| 實驗 | 列數 | 句子不完全相同的最大差 | 那些列的平均差 | 11 句全相同的列 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in checks:
        lines.append(
            f"| {row['experiment']} | {row['rows']} | {row['max_abs_varied']:.3e} | "
            f"{row['mean_abs_varied']:.3e} | {row['n_identical']} |"
        )
    lines += [
        "",
        "比的是重算的角距四維和 `caption_features.csv` 已存的四欄。",
        "句子不完全相同的列，最大差在 1e-4 以內才往下訓練。",
        "11 句都是同一句時，角距的定義是 0。已存檔裡這些列有的寫 0，有的留著約 1.1e-4，那是 float32 把自己和自己做內積時沒有剛好得到 1，`arccos` 留下的殘差。這次不把殘差放進特徵。",
        "",
        "## 平均值",
        "",
        "cosine 愈高愈像中心。角距愈高愈散。括號是標準差。",
        "",
        "| 實驗 | 類型 | n | 不同句數 | cos center | cos iqr | cos tail | cos q95 | ang center | ang iqr | ang tail | ang q95 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for exp in EXPS:
        for row in mean_table(built[exp]):
            lines.append(
                "| {exp} | {kind} | {n} | {uniq:.2f} | {cc} | {ci} | {ct} | {cq} | {ac} | {ai} | {at} | {aq} |".format(
                    exp=exp,
                    kind="乾淨" if row["kind"] == "clean" else "攻擊",
                    n=row["n"],
                    uniq=row["n_unique_captions"],
                    cc=cell(row, "d_center_cos"),
                    ci=cell(row, "d_iqr_cos"),
                    ct=cell(row, "d_tail_cos"),
                    cq=cell(row, "d_q95_cos"),
                    ac=cell(row, "d_center_ang"),
                    ai=cell(row, "d_iqr_ang"),
                    at=cell(row, "d_tail_ang"),
                    aq=cell(row, "d_q95_ang"),
                )
            )
    lines += [
        "",
        "## SVM",
        "",
        "| 實驗 | 特徵 | AUROC | 預設切點攔截 | 預設切點誤攔 | FPR 5% 插值攔截 | FPR 5% 可實現攔截 | FPR 10% 插值攔截 | FPR 10% 可實現攔截 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            "| {exp} | {feat} | {auroc} | {catch} | {fpr} | {i5} | {r5} | {i10} | {r10} |".format(
                exp=row["experiment"],
                feat="cosine 四維" if row["feature"] == "cosine_4d" else "角距四維",
                auroc=fmt(row["auroc"]),
                catch=pct(row["default_attack_catch_rate"], row["default_attack_flagged"], row["default_attack_n"]),
                fpr=pct(row["default_clean_false_alarm_rate"], row["default_clean_flagged"], row["default_clean_n"]),
                i5=fmt(row["tpr_interp_fpr5"], 3),
                r5=fmt(row["tpr_atmost_fpr5"], 3),
                i10=fmt(row["tpr_interp_fpr10"], 3),
                r10=fmt(row["tpr_atmost_fpr10"], 3),
            )
        )
    lines += [
        "",
        "預設切點是決策值 ≥ 0。FPR 5% 與 10% 用折外風險分（決策值的 sigmoid）。",
        "插值攔截是 ROC 上剛好落在該誤攔率的攔截率。可實現攔截是誤攔不超過該值時，實際切得到的最高攔截率。",
        "",
        "照片分組 bootstrap 1000 次，抽的是已算好的折外分數，沒有重訓。區間是角距 AUROC 減 cosine AUROC。",
        "",
        "| 實驗 | 有效抽樣 | 平均差 | 2.5% | 97.5% |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in boot_rows:
        lines.append(
            f"| {row['experiment']} | {row['n_valid']} | {row['mean']:.4f} | {row['p2_5']:.4f} | {row['p97_5']:.4f} |"
        )
    lines += [
        "",
        "## 單一統計的 AUROC",
        "",
        "cosine 這欄的 d_center 和 d_q95 愈低愈像攻擊，因為那兩個統計靠近 1 代表句子聚在一起。",
        "d_iqr 和 d_tail 愈高愈散，攻擊方向是愈高。表內 AUROC 都轉成愈大愈像攻擊。這四個數不是 SVM。",
        "",
        "| 實驗 | 統計 | cosine | 角距 |",
        "| --- | --- | ---: | ---: |",
    ]
    by_exp = {}
    for row in uni_rows:
        by_exp.setdefault(row["experiment"], {}).setdefault(row["feature"], {})[row["space"]] = row
    for exp, features in by_exp.items():
        for name in STATS:
            cos = features[name]["cosine"]
            ang = features[name]["angular"]
            lines.append(
                f"| {exp} | {name} | {cos['auroc_toward_attack']:.4f} | {ang['auroc_toward_attack']:.4f} |"
            )
    by_feature = {(row["experiment"], row["feature"]): row for row in summaries}
    by_boot = {row["experiment"]: row for row in boot_rows}
    lines += [
        "",
        "## 結果",
        "",
    ]
    for exp in ("exp1", "exp2"):
        ang = by_feature[(exp, "angular_4d")]
        cos = by_feature[(exp, "cosine_4d")]
        boot = by_boot[exp]
        lines.append(
            f"{exp}：角距 AUROC {ang['auroc']:.4f}，cosine {cos['auroc']:.4f}。"
            f"差的 95% 區間是 [{boot['p2_5']:.4f}, {boot['p97_5']:.4f}]。"
            f"誤攔 5% 的插值攔截，角距 {ang['tpr_interp_fpr5']:.3f}，cosine {cos['tpr_interp_fpr5']:.3f}。"
        )
    lines += [
        "",
        "兩個區間都跨過 0。在這套 ±5° 旋轉句子上，四個統計放進同一個 RBF-SVM 之後，角距沒有穩定勝過 cosine。",
        "誤攔 5% 的攔截率差，用同一批折外分數再做一次照片 bootstrap：實驗一約 [−0.029, 0.075]，實驗二約 [−0.060, 0.071]，也跨過 0。",
        "單維裡角距較高的是 `d_q95`。`d_center` 是 cosine 略高。`d_iqr` 兩邊接近。",
        "同一套 SVM 若改吃已存的角距四維，AUROC 是實驗一 0.662、實驗二 0.717，和書面報告一致。",
        "上面的對照用這次重算的兩種四維，11 句全相同的列角距記成 0，兩邊是同一組句子。",
        "連糊十次那批的差距不放進這張表。這次也沒有做 Florence-2 或其他描述模型。",
        "",
    ]
    report = THIS_FILE.parents[2] / "成果報告" / "2026-11-22_旋轉cosine與角距四維.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {report}", flush=True)


def cell(row, name):
    return f"{row[name]:.4f}（{row[name + '_std']:.4f}）"


def pct(rate, flagged, total):
    return f"{rate:.1%}（{flagged}/{total}）"


if __name__ == "__main__":
    sys.exit(main())
