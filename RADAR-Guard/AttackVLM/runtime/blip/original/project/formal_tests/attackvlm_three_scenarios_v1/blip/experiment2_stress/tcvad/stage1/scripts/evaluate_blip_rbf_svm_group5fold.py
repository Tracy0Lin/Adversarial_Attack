import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


EXPERIMENT_ROOT = Path(
    "formal_tests/attackvlm_three_scenarios_v1/blip/experiment2_stress/tcvad/stage1"
)

INPUT_CSV = (
    EXPERIMENT_ROOT /
    "features/blip_clip_vitb32_angular_features_experiment2_1000pairs.csv"
)

RESULTS_DIR = EXPERIMENT_ROOT / "results"
FIGURES_DIR = EXPERIMENT_ROOT / "figures"

PREDICTIONS_CSV = (
    RESULTS_DIR /
    "blip_rbf_svm_group5fold_oof_predictions.csv"
)

METRICS_JSON = (
    RESULTS_DIR /
    "blip_rbf_svm_group5fold_oof_metrics.json"
)

CONFUSION_PNG = (
    FIGURES_DIR /
    "blip_rbf_svm_oof_confusion_matrix.png"
)

ROC_PNG = (
    FIGURES_DIR /
    "blip_rbf_svm_oof_roc_curve.png"
)

MARGIN_PNG = (
    FIGURES_DIR /
    "blip_rbf_svm_oof_margin_distribution.png"
)

FEATURE_NAMES = [
    "d_center",
    "d_iqr",
    "d_tail",
    "d_q95",
]

SEED = 20260729
N_SPLITS = 5

SVM_C = 1.0
SVM_GAMMA = "scale"
SVM_CLASS_WEIGHT = "balanced"

TEMPERATURE = 1.0


def safe_divide(numerator, denominator):
    if denominator == 0:
        return 0.0

    return float(numerator / denominator)


def sigmoid(values, temperature=1.0):
    scaled = np.asarray(
        values,
        dtype=np.float64,
    ) / temperature

    scaled = np.clip(
        scaled,
        -60.0,
        60.0,
    )

    return 1.0 / (1.0 + np.exp(-scaled))


def evaluate_predictions(
    y_true,
    y_pred,
    decision_scores,
):
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = matrix.ravel()

    specificity = safe_divide(
        tn,
        tn + fp,
    )

    false_positive_rate = safe_divide(
        fp,
        fp + tn,
    )

    false_negative_rate = safe_divide(
        fn,
        fn + tp,
    )

    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "recall_sensitivity": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "specificity": specificity,
        "fpr": false_positive_rate,
        "fnr": false_negative_rate,
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                decision_scores,
            )
        ),
    }


def summarize_fold_metrics(fold_metrics):
    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall_sensitivity",
        "specificity",
        "fpr",
        "fnr",
        "f1",
        "roc_auc",
    ]

    summary = {}

    for metric_name in metric_names:
        values = np.array([
            fold["metrics"][metric_name]
            for fold in fold_metrics
        ], dtype=np.float64)

        summary[metric_name] = {
            "mean": float(np.mean(values)),
            "std_sample": float(
                np.std(values, ddof=1)
            ),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    return summary


print("===== 讀取四維特徵 =====", flush=True)
print("輸入：", INPUT_CSV, flush=True)

with open(INPUT_CSV, "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    fieldnames = reader.fieldnames or []

    required_fields = {
        "image_id",
        "label",
        "attack_type",
        "source_path",
        *FEATURE_NAMES,
    }

    missing_fields = sorted(
        required_fields - set(fieldnames)
    )

    if missing_fields:
        raise RuntimeError(
            f"CSV 缺少欄位：{missing_fields}"
        )

    records = []

    for row in reader:
        image_id = str(row["image_id"])
        label = int(row["label"])

        if label not in {0, 1}:
            raise RuntimeError(
                f"{image_id} 出現未知 label：{label}"
            )

        features = []

        for feature_name in FEATURE_NAMES:
            value = float(row[feature_name])

            if not np.isfinite(value):
                raise RuntimeError(
                    f"{image_id} 的 {feature_name} "
                    "出現 NaN／Inf"
                )

            if value < 0.0 or value > 1.0:
                raise RuntimeError(
                    f"{image_id} 的 {feature_name} "
                    f"超出 [0,1]：{value}"
                )

            features.append(value)

        records.append({
            "image_id": image_id,
            "label": label,
            "attack_type": row["attack_type"],
            "source_path": row["source_path"],
            "features": features,
        })


if len(records) != 2000:
    raise RuntimeError(
        f"紀錄數應為 2000，目前為 {len(records)}"
    )


label_counts = Counter(
    record["label"]
    for record in records
)

if label_counts != Counter({0: 1000, 1: 1000}):
    raise RuntimeError(
        f"類別數量錯誤：{dict(label_counts)}"
    )


pair_lookup = {}

for index, record in enumerate(records):
    key = (
        record["image_id"],
        record["label"],
    )

    if key in pair_lookup:
        raise RuntimeError(
            f"出現重複紀錄：{key}"
        )

    pair_lookup[key] = index


image_ids = sorted({
    record["image_id"]
    for record in records
})

if len(image_ids) != 1000:
    raise RuntimeError(
        f"Image IDs 應為 1000，目前為 {len(image_ids)}"
    )

for image_id in image_ids:
    if (
        (image_id, 0) not in pair_lookup
        or (image_id, 1) not in pair_lookup
    ):
        raise RuntimeError(
            f"{image_id} 缺少 Clean 或 MF-tt"
        )


X = np.asarray([
    record["features"]
    for record in records
], dtype=np.float64)

y = np.asarray([
    record["label"]
    for record in records
], dtype=np.int64)

groups = np.asarray([
    record["image_id"]
    for record in records
], dtype=object)

if X.shape != (2000, 4):
    raise RuntimeError(
        f"特徵矩陣應為 (2000,4)，目前為 {X.shape}"
    )


print("特徵矩陣：", X.shape, flush=True)
print("Image IDs：", len(image_ids), flush=True)
print("Clean：", label_counts[0], flush=True)
print("MF-tt：", label_counts[1], flush=True)
print("特徵順序：", FEATURE_NAMES, flush=True)


print("\n===== 建立固定配對 Group Folds =====", flush=True)

rng = np.random.default_rng(SEED)

shuffled_ids = rng.permutation(
    np.asarray(image_ids, dtype=object)
)

fold_test_ids = [
    [str(value) for value in fold]
    for fold in np.array_split(
        shuffled_ids,
        N_SPLITS,
    )
]

if any(len(fold) != 200 for fold in fold_test_ids):
    raise RuntimeError(
        "每折測試 Image IDs 應固定為 200"
    )

all_fold_ids = [
    image_id
    for fold in fold_test_ids
    for image_id in fold
]

if (
    len(all_fold_ids) != 1000
    or len(set(all_fold_ids)) != 1000
    or set(all_fold_ids) != set(image_ids)
):
    raise RuntimeError(
        "Group fold 沒有恰好涵蓋全部 1000 個 ID"
    )


oof_predictions = np.full(
    len(records),
    -1,
    dtype=np.int64,
)

oof_margins = np.full(
    len(records),
    np.nan,
    dtype=np.float64,
)

oof_risks = np.full(
    len(records),
    np.nan,
    dtype=np.float64,
)

oof_folds = np.full(
    len(records),
    -1,
    dtype=np.int64,
)

fold_results = []
group_leakage_count = 0


print("\n===== 執行 5-fold Group OOF =====", flush=True)

for fold_index, current_test_ids in enumerate(
    fold_test_ids,
    start=1,
):
    test_id_set = set(current_test_ids)

    test_mask = np.array([
        image_id in test_id_set
        for image_id in groups
    ], dtype=bool)

    train_mask = ~test_mask

    train_indices = np.flatnonzero(train_mask)
    test_indices = np.flatnonzero(test_mask)

    train_ids = set(groups[train_indices])
    actual_test_ids = set(groups[test_indices])

    overlap = train_ids & actual_test_ids

    if overlap:
        group_leakage_count += len(overlap)

        raise RuntimeError(
            f"Fold {fold_index} 發生 Group leakage："
            f"{sorted(overlap)}"
        )

    if len(train_ids) != 800:
        raise RuntimeError(
            f"Fold {fold_index} 訓練 ID "
            f"應為 800，目前為 {len(train_ids)}"
        )

    if len(actual_test_ids) != 200:
        raise RuntimeError(
            f"Fold {fold_index} 測試 ID "
            f"應為 200，目前為 {len(actual_test_ids)}"
        )

    if len(train_indices) != 1600:
        raise RuntimeError(
            f"Fold {fold_index} 訓練紀錄 "
            f"應為 1600，目前為 {len(train_indices)}"
        )

    if len(test_indices) != 400:
        raise RuntimeError(
            f"Fold {fold_index} 測試紀錄 "
            f"應為 400，目前為 {len(test_indices)}"
        )

    if Counter(y[train_indices]) != Counter({
        0: 800,
        1: 800,
    }):
        raise RuntimeError(
            f"Fold {fold_index} 訓練類別不平衡"
        )

    if Counter(y[test_indices]) != Counter({
        0: 200,
        1: 200,
    }):
        raise RuntimeError(
            f"Fold {fold_index} 測試類別不平衡"
        )

    model = Pipeline([
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "svm",
            SVC(
                kernel="rbf",
                C=SVM_C,
                gamma=SVM_GAMMA,
                class_weight=SVM_CLASS_WEIGHT,
                probability=False,
                random_state=SEED,
            ),
        ),
    ])

    model.fit(
        X[train_indices],
        y[train_indices],
    )

    fold_predictions = model.predict(
        X[test_indices]
    ).astype(np.int64)

    fold_margins = model.decision_function(
        X[test_indices]
    ).astype(np.float64)

    fold_risks = sigmoid(
        fold_margins,
        TEMPERATURE,
    )

    if not np.all(np.isfinite(fold_margins)):
        raise RuntimeError(
            f"Fold {fold_index} margin 出現 NaN／Inf"
        )

    if not np.all(np.isfinite(fold_risks)):
        raise RuntimeError(
            f"Fold {fold_index} risk 出現 NaN／Inf"
        )

    threshold_predictions = (
        fold_risks >= 0.5
    ).astype(np.int64)

    if not np.array_equal(
        fold_predictions,
        threshold_predictions,
    ):
        raise RuntimeError(
            f"Fold {fold_index} SVM 預測與 "
            "risk 0.5 門檻不一致"
        )

    if np.any(oof_folds[test_indices] != -1):
        raise RuntimeError(
            f"Fold {fold_index} 重複寫入 OOF 預測"
        )

    oof_predictions[test_indices] = (
        fold_predictions
    )

    oof_margins[test_indices] = (
        fold_margins
    )

    oof_risks[test_indices] = (
        fold_risks
    )

    oof_folds[test_indices] = (
        fold_index
    )

    fold_metrics = evaluate_predictions(
        y_true=y[test_indices],
        y_pred=fold_predictions,
        decision_scores=fold_margins,
    )

    fold_result = {
        "fold": fold_index,
        "train_image_ids": sorted(
            str(value)
            for value in train_ids
        ),
        "test_image_ids": sorted(
            str(value)
            for value in actual_test_ids
        ),
        "num_train_image_ids": len(train_ids),
        "num_test_image_ids": len(
            actual_test_ids
        ),
        "num_train_records": len(
            train_indices
        ),
        "num_test_records": len(
            test_indices
        ),
        "metrics": fold_metrics,
    }

    fold_results.append(fold_result)

    print(
        f"Fold {fold_index}｜"
        f"Train IDs={len(train_ids)}, "
        f"Test IDs={len(actual_test_ids)}｜"
        f"TN={fold_metrics['tn']}, "
        f"FP={fold_metrics['fp']}, "
        f"FN={fold_metrics['fn']}, "
        f"TP={fold_metrics['tp']}｜"
        f"Acc={fold_metrics['accuracy']:.4f}, "
        f"Recall={fold_metrics['recall_sensitivity']:.4f}, "
        f"F1={fold_metrics['f1']:.4f}, "
        f"AUC={fold_metrics['roc_auc']:.4f}",
        flush=True,
    )


if np.any(oof_predictions < 0):
    raise RuntimeError(
        "部分紀錄沒有 OOF prediction"
    )

if np.any(oof_folds < 1):
    raise RuntimeError(
        "部分紀錄沒有 fold 編號"
    )

if not np.all(np.isfinite(oof_margins)):
    raise RuntimeError(
        "OOF margin 出現 NaN／Inf"
    )

if not np.all(np.isfinite(oof_risks)):
    raise RuntimeError(
        "OOF risk 出現 NaN／Inf"
    )


for image_id in image_ids:
    clean_index = pair_lookup[
        (image_id, 0)
    ]

    mf_tt_index = pair_lookup[
        (image_id, 1)
    ]

    if (
        oof_folds[clean_index]
        != oof_folds[mf_tt_index]
    ):
        raise RuntimeError(
            f"{image_id} 的 Clean／MF-tt "
            "被拆到不同折"
        )


overall_metrics = evaluate_predictions(
    y_true=y,
    y_pred=oof_predictions,
    decision_scores=oof_margins,
)

fold_summary = summarize_fold_metrics(
    fold_results
)


print("\n===== 整體 OOF 結果 =====", flush=True)
print(
    "Confusion Matrix："
    f"TN={overall_metrics['tn']}, "
    f"FP={overall_metrics['fp']}, "
    f"FN={overall_metrics['fn']}, "
    f"TP={overall_metrics['tp']}",
    flush=True,
)
print(
    f"Accuracy："
    f"{overall_metrics['accuracy']:.6f}",
    flush=True,
)
print(
    f"Balanced Accuracy："
    f"{overall_metrics['balanced_accuracy']:.6f}",
    flush=True,
)
print(
    f"Precision："
    f"{overall_metrics['precision']:.6f}",
    flush=True,
)
print(
    f"Recall / TPR："
    f"{overall_metrics['recall_sensitivity']:.6f}",
    flush=True,
)
print(
    f"Specificity / TNR："
    f"{overall_metrics['specificity']:.6f}",
    flush=True,
)
print(
    f"FPR："
    f"{overall_metrics['fpr']:.6f}",
    flush=True,
)
print(
    f"FNR："
    f"{overall_metrics['fnr']:.6f}",
    flush=True,
)
print(
    f"F1："
    f"{overall_metrics['f1']:.6f}",
    flush=True,
)
print(
    f"ROC-AUC："
    f"{overall_metrics['roc_auc']:.6f}",
    flush=True,
)


RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


print("\n===== 儲存 OOF 預測 =====", flush=True)

prediction_fieldnames = [
    "image_id",
    "fold",
    "true_label",
    "true_class",
    "predicted_label",
    "predicted_class",
    "correct",
    "decision_margin",
    "risk_score_T1",
    *FEATURE_NAMES,
    "attack_type",
    "source_path",
]

prediction_rows = []

for index, record in enumerate(records):
    true_label = int(y[index])
    predicted_label = int(
        oof_predictions[index]
    )

    row = {
        "image_id": record["image_id"],
        "fold": int(oof_folds[index]),
        "true_label": true_label,
        "true_class": (
            "clean"
            if true_label == 0
            else "mf_tt"
        ),
        "predicted_label": predicted_label,
        "predicted_class": (
            "clean"
            if predicted_label == 0
            else "mf_tt"
        ),
        "correct": int(
            true_label == predicted_label
        ),
        "decision_margin": float(
            oof_margins[index]
        ),
        "risk_score_T1": float(
            oof_risks[index]
        ),
        "attack_type": record["attack_type"],
        "source_path": record["source_path"],
    }

    for feature_name, feature_value in zip(
        FEATURE_NAMES,
        record["features"],
    ):
        row[feature_name] = float(
            feature_value
        )

    prediction_rows.append(row)


prediction_rows.sort(
    key=lambda row: (
        row["fold"],
        row["image_id"],
        row["true_label"],
    )
)

temporary_predictions = (
    PREDICTIONS_CSV.with_suffix(".csv.tmp")
)

with open(
    temporary_predictions,
    "w",
    encoding="utf-8",
    newline="",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=prediction_fieldnames,
    )

    writer.writeheader()
    writer.writerows(prediction_rows)

temporary_predictions.replace(
    PREDICTIONS_CSV
)


metrics_payload = {
    "schema_version": "1.0",
    "experiment": (
        "TC-VAD Experiment 1 BLIP "
        "paired group 5-fold RBF SVM OOF"
    ),
    "input_csv": str(INPUT_CSV),
    "classification_target": {
        "negative_class": {
            "label": 0,
            "name": "clean",
        },
        "positive_class": {
            "label": 1,
            "name": "AttackVLM_MF-tt",
        },
    },
    "feature_order": FEATURE_NAMES,
    "evaluation": {
        "type": (
            "fixed-seed paired group "
            "5-fold out-of-fold"
        ),
        "seed": SEED,
        "num_folds": N_SPLITS,
        "num_image_ids": len(image_ids),
        "num_records": len(records),
        "group_key": "image_id",
        "same_id_pair_kept_together": True,
        "scaler_fit_on_training_fold_only": True,
        "post_hoc_feature_selection": False,
        "post_hoc_hyperparameter_tuning": False,
    },
    "model": {
        "pipeline": [
            "StandardScaler",
            "SVC",
        ],
        "kernel": "rbf",
        "C": SVM_C,
        "gamma": SVM_GAMMA,
        "class_weight": SVM_CLASS_WEIGHT,
        "probability": False,
    },
    "risk_score": {
        "formula": (
            "sigmoid(decision_margin / T)"
        ),
        "temperature": TEMPERATURE,
        "classification_threshold": 0.5,
        "interpretation": (
            "risk ranking score; "
            "not calibrated attack probability"
        ),
    },
    "overall_oof_metrics": overall_metrics,
    "fold_metric_summary": fold_summary,
    "folds": fold_results,
    "prediction_csv": str(PREDICTIONS_CSV),
}

temporary_metrics = (
    METRICS_JSON.with_suffix(".json.tmp")
)

with open(
    temporary_metrics,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        metrics_payload,
        file,
        ensure_ascii=False,
        indent=2,
    )

temporary_metrics.replace(
    METRICS_JSON
)


print("預測 CSV：", PREDICTIONS_CSV, flush=True)
print("指標 JSON：", METRICS_JSON, flush=True)


print("\n===== 產生混淆矩陣 =====", flush=True)

tn = overall_metrics["tn"]
fp = overall_metrics["fp"]
fn = overall_metrics["fn"]
tp = overall_metrics["tp"]

matrix = np.array([
    [tn, fp],
    [fn, tp],
], dtype=np.int64)

figure, axis = plt.subplots(
    figsize=(6.2, 5.3)
)

image = axis.imshow(
    matrix,
    cmap="Blues",
    vmin=0,
)

axis.set_xticks([0, 1])
axis.set_yticks([0, 1])

axis.set_xticklabels([
    "Predicted Clean",
    "Predicted MF-tt",
])

axis.set_yticklabels([
    "Actual Clean",
    "Actual MF-tt",
])

axis.set_xlabel("Predicted class")
axis.set_ylabel("Actual class")

axis.set_title(
    "TC-VAD Experiment 1 — BLIP\n"
    "5-fold Group OOF Confusion Matrix",
    fontweight="bold",
)

threshold = matrix.max() / 2.0

for row_index in range(2):
    for column_index in range(2):
        axis.text(
            column_index,
            row_index,
            str(matrix[row_index, column_index]),
            ha="center",
            va="center",
            fontsize=18,
            fontweight="bold",
            color=(
                "white"
                if matrix[
                    row_index,
                    column_index
                ] > threshold
                else "black"
            ),
        )

figure.colorbar(
    image,
    ax=axis,
    fraction=0.046,
    pad=0.04,
)

figure.tight_layout()

figure.savefig(
    CONFUSION_PNG,
    dpi=300,
    bbox_inches="tight",
)

plt.close(figure)


print("混淆矩陣：", CONFUSION_PNG, flush=True)


print("\n===== 產生 ROC Curve =====", flush=True)

roc_fpr, roc_tpr, _ = roc_curve(
    y,
    oof_margins,
)

figure, axis = plt.subplots(
    figsize=(6.4, 5.3)
)

axis.plot(
    roc_fpr,
    roc_tpr,
    linewidth=2.5,
    color="#E45756",
    label=(
        f"RBF SVM "
        f"(AUC={overall_metrics['roc_auc']:.3f})"
    ),
)

axis.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1.5,
    color="#777777",
    label="Random classifier",
)

axis.set_xlim(0.0, 1.0)
axis.set_ylim(0.0, 1.02)

axis.set_xlabel("False positive rate")
axis.set_ylabel("True positive rate")

axis.set_title(
    "TC-VAD Experiment 1 — BLIP\n"
    "5-fold Group OOF ROC Curve",
    fontweight="bold",
)

axis.grid(alpha=0.25)
axis.legend(loc="lower right")

figure.tight_layout()

figure.savefig(
    ROC_PNG,
    dpi=300,
    bbox_inches="tight",
)

plt.close(figure)


print("ROC Curve：", ROC_PNG, flush=True)


print("\n===== 產生 Margin 分布圖 =====", flush=True)

clean_margins = oof_margins[y == 0]
mf_tt_margins = oof_margins[y == 1]

combined_margins = np.concatenate([
    clean_margins,
    mf_tt_margins,
])

minimum_margin = float(
    np.min(combined_margins)
)

maximum_margin = float(
    np.max(combined_margins)
)

if np.isclose(
    minimum_margin,
    maximum_margin,
):
    minimum_margin -= 0.5
    maximum_margin += 0.5

bins = np.linspace(
    minimum_margin,
    maximum_margin,
    16,
)

figure, axis = plt.subplots(
    figsize=(7.2, 5.2)
)

axis.hist(
    clean_margins,
    bins=bins,
    alpha=0.65,
    color="#4C78A8",
    edgecolor="white",
    label="Clean",
)

axis.hist(
    mf_tt_margins,
    bins=bins,
    alpha=0.60,
    color="#E45756",
    edgecolor="white",
    label="MF-tt",
)

axis.axvline(
    0.0,
    color="black",
    linestyle="--",
    linewidth=2,
    label="Decision boundary",
)

axis.set_xlabel("Out-of-fold SVM decision margin")
axis.set_ylabel("Number of records")

axis.set_title(
    "TC-VAD Experiment 1 — BLIP\n"
    "OOF Decision Margin Distribution",
    fontweight="bold",
)

axis.grid(
    axis="y",
    alpha=0.22,
)

axis.legend()

figure.tight_layout()

figure.savefig(
    MARGIN_PNG,
    dpi=300,
    bbox_inches="tight",
)

plt.close(figure)


print("Margin 分布：", MARGIN_PNG, flush=True)


print("\n===== 完整性驗證 =====", flush=True)

expected_outputs = [
    PREDICTIONS_CSV,
    METRICS_JSON,
    CONFUSION_PNG,
    ROC_PNG,
    MARGIN_PNG,
]

missing_outputs = [
    str(path)
    for path in expected_outputs
    if (
        not path.exists()
        or path.stat().st_size == 0
    )
]

if missing_outputs:
    raise RuntimeError(
        f"輸出不存在或為空：{missing_outputs}"
    )

print("Image IDs：", len(image_ids), flush=True)
print("OOF 預測：", len(oof_predictions), flush=True)
print("每個 ID 恰好測試一次：True", flush=True)
print("Clean／MF-tt 保持同折：True", flush=True)
print(
    "Train／Test Group leakage：",
    group_leakage_count,
    flush=True,
)
print("NaN／Inf：0", flush=True)
print("輸出檔案：", len(expected_outputs), flush=True)

print("\nGROUP_5FOLD_RBF_SVM_OK", flush=True)
