import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


PROJECT = Path(
    "/workspace/projects/couser/AttackVLM_Official_Reproduction"
)
SEED = 20260729
FEATURE_ORDER = ["d_center", "d_iqr", "d_tail", "d_q95"]


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def calculate_metrics(y_true, y_pred, scores):
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "specificity": float(tn / (tn + fp)),
        "false_positive_rate": float(fp / (tn + fp)),
    }


def evaluate(slug, display):
    defense = (
        PROJECT
        / "formal_tests/attackvlm_three_scenarios_v1/git_large_coco/experiment2_stress/tcvad/stage1"
        / "."
    )
    feature_path = (
        defense / "features/git_large_coco_clip_vitb32_angular_features_experiment2_1000pairs.json"
    )
    output = defense / "svm_group5fold"
    model_dir = output / "models"

    output.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    assert feature_path.is_file(), f"找不到：{feature_path}"

    payload = json.loads(feature_path.read_text(encoding="utf-8"))
    assert payload["feature_order"] == FEATURE_ORDER
    assert payload["summary"]["num_records"] == 2000
    assert payload["summary"]["num_clean"] == 1000
    assert payload["summary"]["num_mf_tt"] == 1000

    rows = sorted(
        payload["records"],
        key=lambda row: (str(row["image_id"]), row["label"]),
    )

    features = []
    labels = []
    groups = []
    metadata = []

    label_map = {
        0: 0,
        1: 1,
        "0": 0,
        "1": 1,
        "clean": 0,
        "mf_tt": 1,
        "MF-tt": 1,
        "mf-tt": 1,
    }

    for row in rows:
        label_name = row["label"]
        assert label_name in label_map, label_name

        vector = [
            float(row["features_all"]["features"][name])
            for name in FEATURE_ORDER
        ]

        features.append(vector)
        labels.append(label_map[label_name])
        groups.append(str(row["image_id"]))
        metadata.append(
            {
                "image_id": str(row["image_id"]),
                "label": label_name,
            }
        )

    X = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    group_array = np.asarray(groups)

    assert X.shape == (2000, 4)
    assert np.isfinite(X).all()
    assert int(y.sum()) == 1000
    assert len(set(groups)) == 1000

    grouped_labels = defaultdict(list)
    for group, label in zip(groups, labels):
        grouped_labels[group].append(label)

    assert all(
        sorted(values) == [0, 1]
        for values in grouped_labels.values()
    ), "每個 ID 必須恰好包含一筆 Clean 與一筆 MF-tt"

    splitter = GroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED,
    )

    oof_prediction = np.full(2000, -1, dtype=np.int64)
    oof_margin = np.full(2000, np.nan, dtype=np.float64)
    fold_number = np.full(2000, -1, dtype=np.int64)

    fold_results = []

    print(f"\n===== {display}：RBF-SVM Group 5-fold =====")
    print("資料：1000 Clean + 1000 MF-tt")
    print("特徵：", FEATURE_ORDER)
    print("Seed：", SEED)

    for fold, (train_index, test_index) in enumerate(
        splitter.split(X, y, groups=group_array),
        start=1,
    ):
        train_groups = sorted(set(group_array[train_index]))
        test_groups = sorted(set(group_array[test_index]))

        assert len(train_groups) == 800
        assert len(test_groups) == 200
        assert set(train_groups).isdisjoint(test_groups)
        assert len(train_index) == 1600
        assert len(test_index) == 400

        classifier = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "svm",
                    SVC(
                        kernel="rbf",
                        C=1.0,
                        gamma="scale",
                        class_weight="balanced",
                    ),
                ),
            ]
        )

        classifier.fit(X[train_index], y[train_index])

        prediction = classifier.predict(X[test_index])
        margin = classifier.decision_function(X[test_index])

        oof_prediction[test_index] = prediction
        oof_margin[test_index] = margin
        fold_number[test_index] = fold

        metrics = calculate_metrics(
            y[test_index],
            prediction,
            margin,
        )

        model_path = model_dir / f"fold_{fold}.joblib"
        joblib.dump(classifier, model_path)

        fold_result = {
            "fold": fold,
            "train_image_ids": train_groups,
            "test_image_ids": test_groups,
            "train_ids": len(train_groups),
            "test_ids": len(test_groups),
            "train_records": len(train_index),
            "test_records": len(test_index),
            "model_file": str(model_path),
            "model_sha256": digest(model_path),
            **metrics,
        }
        fold_results.append(fold_result)

        print(
            f"Fold {fold}｜"
            f"Acc={metrics['accuracy']:.4f}｜"
            f"Recall={metrics['recall']:.4f}｜"
            f"F1={metrics['f1']:.4f}｜"
            f"AUC={metrics['roc_auc']:.4f}｜"
            f"FP={metrics['fp']}｜FN={metrics['fn']}"
        )

    assert (oof_prediction >= 0).all()
    assert np.isfinite(oof_margin).all()
    assert (fold_number >= 1).all()

    overall = calculate_metrics(y, oof_prediction, oof_margin)

    clipped_margin = np.clip(oof_margin, -50.0, 50.0)
    risk_score = 1.0 / (1.0 + np.exp(-clipped_margin))

    prediction_path = output / "oof_predictions.csv"

    with prediction_path.open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_id",
                "label",
                "y_true",
                "fold",
                "prediction",
                "decision_margin",
                "sigmoid_margin_score",
                *FEATURE_ORDER,
            ],
        )
        writer.writeheader()

        for index, row in enumerate(metadata):
            writer.writerow(
                {
                    "image_id": row["image_id"],
                    "label": row["label"],
                    "y_true": int(y[index]),
                    "fold": int(fold_number[index]),
                    "prediction": int(oof_prediction[index]),
                    "decision_margin": float(oof_margin[index]),
                    "sigmoid_margin_score": float(
                        risk_score[index]
                    ),
                    **{
                        name: float(X[index, column])
                        for column, name in enumerate(
                            FEATURE_ORDER
                        )
                    },
                }
            )

    original_script = (
        PROJECT
        / "formal_tests/paper_coco1000_v1/tcvad/"
        "experiment1_blip/scripts/"
        "evaluate_blip_rbf_svm_group5fold.py"
    )

    result = {
        "status": "complete",
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "model": display,
        "feature_file": str(feature_path),
        "feature_file_sha256": digest(feature_path),
        "feature_order": FEATURE_ORDER,
        "positive_class": "mf_tt",
        "num_image_ids": 1000,
        "num_records": 2000,
        "cross_validation": {
            "method": "GroupKFold",
            "n_splits": 5,
            "shuffle": True,
            "random_state": SEED,
            "group": "image_id",
            "train_ids_per_fold": 800,
            "test_ids_per_fold": 200,
        },
        "classifier": {
            "pipeline": ["StandardScaler", "SVC"],
            "kernel": "rbf",
            "C": 1.0,
            "gamma": "scale",
            "class_weight": "balanced",
            "prediction_threshold": "SVC decision margin >= 0",
            "sigmoid_margin_note": (
                "Used only as a monotonic risk ranking; "
                "not a calibrated probability."
            ),
        },
        "sklearn": sklearn.__version__,
        "folds": fold_results,
        "oof": overall,
        "oof_predictions": str(prediction_path),
        "oof_predictions_sha256": digest(prediction_path),
        "original_blip_evaluation_script": (
            str(original_script)
            if original_script.is_file()
            else None
        ),
        "original_blip_evaluation_script_sha256": (
            digest(original_script)
            if original_script.is_file()
            else None
        ),
    }

    result_path = output / "results.json"
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"\n===== {display} OOF 總結果 =====")
    print(f"Accuracy：{overall['accuracy']:.4f}")
    print(f"Balanced Accuracy：{overall['balanced_accuracy']:.4f}")
    print(f"Precision：{overall['precision']:.4f}")
    print(f"Recall：{overall['recall']:.4f}")
    print(f"F1：{overall['f1']:.4f}")
    print(f"ROC-AUC：{overall['roc_auc']:.4f}")
    print(
        "Confusion matrix："
        f"TN={overall['tn']}、FP={overall['fp']}、"
        f"FN={overall['fn']}、TP={overall['tp']}"
    )
    print("結果：", result_path)
    print(f"{display}_RBF_SVM_GROUP5FOLD_PASS")

    return result


def main():
    results = [
        evaluate("git_large_coco", "GIT-large-COCO"),
    ]

    comparison = {
        result["model"]: result["oof"]
        for result in results
    }

    comparison_path = (
        PROJECT
        / "formal_tests/caption_small_formal1000_v1"
        / "svm_group5fold_comparison.json"
    )
    comparison_path.write_text(
        json.dumps(
            comparison,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print("\n===== 兩模型比較 =====")
    for model, metrics in comparison.items():
        print(
            f"{model}｜"
            f"Acc={metrics['accuracy']:.4f}｜"
            f"Recall={metrics['recall']:.4f}｜"
            f"F1={metrics['f1']:.4f}｜"
            f"AUC={metrics['roc_auc']:.4f}｜"
            f"FP={metrics['fp']}｜FN={metrics['fn']}"
        )

    print("比較結果：", comparison_path)
    print("SMALL_CAPTION_RBF_SVM_GROUP5FOLD_PASS")


if __name__ == "__main__":
    main()
