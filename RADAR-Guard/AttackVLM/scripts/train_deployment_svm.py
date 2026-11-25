"""Export full-data GIT/ViT-GPT2 TC-VAD detectors from recorded formal features.

These artifacts are separate from the held-out five-fold evaluation models.
"""
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
import argparse
import json
import math
import shutil
import sys
import tempfile

import joblib
import numpy as np
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

FEATURE_ORDER = ["d_center", "d_iqr", "d_tail", "d_q95"]
SOURCES = {
    ("git", "natural"): "results/git/natural/defense/features/angular_features_1000pairs.json",
    ("git", "stress"): "results/git/stress/tcvad/stage1/features/git_large_coco_clip_vitb32_angular_features_experiment2_1000pairs.json",
    ("vitgpt2", "natural"): "results/vitgpt2/natural/defense/features/angular_features_1000pairs.json",
    ("vitgpt2", "stress"): "results/vitgpt2/stress/tcvad/stage1/features/vitgpt2_clip_vitb32_angular_features_experiment2_1000pairs.json",
}
LABEL_MAP = {0: 0, 1: 1, "0": 0, "1": 1,
             "clean": 0, "mf_tt": 1, "MF-tt": 1, "mf-tt": 1}
EXPECTED_SVM = {"kernel": "rbf", "C": 1.0,
                "gamma": "scale", "class_weight": "balanced"}


def digest(path):
    h = sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_formal(root, model, scenario):
    path = root / SOURCES[(model, scenario)]
    if not path.is_file():
        raise ValueError(f"Missing feature file: {path}")
    result_path = path.parent.parent / "svm_group5fold/results.json"
    if not result_path.is_file():
        raise ValueError(f"Missing recorded SVM configuration: {result_path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if payload.get("feature_order") != FEATURE_ORDER or result.get("feature_order") != FEATURE_ORDER:
        raise ValueError(f"Feature order differs: {path} or {result_path}")
    if len(payload.get("records", [])) != 2000:
        raise ValueError(f"Expected 2000 records: {path}")
    if result.get("num_records") != 2000 or result.get("positive_class") != "mf_tt":
        raise ValueError(f"Recorded result sample count or positive class differs: {result_path}")
    if result.get("feature_file_sha256") != digest(path):
        raise ValueError(f"Feature file differs from recorded evaluation: {path}")
    cfg = result.get("classifier", {})
    for key, expected in EXPECTED_SVM.items():
        if cfg.get(key) != expected:
            raise ValueError(f"Recorded {model}/{scenario} {key}={cfg.get(key)!r}, expected {expected!r}")
    counts, pair_labels = Counter(), defaultdict(set)
    X, y = [], []
    for row in payload["records"]:
        label = LABEL_MAP.get(row.get("label"))
        if label is None:
            raise ValueError(f"Unknown label: {row.get('label')!r}")
        image_id = str(row["image_id"])
        vector = [float(row["features_all"]["features"][f]) for f in FEATURE_ORDER]
        if any(not math.isfinite(v) or v < 0 or v > 1 for v in vector):
            raise ValueError(f"Feature out of [0,1] for {image_id}")
        counts[label] += 1
        if label in pair_labels[image_id]:
            raise ValueError(f"Duplicate ID/label: {image_id}/{label}")
        pair_labels[image_id].add(label)
        X.append(vector)
        y.append(label)
    if counts != {0: 1000, 1: 1000} or len(pair_labels) != 1000 or any(v != {0, 1} for v in pair_labels.values()):
        raise ValueError(f"Expected 1000 paired IDs: {model}/{scenario}: {counts}, {len(pair_labels)}")
    return path, result_path, np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path,
                        help="Model output directory; defaults to RELEASE_ROOT/models. Must not contain destination folders.")
    args = parser.parse_args()
    root = args.release_root.resolve()
    if not root.is_dir():
        raise SystemExit(f"Missing release root: {root}")
    model_root = args.output_root.resolve() if args.output_root else root / "models"
    model_root.mkdir(parents=True, exist_ok=True)
    planned = []
    for model, scenario in SOURCES:
        output = model_root / model / scenario
        if output.exists():
            raise SystemExit(f"Will not overwrite: {output}")
        path, result_path, X, y = read_formal(root, model, scenario)
        planned.append((model, scenario, output, path, result_path, X, y))

    temp = Path(tempfile.mkdtemp(prefix=".deploy_svm_", dir=str(model_root)))
    try:
        for model, scenario, output, source, formal_result, X, y in planned:
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("svm", SVC(**EXPECTED_SVM)),
            ])
            pipeline.fit(X, y)
            if list(pipeline.classes_) != [0, 1]:
                raise ValueError(f"Class order mismatch: {model}/{scenario}")
            folder = temp / model / scenario
            folder.mkdir(parents=True)
            artifact = {
                "schema_version": 1,
                "purpose": "deployment_full_training",
                "caption_model": model,
                "scenario": scenario,
                "feature_order": FEATURE_ORDER,
                "sklearn": sklearn.__version__,
                "numpy": np.__version__,
                "training_samples": 2000,
                "training_image_ids": 1000,
                "labels": {"clean": 0, "mf_tt": 1},
                "input_sha256": {str(source.relative_to(root)): digest(source)},
                "formal_result_sha256": digest(formal_result),
                "report_oof_scores_apply_to_this_model": False,
                "pipeline": pipeline,
            }
            file = folder / "deployment.joblib"
            joblib.dump(artifact, file, compress=3)
            loaded = joblib.load(file)
            if loaded["purpose"] != "deployment_full_training" or loaded["sklearn"] != sklearn.__version__:
                raise ValueError(f"Reload metadata mismatch: {file}")
            before = float(pipeline.decision_function(X[:1])[0])
            after = float(loaded["pipeline"].decision_function(X[:1])[0])
            if not math.isfinite(after) or not np.isclose(before, after, atol=1e-12, rtol=0):
                raise ValueError(f"Reload decision mismatch: {file}")
            (folder / "manifest.json").write_text(json.dumps({
                "caption_model": model, "scenario": scenario,
                "purpose": artifact["purpose"], "feature_order": FEATURE_ORDER,
                "training_samples": 2000, "training_image_ids": 1000,
                "svm": EXPECTED_SVM, "sklearn": sklearn.__version__,
                "source_features": str(source.relative_to(root)),
                "source_features_sha256": digest(source),
                "formal_result": str(formal_result.relative_to(root)),
                "formal_result_sha256": digest(formal_result),
                "deployment_joblib_sha256": digest(file),
                "independent_test_metric": None,
                "note": "Full-data deployment artifact; formal five-fold OOF metrics do not apply.",
            }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(model, scenario, "model_bytes", file.stat().st_size, "reload_margin", after, flush=True)
        for model, scenario, output, *_ in planned:
            output.parent.mkdir(parents=True, exist_ok=True)
            (temp / model / scenario).rename(output)
        print("DEPLOYMENT_SVM_EXPORT_VERIFIED", model_root)
    finally:
        shutil.rmtree(temp, ignore_errors=True)


if __name__ == "__main__":
    main()
