"""Rebuild and verify the recorded OOF models on CPU, then export them.

No torch, BLIP, CLIP, images, or GPU are used. Original CSVs are read only.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import shutil
import sys
import tempfile
from collections import Counter

import joblib
import numpy as np
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ['d_center', 'd_iqr', 'd_tail', 'd_q95']
SEED = 20260729
SCENARIOS = {
    'natural': ('formal_tests/paper_coco1000_v1/tcvad/experiment1_blip',
                'blip_clip_vitb32_angular_features_all_1000pairs.csv'),
    'stress': ('formal_tests/attackvlm_three_scenarios_v1/blip/experiment2_stress/tcvad/stage1',
               'blip_clip_vitb32_angular_features_experiment2_1000pairs.csv'),
}

def sigmoid(margin):
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(margin, dtype=np.float64), -60.0, 60.0)))

def tiers(risk):
    return np.where(risk < .30, 'low', np.where(risk < .80, 'medium', 'high'))

def make_model():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced',
                    probability=False, random_state=SEED)),
    ])

def read_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def export_scenario(scenario, source_root, destination, include_deployment=False):
    folder, feature_name = SCENARIOS[scenario]
    source = source_root / folder
    feature_path = source / 'features' / feature_name
    reference_path = source / 'results/blip_rbf_svm_group5fold_oof_predictions.csv'
    metrics_path = source / 'results/blip_rbf_svm_group5fold_oof_metrics.json'
    rows, reference = read_rows(feature_path), read_rows(reference_path)
    if len(rows) != 2000 or len(reference) != 2000:
        raise ValueError('Expected 2,000 feature and OOF rows for ' + scenario)
    keys = [(r['image_id'], int(r['label'])) for r in rows]
    if len(set(keys)) != 2000 or Counter(label for _, label in keys) != {0:1000, 1:1000}:
        raise ValueError('Invalid labels or duplicate feature rows')
    groups = np.array([r['image_id'] for r in rows], dtype=object)
    ids = np.array(sorted(set(groups)), dtype=object)
    if len(ids) != 1000 or any((i,0) not in keys or (i,1) not in keys for i in ids):
        raise ValueError('Expected 1,000 paired IDs')
    X = np.array([[float(r[f]) for f in FEATURES] for r in rows], dtype=np.float64)
    y = np.array([label for _, label in keys], dtype=np.int64)
    if not np.isfinite(X).all() or (X < 0).any() or (X > 1).any():
        raise ValueError('Invalid features')
    lookup = {(r['image_id'], int(r['true_label'])):r for r in reference}
    if set(lookup) != set(keys):
        raise ValueError('OOF keys differ from feature keys')
    expected = [lookup[k] for k in keys]
    ref_X = np.array([[float(r[f]) for f in FEATURES] for r in expected])
    if not np.array_equal(X, ref_X):
        raise ValueError('OOF feature values do not match input CSV')
    metadata = {
        'schema_version':1, 'scenario':scenario, 'feature_order':FEATURES,
        'seed':SEED, 'risk_temperature':1.0, 'risk_thresholds':[.30,.80],
        'python':sys.version, 'numpy':np.__version__, 'sklearn':sklearn.__version__,
        'input_sha256':{str(p.relative_to(source_root)):sha256(p)
                        for p in (feature_path,reference_path,metrics_path)},
    }
    output = destination / scenario
    output.mkdir()
    observed_margin = np.zeros(len(y), dtype=np.float64)
    observed_prediction = np.zeros(len(y), dtype=np.int64)
    observed_fold = np.zeros(len(y), dtype=np.int64)
    fold_ids = np.array_split(np.random.default_rng(SEED).permutation(ids), 5)
    for fold, test_ids in enumerate(fold_ids, 1):
        mask = np.isin(groups, test_ids)
        if mask.sum() != 400 or set(groups[mask]) & set(groups[~mask]):
            raise ValueError('Invalid paired split')
        model = make_model()
        model.fit(X[~mask], y[~mask])
        if list(model.classes_) != [0,1]:
            raise ValueError('Class order differs from the recorded detector')
        observed_margin[mask] = model.decision_function(X[mask])
        observed_prediction[mask] = model.predict(X[mask])
        observed_fold[mask] = fold
        artifact = {**metadata, 'purpose':'oof_replay', 'fold':fold,
                    'test_image_ids':[str(i) for i in test_ids], 'pipeline':model}
        joblib.dump(artifact, output / ('fold_%d.joblib' % fold), compress=3)
    ref_margin = np.array([float(r['decision_margin']) for r in expected])
    ref_risk = np.array([float(r['risk_score_T1']) for r in expected])
    ref_prediction = np.array([int(r['predicted_label']) for r in expected])
    ref_fold = np.array([int(r['fold']) for r in expected])
    risk = sigmoid(observed_margin)
    audit = {
        'scenario':scenario, 'records':len(rows),
        'max_margin_error':float(np.max(np.abs(observed_margin-ref_margin))),
        'max_risk_error':float(np.max(np.abs(risk-ref_risk))),
        'prediction_mismatches':int(np.sum(observed_prediction!=ref_prediction)),
        'fold_mismatches':int(np.sum(observed_fold!=ref_fold)),
        'routing_mismatches':int(np.sum(tiers(risk)!=tiers(ref_risk))),
        'sklearn':sklearn.__version__, 'numpy':np.__version__,
        'reference_oof_metrics':json.loads(metrics_path.read_text())['overall_oof_metrics'],
    }
    if (audit['prediction_mismatches'] or audit['fold_mismatches'] or
            audit['routing_mismatches'] or audit['max_margin_error']>1e-8 or audit['max_risk_error']>1e-8):
        raise ValueError('OOF reconstruction failed: ' + json.dumps(audit))
    if include_deployment:
        full = make_model()
        full.fit(X,y)
        joblib.dump({**metadata, 'purpose':'deployment_full_training', 'pipeline':full,
                     'report_oof_scores_apply_to_this_model':False},
                    output/'deployment.joblib', compress=3)
    # Check that serialization preserves the rebuilt scores.
    for fold in range(1,6):
        mask = observed_fold == fold
        loaded = joblib.load(output/('fold_%d.joblib'%fold))['pipeline']
        if not np.allclose(loaded.decision_function(X[mask]),observed_margin[mask],atol=1e-12,rtol=0):
            raise ValueError('Serialization changed model decisions')
    audit['serialized_models_verified'] = 5
    audit['deployment_model_exported'] = bool(include_deployment)
    (output/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    (output/'fold_test_ids.json').write_text(json.dumps({str(k+1):[str(i) for i in v]
                                                      for k,v in enumerate(fold_ids)},indent=2))
    return audit

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=ROOT/'reference_data/project')
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts')
    parser.add_argument('--scenario',choices=['both','natural','stress'],default='both')
    parser.add_argument('--include-deployment-model',action='store_true',
                        help='Optional new full-data model; not part of the recorded OOF experiment.')
    parser.add_argument('--allow-version-difference',action='store_true',
                        help='For inspection only; deployment exports should use recorded sklearn 1.6.1.')
    args = parser.parse_args()
    if sklearn.__version__ != '1.6.1' and not args.allow_version_difference:
        raise SystemExit('Use the original attackvlm Python environment (scikit-learn 1.6.1).')
    output = args.output.resolve()
    if output.exists():
        raise SystemExit('Output already exists; choose a new --output directory: '+str(output))
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.detector_export_',dir=str(output.parent)))
    try:
        audits = [export_scenario(s,args.data_root.resolve(),temporary,args.include_deployment_model)
                  for s in (SCENARIOS if args.scenario=='both' else [args.scenario])]
        (temporary/'export_summary.json').write_text(json.dumps(audits,indent=2))
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary,ignore_errors=True)
        raise
    for audit in audits:
        print(json.dumps(audit,ensure_ascii=False))
    print('BLIP_DETECTOR_EXPORT_VERIFIED',output)

if __name__ == '__main__':
    main()
