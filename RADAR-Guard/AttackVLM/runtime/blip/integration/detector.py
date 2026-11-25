"""CPU risk scoring from a verified exported scaler/SVM pipeline."""
from pathlib import Path
import json
import joblib
import numpy as np
import sklearn

FEATURE_ORDER = ['d_center','d_iqr','d_tail','d_q95']

def load_detector(path):
    artifact = joblib.load(Path(path))
    if artifact.get('feature_order') != FEATURE_ORDER:
        raise ValueError('Detector feature order mismatch')
    if artifact.get('sklearn') != sklearn.__version__:
        raise ValueError('Load this model with the same scikit-learn version used to export it')
    if list(artifact['pipeline'].classes_) != [0,1]:
        raise ValueError('Expected Clean=0 and MF-tt=1')
    return artifact

def score_features(artifact, features, image_id=None):
    if artifact['purpose']=='oof_replay':
        if image_id is None or str(image_id) not in artifact['test_image_ids']:
            raise ValueError('OOF replay requires the matching held-out image_id')
    values = np.array([[float(features[k]) for k in FEATURE_ORDER]],dtype=np.float64)
    if not np.isfinite(values).all() or (values<0).any() or (values>1).any():
        raise ValueError('Expected four finite angular features in [0,1]')
    margin = float(artifact['pipeline'].decision_function(values)[0])
    risk = float(1/(1+np.exp(-np.clip(margin,-60,60))))
    level = 'low' if risk<.30 else 'medium' if risk<.80 else 'high'
    return {'decision_margin':margin,'risk_score':risk,'risk_level':level,
            'action':{'low':'pass','medium':'repair','high':'block'}[level],
            'detector_scenario':artifact['scenario'],'detector_purpose':artifact['purpose']}
