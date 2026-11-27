"""Shared paths, experiment definitions and image helpers for the multi-captioner study."""
import hashlib
import io
import json
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

THIS_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = THIS_DIR.parent
BLIP_ROOT = OUTPUT_ROOT.parent
VQA_OUT = OUTPUT_ROOT / "VQA"
_STUDY = os.getenv("VQA_STUDY", "multi_captioner")
_STUDY_DIR = {
    "florence_victim": "florence_victim",
    "florence_vlattack": "florence_vlattack",
}.get(_STUDY, "multi_captioner")
DATA_ROOT = VQA_OUT / _STUDY_DIR
CACHE_ROOT = DATA_ROOT / "caption_cache"
RUN1000 = VQA_OUT / "steps100_alpha0.002_run1000"
EXP2_DIR = DATA_ROOT / "exp2_success1000" if _STUDY.startswith("florence") else VQA_OUT / "exp2_success1000"
EXP3_DIR = DATA_ROOT / "exp3_success1000"
SNAPSHOT = DATA_ROOT / "attack_log_snapshot_2298.json"
FLORENCE_EXP1 = DATA_ROOT / "exp1_mixed1000"

sys.path.insert(0, str(OUTPUT_ROOT))
sys.path.insert(0, str(BLIP_ROOT))

EXPERIMENTS = ("exp1", "exp2", "exp3")
CAPTIONERS = ("blip", "vitgpt2", "git", "gemma3")
SIGMAS = (0.3, 0.5, 0.8, 1.0, 1.5)
KERNEL = 5
K_ROT = 10
SEED = 42
METHOD = "rotate_5"
STAGE1_SIZE = 384
LOW_RISK = 0.3
HIGH_RISK = 0.7


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def photo_id(qid):
    return int(qid) // 1000


def experiment_spec(exp):
    """Return (visual_dir, log entries, include_clean) for one experiment."""
    if _STUDY in ("florence_victim", "florence_vlattack"):
        if exp == "exp1":
            log_path = FLORENCE_EXP1 / "attack_dual_defense_log.json"
            return FLORENCE_EXP1 / "visual_results", read_json(log_path), True
        if exp == "exp2":
            return EXP2_DIR / "visual_results", read_json(EXP2_DIR / "attack_dual_defense_log.json"), True
        if exp == "exp3":
            return EXP3_DIR / "visual_results", read_json(EXP3_DIR / "attack_dual_defense_log.json"), False
        raise ValueError(exp)
    if exp == "exp1":
        return RUN1000 / "visual_results", read_json(SNAPSHOT)[:1000], True
    if exp == "exp2":
        return EXP2_DIR / "visual_results", read_json(EXP2_DIR / "attack_dual_defense_log.json"), True
    if exp == "exp3":
        return EXP3_DIR / "visual_results", read_json(EXP3_DIR / "attack_dual_defense_log.json"), False
    raise ValueError(exp)


def experiment_rows(exp):
    """One dict per (qid, label) row, in log order; clean rows first per qid."""
    visual_dir, logs, include_clean = experiment_spec(exp)
    rows = []
    for item in logs:
        qid = str(item["question_id"])
        base = {
            "image_id": qid,
            "photo_id": photo_id(qid),
            "question": item.get("question_original", ""),
            "original_prediction": item.get("original_prediction", ""),
            "ground_truth": item.get("ground_truth", ""),
            "log_attack_success": bool(item.get("attack_success", False)),
        }
        if include_clean:
            rows.append({**base, "label": 0, "path": str(visual_dir / f"{qid}_1_ori.png")})
        rows.append({**base, "label": 1, "path": str(visual_dir / f"{qid}_2_atk.png")})
    return rows


def open_rgb(path, retries=3):
    """Open a PNG; OneDrive occasionally fails a read, and exp2 copies can be refetched."""
    path = Path(path)
    last = None
    for _ in range(retries):
        try:
            with Image.open(path) as im:
                return im.convert("RGB")
        except OSError as err:
            last = err
            fallback = RUN1000 / "visual_results" / path.name
            if fallback.is_file() and fallback != path:
                shutil.copy2(fallback, path)
    raise last


def image_hash(pil_img):
    arr = np.asarray(pil_img, dtype=np.uint8)
    h = hashlib.md5(arr.tobytes())
    h.update(str(arr.shape).encode())
    return h.hexdigest()


def png_bytes(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


def rotation_angles(image_id, label, k=K_ROT, seed=SEED):
    """Same RNG stream as evaluate_best_stage1_new.extract_four_dimensional_features."""
    tag = f"{METHOD}|{'clean' if label == 0 else 'atk'}"
    return [random.Random(f"{seed}|{image_id}|{tag}|{i}").choice([-5.0, 5.0]) for i in range(k)]


def rotate(pil_img, angle):
    return TF.rotate(pil_img, angle, interpolation=InterpolationMode.BILINEAR, fill=0)


def gaussian(pil_img, sigma, kernel=KERNEL):
    t = TF.to_tensor(pil_img)
    b = TF.gaussian_blur(t, kernel_size=[kernel, kernel], sigma=[sigma, sigma])
    return TF.to_pil_image(b.clamp(0.0, 1.0))


def exp_dir(exp, captioner=None):
    d = DATA_ROOT / exp
    if captioner:
        d = d / captioner
    d.mkdir(parents=True, exist_ok=True)
    return d
