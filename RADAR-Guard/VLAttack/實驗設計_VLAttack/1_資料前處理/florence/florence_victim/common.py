"""Paths for Florence victim study (separate from BLIP multi_captioner)."""
import json
import sys
from pathlib import Path

VQA_OUT = Path(__file__).resolve().parents[1] / "VQA"
DATA_ROOT = VQA_OUT / "florence_victim"
EXP1_DIR = DATA_ROOT / "exp1_mixed1000"
EXP2_DIR = DATA_ROOT / "exp2_success1000"
EXP3_DIR = DATA_ROOT / "exp3_success1000"
BLIP_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "multi_captioner"))
sys.path.insert(0, str(BLIP_ROOT))

EXPERIMENTS = ("exp1", "exp2", "exp3")
CAPTIONER = "florence2"
VQA_BACKEND = "florence2"

EXP_MAP = {
    "exp1": EXP1_DIR,
    "exp2": EXP2_DIR,
    "exp3": EXP3_DIR,
}


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def experiment_spec(exp):
    if exp == "exp1":
        log = read_json(EXP1_DIR / "attack_dual_defense_log.json")
        return EXP1_DIR / "visual_results", log, True
    if exp == "exp2":
        log = read_json(EXP2_DIR / "attack_dual_defense_log.json")
        return EXP2_DIR / "visual_results", log, True
    if exp == "exp3":
        log = read_json(EXP3_DIR / "attack_dual_defense_log.json")
        return EXP3_DIR / "visual_results", log, False
    raise ValueError(exp)
