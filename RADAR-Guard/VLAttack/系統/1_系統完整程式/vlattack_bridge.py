"""Load the VLAttack demo catalog built from VLAttack_zip experiment records."""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CATALOG_PATH = ROOT / "data" / "vlattack_demo" / "catalog.json"
DEFAULT_VLATTACK_ROOT = Path(
    os.environ.get(
        "VLATTACK_ROOT",
        r"C:\Users\tracy_aboikg6\OneDrive\Desktop\program\VLAttack_zip",
    )
)


@lru_cache(maxsize=1)
def load_catalog() -> dict:
    if not CATALOG_PATH.is_file():
        raise FileNotFoundError(
            f"找不到 {CATALOG_PATH}。請先執行 python scripts/build_vlattack_demo.py"
        )
    with CATALOG_PATH.open(encoding="utf-8") as f:
        catalog = json.load(f)
    catalog["vlattack_root"] = str(DEFAULT_VLATTACK_ROOT)
    catalog["vlattack_root_exists"] = DEFAULT_VLATTACK_ROOT.is_dir()
    return catalog


METHOD_EFFECTS = {
    "asr": 0.462,
    "mean_ssim": 0.8003,
    "residual_asr": {
        "none": 46.2,
        "mild_resize": 74.46,
        "gaussian_3x3": 57.14,
        "gaussian_5x5": 37.01,
        "resize_gaussian": 41.34,
    },
    "end_to_end_j": {
        "stage1_high_only": 0.232,
        "answer_flip": 0.370,
        "flip_count": 0.387,
    },
}

DISPLAY = {
    "A_correct_clean": "正確放行",
    "B_correct_attack": "成功攔截",
    "C_false_alarm": "假警報",
    "D_miss_attack": "漏抓",
    "E_clean_mid": "中風險（乾淨）",
    "F_boundary": "邊界",
    "G_attack_mid": "中風險走完二階",
}


def list_samples() -> list[dict]:
    return [
        {
            "id": s["id"],
            "label": DISPLAY.get(s["id"], s["id"]),
            "question": s["question"],
            "why": s["why"],
            "feed": s["feed"],
            "attack_success": s["attack_success"],
            "risk_bin": s["stage1"]["risk_bin"],
            "thumb": s["images"]["ori"],
        }
        for s in load_catalog()["samples"]
    ]


def get_sample(sample_id: str) -> dict | None:
    for sample in load_catalog()["samples"]:
        if sample["id"] == sample_id or sample["question_id"] == str(sample_id):
            return sample
    return None
