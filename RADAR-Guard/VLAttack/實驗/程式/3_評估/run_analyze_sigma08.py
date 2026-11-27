"""Re-run analyze only with stage-2 Gaussian sigma fixed at 0.8 (no overwrite of default outputs)."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MC = Path(__file__).resolve().parents[1] / "multi_captioner"
PY = sys.executable
TAG = "sigma08_fixed"
THETA = "0.8"

RUNS = [
    {"env": {"VQA_STUDY": "florence_victim"}, "vqa": "florence2", "captioner": "florence2"},
    {"env": {}, "vqa": "blip", "captioner": "blip"},
]


def main():
    for spec in RUNS:
        env = os.environ.copy()
        env.update(spec["env"])
        env.setdefault("PYTHONPATH", os.path.join(os.environ.get("LOCALAPPDATA", ""), "vlattack_florence_pkgs"))
        cmd = [
            PY, "-u", "analyze.py",
            "--exp", "exp1", "exp2", "exp3",
            "--captioner", spec["captioner"],
            "--vqa-backend", spec["vqa"],
            "--output-tag", TAG,
            "--fix-theta-star", THETA,
            "--fix-theta-det", THETA,
        ]
        print("===", spec["vqa"], "===", flush=True)
        subprocess.run(cmd, cwd=MC, env=env, check=True)


if __name__ == "__main__":
    main()
