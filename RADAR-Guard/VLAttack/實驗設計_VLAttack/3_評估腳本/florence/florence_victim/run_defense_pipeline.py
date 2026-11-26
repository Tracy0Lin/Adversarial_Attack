"""Run probe + captions + analyze for florence_victim study."""
import os
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "multi_captioner"
PY = sys.executable
ENV = os.environ.copy()
ENV["VQA_STUDY"] = "florence_victim"
ENV["PYTHONPATH"] = os.path.join(os.environ.get("LOCALAPPDATA", ""), "vlattack_florence_pkgs")
ENV["FLORENCE_MODEL_ID"] = "microsoft/Florence-2-large-ft"
ENV["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


def run(cmd):
    print("RUN", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=SCRIPTS, env=ENV, check=True)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--exp", nargs="+", default=["exp1", "exp2", "exp3"])
    args = p.parse_args()
    for exp in args.exp:
        run([PY, "-u", "run_probe.py", "--exp", exp, "--vqa-backend", "florence2"])
        run([PY, "-u", "run_captions.py", "--exp", exp, "--captioner", "florence2"])
        run([PY, "-u", "analyze.py", "--exp", exp, "--captioner", "florence2", "--vqa-backend", "florence2"])


if __name__ == "__main__":
    main()
