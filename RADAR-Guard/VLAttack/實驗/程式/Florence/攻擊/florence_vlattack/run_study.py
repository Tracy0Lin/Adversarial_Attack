"""Resume VLAttack image generation, Florence scoring, then the three-experiment defense."""
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = HERE.parent / "VQA" / "florence_vlattack"
MASTER = DATA / "master_attack"
PY = ROOT / "vlattack_gpu" / "Scripts" / "python.exe"
MC = HERE.parent / "multi_captioner"
CUDA_RESTART_EXIT = 75

BASE_ENV = os.environ.copy()
BASE_ENV["PYTHONUNBUFFERED"] = "1"
BASE_ENV["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
BASE_ENV["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
FLORENCE_ENV = BASE_ENV.copy()
FLORENCE_ENV["PYTHONPATH"] = str(Path(os.environ.get("LOCALAPPDATA", "")) / "vlattack_florence_pkgs")
FLORENCE_ENV["FLORENCE_MODEL_ID"] = "microsoft/Florence-2-large-ft"
FLORENCE_ENV["VQA_STUDY"] = "florence_vlattack"


def log_count():
    path = MASTER / "attack_dual_defense_log.json"
    if not path.is_file():
        return 0
    try:
        return len(json.loads(path.read_text(encoding="utf-8")))
    except json.JSONDecodeError:
        return 0


def run(cmd, env, cwd):
    print("RUN", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run(cmd, cwd=cwd, env=env).returncode


def attack_until(n):
    cmd = [
        str(PY), "-u", "attack_vlattack.py",
        "--experiment_root", str(MASTER),
        "--num_samples", str(n),
    ]
    stalls = 0
    while log_count() < n:
        before = log_count()
        code = run(cmd, BASE_ENV, HERE)
        if code == CUDA_RESTART_EXIT:
            continue
        if code != 0:
            raise SystemExit(f"attack exited {code}")
        if log_count() == before:
            stalls += 1
            if stalls > 2:
                raise SystemExit(f"attack made no progress at {log_count()}/{n}")
        else:
            stalls = 0


def score():
    code = run(
        [str(PY), "-u", "score_florence.py", "--experiment_root", str(MASTER)],
        FLORENCE_ENV, HERE,
    )
    if code != 0:
        raise SystemExit(f"score exited {code}")


def main():
    step = 1000
    while True:
        attack_until(step)
        score()
        code = run([str(PY), "-u", "build_sets.py"], BASE_ENV, HERE)
        if code == 0:
            break
        if code != 2:
            raise SystemExit(f"build_sets exited {code}")
        step += 250
        print(f"need more attacks, next target {step}", flush=True)
    for exp in ("exp1", "exp2", "exp3"):
        for cmd in (
            [str(PY), "-u", "run_probe.py", "--exp", exp, "--vqa-backend", "florence2"],
            [str(PY), "-u", "run_captions.py", "--exp", exp, "--captioner", "florence2"],
            [str(PY), "-u", "analyze.py", "--exp", exp, "--captioner", "florence2", "--vqa-backend", "florence2"],
            [str(PY), "-u", "analyze.py", "--exp", exp, "--captioner", "florence2", "--vqa-backend", "florence2",
             "--output-tag", "sigma08_fixed", "--fix-theta-star", "0.8", "--fix-theta-det", "0.8"],
        ):
            if run(cmd, FLORENCE_ENV, MC) != 0:
                raise SystemExit(f"defense failed: {cmd}")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
