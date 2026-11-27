"""End-to-end Florence victim pipeline with git commit+push after each milestone."""
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # VLAttack_zip
FV = Path(__file__).resolve().parent
MC = FV.parent / "multi_captioner"
VQA_FV = FV.parent / "VQA" / "florence_victim"
PY = ROOT / "vlattack_gpu" / "Scripts" / "python.exe"
if not PY.is_file():
    PY = Path(sys.executable)

LOG = VQA_FV / "e2e_pipeline.log"
STATE = VQA_FV / "e2e_state.json"

ENV = os.environ.copy()
ENV["VQA_STUDY"] = "florence_victim"
ENV["PYTHONPATH"] = str(Path(os.environ.get("LOCALAPPDATA", "")) / "vlattack_florence_pkgs")
ENV["FLORENCE_MODEL_ID"] = "microsoft/Florence-2-large-ft"
ENV["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
ENV["PYTHONUNBUFFERED"] = "1"
ENV["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


def say(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    VQA_FV.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


CUDA_RESTART_EXIT = 75


def run(cmd, cwd=None, allow_cuda_restart=False):
    say("RUN " + " ".join(str(c) for c in cmd))
    proc = subprocess.run(cmd, cwd=cwd or FV, env=ENV, check=False)
    if proc.returncode == 0:
        return
    if allow_cuda_restart and proc.returncode == CUDA_RESTART_EXIT:
        return
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def attack_log_count(log_path: Path) -> int:
    if not log_path.is_file():
        return 0
    try:
        return len(json.loads(log_path.read_text(encoding="utf-8")))
    except json.JSONDecodeError:
        return 0


def run_attack_until(root: Path, num_samples: int, exclude_log=()):
    log_path = root / "attack_dual_defense_log.json"
    cmd = [
        str(PY), "-u", "attack_florence_pgd.py",
        "--experiment_root", str(root),
        "--num_samples", str(num_samples),
    ]
    for ex in exclude_log:
        cmd.extend(["--exclude_log", str(ex)])
    while attack_log_count(log_path) < num_samples:
        run(cmd, allow_cuda_restart=True)
        if attack_log_count(log_path) >= num_samples:
            break
        say(f"attack resume {root.name}: {attack_log_count(log_path)}/{num_samples} (CUDA restart)")


def load_state():
    if STATE.is_file():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {"done": []}


def save_state(st):
    STATE.write_text(json.dumps(st, indent=2), encoding="utf-8")


def done(name):
    return name in load_state()["done"]


def mark(name):
    st = load_state()
    if name not in st["done"]:
        st["done"].append(name)
        save_state(st)


def git_milestone(message, paths):
    rel = []
    for p in paths:
        p = Path(p)
        if not p.is_absolute():
            p = ROOT / p
        if p.exists():
            rel.append(str(p.relative_to(ROOT)).replace("\\", "/"))
    # Large / gitignored logs: commit pool + code only unless forced.
    skip_names = {"e2e_pipeline.log", "screening_log.jsonl"}
    rel = [r for r in rel if Path(r).name not in skip_names]
    if not rel:
        say(f"git skip (no paths): {message}")
        return
    add = subprocess.run(["git", "add", "--"] + rel, cwd=ROOT, capture_output=True, text=True)
    if add.returncode != 0:
        say(f"git add warn: {add.stderr.strip() or add.stdout.strip()}")
        # Retry without paths that git refused (e.g. ignored).
        staged = []
        for r in rel:
            one = subprocess.run(["git", "add", "--", r], cwd=ROOT, capture_output=True, text=True)
            if one.returncode == 0:
                staged.append(r)
        if not staged:
            say(f"git skip commit (nothing staged): {message}")
            return
    st = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT)
    if st.returncode == 0:
        say(f"git nothing to commit: {message}")
        return
    subprocess.run(["git", "commit", "-m", message], cwd=ROOT, check=True)
    for attempt in range(8):
        try:
            subprocess.run(["git", "push", "origin", "HEAD"], cwd=ROOT, check=True)
            say(f"git pushed: {message}")
            return
        except subprocess.CalledProcessError:
            time.sleep(min(30 * (2 ** attempt), 600))
    raise RuntimeError(f"git push failed: {message}")


COMMIT_PATHS_BASE = [
    "BLIP_attack/output/florence_victim",
    "BLIP_attack/output/VQA/florence_victim/PLAN.md",
    "BLIP_attack/output/VQA/florence_victim/e2e_state.json",
    "BLIP_attack/output/VQA/florence_victim/e2e_pipeline.log",
    "BLIP_attack/output/multi_captioner/common.py",
    "BLIP_attack/output/multi_captioner/vqa_backends.py",
    "BLIP_attack/output/multi_captioner/captioners.py",
    "BLIP_attack/output/multi_captioner/analyze.py",
    "BLIP_attack/output/multi_captioner/run_probe.py",
]


def paths_exp(root_name):
    base = f"BLIP_attack/output/VQA/florence_victim/{root_name}"
    return COMMIT_PATHS_BASE + [
        f"{base}/attack_dual_defense_log.json",
        f"{base}/dataset_manifest.csv",
        f"{base}/dataset_info.txt",
        f"{base}/probe_gaussian_florence2.csv",
        f"{base}/florence2/caption_features.csv",
        f"{base}/florence2/summary.json",
        f"{base}/florence2/pipeline_variants.csv",
        f"{base}/florence2/rows_scored.csv",
        f"{base}/florence2/sss_grid_mid.csv",
        f"{base}/florence2/theta_det_grid_mid.csv",
    ]


def main():
    VQA_FV.mkdir(parents=True, exist_ok=True)
    say("===== Florence victim E2E start =====")

    if not done("prepare"):
        run([str(PY), "-u", "prepare_eligible.py", "--max_pool", "25000"])
        git_milestone(
            "florence_victim: VQAv2 screening and eligible pool.",
            COMMIT_PATHS_BASE + [
                "BLIP_attack/output/VQA/florence_victim/eligible_pool.json",
                "BLIP_attack/output/VQA/florence_victim/florence_correct_ans_table.json",
            ],
        )
        mark("prepare")

    exp1 = VQA_FV / "exp1_mixed1000"
    if not done("attack_exp1"):
        run_attack_until(exp1, 1000)
        git_milestone("florence_victim: experiment 1 PGD attack log (1000).", paths_exp("exp1_mixed1000"))
        mark("attack_exp1")

    if not done("defense_exp1"):
        run([str(PY), "-u", "run_defense_pipeline.py", "--exp", "exp1"])
        git_milestone("florence_victim: experiment 1 defense metrics.", paths_exp("exp1_mixed1000"))
        mark("defense_exp1")

    if not done("build_exp2"):
        run([str(PY), "-u", "build_exp2.py", "--n", "1000"])
        git_milestone("florence_victim: experiment 2 dataset.", paths_exp("exp2_success1000"))
        mark("build_exp2")

    if not done("defense_exp2"):
        run([str(PY), "-u", "run_defense_pipeline.py", "--exp", "exp2"])
        git_milestone("florence_victim: experiment 2 defense + ablation.", paths_exp("exp2_success1000"))
        mark("defense_exp2")

    master = VQA_FV / "master_attack"
    exp1_log = str(exp1 / "attack_dual_defense_log.json")
    if not done("attack_master_exp3"):
        from build_exp3 import eligible_count

        goal = 1000
        while eligible_count(goal) < goal:
            n = attack_log_count(master / "attack_dual_defense_log.json")
            run_attack_until(master, n + 150, exclude_log=(exp1_log,))
        git_milestone(
            "florence_victim: master attack pool for experiment 3.",
            paths_exp("master_attack"),
        )
        mark("attack_master_exp3")

    if not done("build_exp3"):
        run([str(PY), "-u", "build_exp3.py", "--target", "1000"])
        git_milestone("florence_victim: experiment 3 dataset.", paths_exp("exp3_success1000"))
        mark("build_exp3")

    if not done("defense_exp3"):
        run([str(PY), "-u", "run_defense_pipeline.py", "--exp", "exp3"])
        git_milestone("florence_victim: experiment 3 defense metrics.", paths_exp("exp3_success1000"))
        mark("defense_exp3")

    say("===== Florence victim E2E COMPLETE =====")


if __name__ == "__main__":
    main()
