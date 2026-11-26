"""Run the multi-captioner study to completion; git commit+push after each milestone.

Assumes run_captions for exp1/exp2 may already be in flight (resume via caption cache).
"""
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # BLIP_attack
REPO = ROOT.parent
DATA = ROOT / "output" / "VQA" / "multi_captioner"
SCRIPTS = Path(__file__).resolve().parent
PY = REPO / "vlattack_gpu" / "Scripts" / "python.exe"
if not PY.is_file():
    PY = Path(sys.executable)

CAPTIONERS = ("blip", "vitgpt2", "git", "gemma3")
STATE = DATA / "pipeline_state.json"
LOG = DATA / "pipeline_e2e.log"


def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    DATA.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_state():
    if STATE.is_file():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {"done": []}


def save_state(state):
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def run(cmd, cwd=None, timeout=None):
    log(f"RUN {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, cwd=cwd or SCRIPTS, check=True, timeout=timeout)


def milestone(name):
    return name in load_state()["done"]


def mark(name):
    s = load_state()
    if name not in s["done"]:
        s["done"].append(name)
        save_state(s)


def git_commit(message, paths):
    rel = []
    for p in paths:
        p = Path(p)
        if not p.is_absolute():
            p = REPO / p
        if p.exists():
            rel.append(str(p.relative_to(REPO)).replace("\\", "/"))
    if not rel:
        log(f"git skip (no files): {message}")
        return
    run(["git", "add", "--"] + rel, cwd=REPO)
    st = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=REPO)
    if st.returncode == 0:
        log(f"git nothing to commit: {message}")
        return
    run(["git", "commit", "-m", message], cwd=REPO)
    for attempt in range(8):
        try:
            run(["git", "push", "origin", "HEAD"], cwd=REPO)
            log(f"git pushed: {message}")
            return
        except subprocess.CalledProcessError:
            delay = min(30 * (2 ** attempt), 600)
            log(f"git push failed (attempt {attempt + 1}/8), retry in {delay}s")
            time.sleep(delay)
    raise RuntimeError(f"git push failed after retries: {message}")


def caption_done(exp, cap):
    p = DATA / exp / cap / "caption_features.csv"
    if not p.is_file():
        return False
    n = sum(1 for _ in open(p, encoding="utf-8")) - 1
    want = 1000 if exp == "exp3" else 2000
    return n >= want


def wait_caption_jobs():
    """Wait until no run_captions.py from an earlier manual launch."""
    for _ in range(3600):
        ps = subprocess.run(
            [
                "powershell", "-NoProfile", "-Command",
                "(Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | "
                "Where-Object { $_.CommandLine -match 'run_captions.py' }).Count",
            ],
            capture_output=True,
            text=True,
            cwd=REPO,
        )
        try:
            n = int(ps.stdout.strip() or "0")
        except ValueError:
            n = 0
        if n == 0:
            return
        log(f"waiting for {n} run_captions process(es)...")
        time.sleep(60)
    raise TimeoutError("run_captions still running after 60h")


def ensure_captions(exp, caps):
    for cap in caps:
        if caption_done(exp, cap):
            continue
        run([str(PY), "-u", "run_captions.py", "--exp", exp, "--captioner", cap])


def ensure_analyze(exp, caps):
    for cap in caps:
        if (DATA / exp / cap / "summary.json").is_file():
            continue
        run([str(PY), "-u", "analyze.py", "--exp", exp, "--captioner", cap])


def commit_probe():
    git_commit(
        "multi_captioner: Gaussian probe for exp1 and exp2.",
        [
            "BLIP_attack/output/multi_captioner",
            "BLIP_attack/output/VQA/multi_captioner/PLAN.md",
            "BLIP_attack/output/VQA/multi_captioner/exp1/probe_gaussian.csv",
            "BLIP_attack/output/VQA/multi_captioner/exp2/probe_gaussian.csv",
            "BLIP_attack/output/VQA/multi_captioner/probe_cache.csv",
            "BLIP_attack/output/VQA/multi_captioner/attack_log_snapshot_2298.json",
            "BLIP_attack/output/VQA/multi_captioner/pipeline_state.json",
        ],
    )


def commit_caption(exp, cap):
    git_commit(
        f"multi_captioner: caption features {exp}/{cap}.",
        [
            f"BLIP_attack/output/VQA/multi_captioner/{exp}/{cap}/caption_features.csv",
            f"BLIP_attack/output/VQA/multi_captioner/caption_cache/{cap}.jsonl",
        ],
    )


def commit_analyze(exp, cap):
    d = f"BLIP_attack/output/VQA/multi_captioner/{exp}/{cap}"
    git_commit(
        f"multi_captioner: metrics {exp}/{cap}.",
        [
            f"{d}/summary.json",
            f"{d}/pipeline_variants.csv",
            f"{d}/rows_scored.csv",
            f"{d}/sss_grid_mid.csv",
            f"{d}/theta_det_grid_mid.csv",
        ],
    )


def commit_exp3_dataset():
    git_commit(
        "multi_captioner: experiment 3 dataset manifest and attack log.",
        [
            "BLIP_attack/output/VQA/multi_captioner/exp3_success1000/attack_dual_defense_log.json",
            "BLIP_attack/output/VQA/multi_captioner/exp3_success1000/dataset_manifest.csv",
            "BLIP_attack/output/VQA/multi_captioner/exp3_success1000/dataset_info.txt",
        ],
    )


def commit_final():
    git_commit(
        "multi_captioner: final tables and readable summary.",
        [
            "BLIP_attack/output/VQA/multi_captioner/RESULTS_tables.md",
            "BLIP_attack/output/VQA/multi_captioner/RESULTS_簡明.md",
            "BLIP_attack/output/VQA/multi_captioner/pipeline_state.json",
        ],
    )


def eligible_exp3_count():
    snap = json.loads((DATA / "attack_log_snapshot_2298.json").read_text(encoding="utf-8"))
    run_log = json.loads((ROOT / "output" / "VQA" / "steps100_alpha0.002_run1000" / "attack_dual_defense_log.json").read_text(encoding="utf-8"))
    e1 = {str(x["question_id"]) for x in snap[:1000]}
    e2 = {str(x["question_id"]) for x in json.loads((ROOT / "output" / "VQA" / "exp2_success1000" / "attack_dual_defense_log.json").read_text(encoding="utf-8"))}
    photos = {int(q) // 1000 for q in e1 | e2}
    n = 0
    for item in run_log:
        qid = str(item["question_id"])
        if not item.get("attack_success"):
            continue
        if qid in e1 or qid in e2 or int(qid) // 1000 in photos:
            continue
        n += 1
    return n


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    log("===== end-to-end start =====")

    if not milestone("probe_commit"):
        if (DATA / "exp1" / "probe_gaussian.csv").is_file():
            commit_probe()
        mark("probe_commit")

    if not all(milestone(f"cap_{e}_{c}") for e in ("exp1", "exp2") for c in CAPTIONERS):
        wait_caption_jobs()

    for cap in CAPTIONERS:
        for exp in ("exp1", "exp2"):
            key = f"cap_{exp}_{cap}"
            if not caption_done(exp, cap):
                ensure_captions(exp, (cap,))
            if caption_done(exp, cap) and not milestone(key):
                commit_caption(exp, cap)
                mark(key)

    for cap in CAPTIONERS:
        for exp in ("exp1", "exp2"):
            key = f"analyze_{exp}_{cap}"
            ensure_analyze(exp, (cap,))
            if (DATA / exp / cap / "summary.json").is_file() and not milestone(key):
                try:
                    commit_analyze(exp, cap)
                except RuntimeError as err:
                    if "git push" not in str(err):
                        raise
                    log(f"WARN analyze committed locally, push pending: {exp}/{cap}")
                mark(key)

    if not milestone("exp3_attack"):
        log("starting attack_supervisor for experiment 3")
        run([str(PY), "-u", "attack_supervisor.py", "--poll_sec", "180"], timeout=None)
        mark("exp3_attack")

    if eligible_exp3_count() < 1000:
        raise SystemExit(f"exp3 only has {eligible_exp3_count()} eligible successes; attack incomplete")

    if not milestone("exp3_build"):
        run([str(PY), "-u", "build_exp3.py"])
        commit_exp3_dataset()
        mark("exp3_build")

    if not (DATA / "exp3" / "probe_gaussian.csv").is_file():
        run([str(PY), "-u", "run_probe.py", "--exp", "exp3"])

    for cap in CAPTIONERS:
        key = f"cap_exp3_{cap}"
        if not caption_done("exp3", cap):
            ensure_captions("exp3", (cap,))
        if caption_done("exp3", cap) and not milestone(key):
            commit_caption("exp3", cap)
            mark(key)

    for cap in CAPTIONERS:
        key = f"analyze_exp3_{cap}"
        ensure_analyze("exp3", (cap,))
        if (DATA / "exp3" / cap / "summary.json").is_file() and not milestone(key):
            commit_analyze("exp3", cap)
            mark(key)

    run([str(PY), "-u", "make_tables.py"])
    run([str(PY), "-u", "make_final_report.py"])
    commit_final()
    mark("complete")
    log("===== end-to-end COMPLETE =====")


if __name__ == "__main__":
    main()
