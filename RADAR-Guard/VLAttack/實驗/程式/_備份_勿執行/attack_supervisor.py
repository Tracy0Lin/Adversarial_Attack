"""Keep VLAttack running until experiment 3 has enough photo-disjoint successes.

attack_vqa.py resumes from attack_dual_defense_log.json but rewrites the whole file
after every sample. A crash mid-write would leave invalid JSON, which the attack
treats as "start fresh" and then overwrites. So every poll keeps a last-good copy
and restores it before any relaunch.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
BLIP_ROOT = THIS_DIR.parents[1]
VQA_OUT = BLIP_ROOT / "output" / "VQA"
RUN_DIR = VQA_OUT / "steps100_alpha0.002_run1000"
LOG_PATH = RUN_DIR / "attack_dual_defense_log.json"
EXP2_LOG = VQA_OUT / "exp2_success1000" / "attack_dual_defense_log.json"
OUT_DIR = VQA_OUT / "multi_captioner"
SNAPSHOT = OUT_DIR / "attack_log_snapshot_2298.json"
GOOD_COPY = OUT_DIR / "attack_log_last_good.json"
SUPERVISOR_LOG = OUT_DIR / "attack_supervisor.log"
LOCK = OUT_DIR / "attack_supervisor.lock"
EXP1_N = 1000
STALL_POLLS = 5  # no new log rows for this many polls (~15 min @ 180s) -> kill and restart


def say(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(SUPERVISOR_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def excluded_sets():
    snap = read_json(SNAPSHOT)
    exp1 = [str(x["question_id"]) for x in snap[:EXP1_N]]
    exp2 = [str(x["question_id"]) for x in read_json(EXP2_LOG)]
    qids = set(exp1) | set(exp2)
    photos = {int(q) // 1000 for q in qids}
    return qids, photos


def eligible(logs, qids, photos):
    out = []
    for item in logs:
        qid = str(item["question_id"])
        if not item.get("attack_success"):
            continue
        if qid in qids or int(qid) // 1000 in photos:
            continue
        out.append(qid)
    return out


def check_log(attack_running=False):
    """Return parsed log, restoring the last good copy if the live file is broken.

    While the attack runs, an unreadable file is usually a write in progress, so
    fall back to the good copy without touching the live file.
    """
    try:
        logs = read_json(LOG_PATH)
    except (OSError, ValueError) as err:
        if attack_running:
            return read_json(GOOD_COPY)
        say(f"live log unreadable ({err}); restoring {GOOD_COPY.name}")
        shutil.copy2(GOOD_COPY, LOG_PATH)
        return read_json(LOG_PATH)
    # Dump the parsed copy rather than copying the file: the attack may rewrite it
    # between our read and a file copy.
    tmp = GOOD_COPY.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False)
    tmp.replace(GOOD_COPY)
    return logs


def acquire_lock():
    if LOCK.is_file():
        try:
            pid = int(LOCK.read_text(encoding="utf-8").strip())
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command", f"Get-Process -Id {pid} -ErrorAction SilentlyContinue"],
                capture_output=True,
            )
            if out.returncode == 0:
                say(f"another supervisor pid={pid} is running; exit")
                raise SystemExit(0)
        except (ValueError, OSError):
            pass
    LOCK.write_text(str(os.getpid()), encoding="utf-8")


def release_lock():
    LOCK.unlink(missing_ok=True)


def kill_orphan_attacks():
    subprocess.run(
        [
            "powershell", "-NoProfile", "-Command",
            "Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | "
            "Where-Object { $_.CommandLine -match 'attack_vqa.py' } | "
            "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }",
        ],
        check=False,
    )


def launch(num_samples, stdout_path):
    kill_orphan_attacks()
    cmd = [
        sys.executable, "-u", "attack_vqa.py",
        "--method", "VLAttack", "--steps", "100", "--alphas", "0.002",
        "--run_suffix", "_run1000", "--num_samples", str(num_samples),
    ]
    out = open(stdout_path, "a", encoding="utf-8")
    err = open(str(stdout_path).replace(".log", ".err.log"), "a", encoding="utf-8")
    say(f"launch: {' '.join(cmd[2:])}")
    return subprocess.Popen(cmd, cwd=str(BLIP_ROOT), stdout=out, stderr=err), out, err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=1000)
    parser.add_argument("--margin", type=int, default=15, help="extra successes in case of missing images")
    parser.add_argument("--max_samples", type=int, default=6000)
    parser.add_argument("--poll_sec", type=int, default=120)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not SNAPSHOT.is_file():
        shutil.copy2(LOG_PATH, SNAPSHOT)
        say(f"snapshot saved: {SNAPSHOT.name} ({len(read_json(SNAPSHOT))} entries)")
    qids, photos = excluded_sets()
    say(f"excluded qids={len(qids)} photos={len(photos)}")

    acquire_lock()
    need = args.target + args.margin
    stdout_path = OUT_DIR / "attack_phase.log"
    try:
        _main_loop(args, need, stdout_path, qids, photos)
    finally:
        release_lock()


def _main_loop(args, need, stdout_path, qids, photos):
    proc = None
    restarts = 0
    last_len = -1
    stall_polls = 0
    while True:
        logs = check_log(attack_running=proc is not None and proc.poll() is None)
        n_ok = len(eligible(logs, qids, photos))
        if proc is not None and proc.poll() is None:
            if len(logs) == last_len:
                stall_polls += 1
            else:
                stall_polls = 0
                last_len = len(logs)
            if stall_polls >= STALL_POLLS:
                say(f"stall: log stuck at {len(logs)} rows for {stall_polls} polls; killing attack")
                proc.kill()
                proc.wait(timeout=60)
                proc = None
                stall_polls = 0
                restarts += 1
                continue
        if proc is None or proc.poll() is not None:
            if proc is not None:
                say(f"attack exited code={proc.returncode}; log={len(logs)} eligible={n_ok}")
                restarts += 1
            if n_ok >= need:
                break
            if len(logs) >= args.max_samples:
                say("reached max_samples without enough successes; stopping")
                break
            if restarts > 50:
                say("too many restarts; stopping")
                break
            proc, fo, fe = launch(args.max_samples, stdout_path)
        else:
            say(f"progress log={len(logs)} eligible={n_ok}/{need}")
            if n_ok >= need:
                say("target reached; terminating attack")
                proc.terminate()
                proc.wait(timeout=120)
                check_log()
                break
        time.sleep(args.poll_sec)
    logs = check_log()
    say(f"DONE log={len(logs)} eligible={len(eligible(logs, qids, photos))}")


if __name__ == "__main__":
    main()
