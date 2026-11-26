"""Drop exp3 question 263163002 and replace it with one new beams-3 success.

The saved adversarial PNG for 263163002 answers the ground truth when Florence
re-asks with num_beams=3. This script attacks the next eligible question whose
photo is outside exp1, exp2, and exp3, keeps the first PNG-verified success,
and splices it into the exp3 log without reshuffling the other 999 rows.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from common import DATA_ROOT, EXP1_DIR, EXP2_DIR, EXP3_DIR, read_json, write_json

DROP_QID = 263163002
MAX_ATTEMPTS = 8
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PY = ROOT / "vlattack_gpu" / "Scripts" / "python.exe"
IMG_ROOT = Path(r"C:\Users\tracy_aboikg6\OneDrive\Desktop\program\VLAttack_Data\vqa_images")
REPLACEMENT = DATA_ROOT / "exp3_replacement"
POOL_PATH = REPLACEMENT / "candidate_pool.json"
CONFIG = HERE / "vqa_local.yaml"
CUDA_RESTART_EXIT = 75

ENV = os.environ.copy()
ENV["PYTHONPATH"] = str(Path(os.environ.get("LOCALAPPDATA", "")) / "vlattack_florence_pkgs")
ENV["FLORENCE_MODEL_ID"] = "microsoft/Florence-2-large-ft"
ENV["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
ENV["PYTHONUNBUFFERED"] = "1"
ENV["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
ENV["VQA_STUDY"] = "florence_victim"


def photo_id(qid):
    return int(qid) // 1000


def load_logs(path):
    path = Path(path)
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def candidate_pool():
    banned_photos = set()
    used = set()
    for folder in (EXP1_DIR, EXP2_DIR, EXP3_DIR, DATA_ROOT / "master_attack"):
        for item in load_logs(folder / "attack_dual_defense_log.json"):
            qid = int(item["question_id"])
            used.add(qid)
            banned_photos.add(photo_id(qid))
    pool = json.loads((DATA_ROOT / "eligible_pool.json").read_text(encoding="utf-8"))["question_ids"]
    ans = json.loads((DATA_ROOT / "florence_correct_ans_table.json").read_text(encoding="utf-8"))
    picked = []
    for qid in pool:
        qid = int(qid)
        if qid in used or photo_id(qid) in banned_photos:
            continue
        if not str(ans.get(str(qid), "")).strip():
            continue
        img = IMG_ROOT / f"val2014/COCO_val2014_{photo_id(qid):012d}.jpg"
        if not img.is_file():
            continue
        picked.append(qid)
        if len(picked) >= 24:
            break
    if not picked:
        raise SystemExit("no unused-photo candidate with a local COCO image")
    REPLACEMENT.mkdir(parents=True, exist_ok=True)
    POOL_PATH.write_text(json.dumps({"question_ids": picked}, indent=2), encoding="utf-8")
    print(f"candidates={picked}", flush=True)
    return picked


def attack_one_more():
    log_path = REPLACEMENT / "attack_dual_defense_log.json"
    n = len(load_logs(log_path))
    cmd = [
        str(PY), "-u", "attack_florence_pgd.py",
        "--experiment_root", str(REPLACEMENT),
        "--config", str(CONFIG),
        "--pool", str(POOL_PATH),
        "--num_samples", str(n + 1),
    ]
    print("RUN", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=HERE, env=ENV)
    if proc.returncode not in (0, CUDA_RESTART_EXIT):
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    return proc.returncode, load_logs(log_path)


def find_success():
    cuda_retries = 0
    seen = 0
    while seen < MAX_ATTEMPTS:
        before = len(load_logs(REPLACEMENT / "attack_dual_defense_log.json"))
        code, logs = attack_one_more()
        if code == CUDA_RESTART_EXIT or len(logs) == before:
            cuda_retries += 1
            print(f"[cuda-restart] retry {cuda_retries}", flush=True)
            if cuda_retries > 3:
                raise SystemExit("CUDA restart did not recover")
            continue
        cuda_retries = 0
        seen += 1
        entry = logs[-1]
        print(
            f"attempt {seen}: qid={entry['question_id']} success={entry['attack_success']} "
            f"adv={entry['adversarial_prediction']!r} gt={entry['ground_truth']!r}",
            flush=True,
        )
        if entry.get("attack_success"):
            return entry
    raise SystemExit(f"no beams-3 PNG success in {MAX_ATTEMPTS} attempts")


def splice(entry):
    exp3_log = EXP3_DIR / "attack_dual_defense_log.json"
    rows = [x for x in read_json(exp3_log) if int(x["question_id"]) != DROP_QID]
    if len(rows) != 999:
        raise SystemExit(f"expected 999 rows after drop, got {len(rows)}")
    if any(int(x["question_id"]) == int(entry["question_id"]) for x in rows):
        raise SystemExit("replacement qid already in exp3")
    rows.append(entry)
    write_json(exp3_log, rows)

    vis = EXP3_DIR / "visual_results"
    for sfx in ("_1_ori.png", "_2_atk.png"):
        old = vis / f"{DROP_QID}{sfx}"
        if old.is_file():
            old.unlink()
        shutil.copy2(REPLACEMENT / "visual_results" / f"{entry['question_id']}{sfx}", vis / f"{entry['question_id']}{sfx}")

    import pandas as pd
    pd.DataFrame(
        [{"question_id": it["question_id"], "photo_id": photo_id(it["question_id"])} for it in rows]
    ).to_csv(EXP3_DIR / "dataset_manifest.csv", index=False, encoding="utf-8")
    (EXP3_DIR / "dataset_info.txt").write_text(
        "source=exp3_success1000 log after one-for-one replacement\n"
        f"dropped={DROP_QID} reason=beams3 re-ask of saved PNG matches ground_truth yes\n"
        f"added={entry['question_id']} photo={photo_id(entry['question_id'])} "
        f"gt={entry['ground_truth']!r} adv={entry['adversarial_prediction']!r}\n"
        f"selected={len(rows)}\n",
        encoding="utf-8",
    )
    exclude_path = DATA_ROOT / "exp3_exclude_qids.json"
    write_json(exclude_path, {"question_ids": [DROP_QID], "reason": "beams3 PNG re-ask equals ground_truth"})

    master_log = DATA_ROOT / "master_attack" / "attack_dual_defense_log.json"
    master = read_json(master_log)
    if not any(int(x["question_id"]) == int(entry["question_id"]) for x in master):
        master.append(entry)
        write_json(master_log, master)
    master_vis = DATA_ROOT / "master_attack" / "visual_results"
    master_vis.mkdir(parents=True, exist_ok=True)
    for sfx in ("_1_ori.png", "_2_atk.png"):
        src = vis / f"{entry['question_id']}{sfx}"
        dst = master_vis / f"{entry['question_id']}{sfx}"
        if src.is_file():
            shutil.copy2(src, dst)
    print(f"spliced qid={entry['question_id']} into exp3 ({len(rows)} rows)", flush=True)


def run_defense():
    mc = HERE.parent / "multi_captioner"
    py = str(PY)
    commands = [
        [py, "-u", "run_probe.py", "--exp", "exp3", "--vqa-backend", "florence2"],
        [py, "-u", "run_captions.py", "--exp", "exp3", "--captioner", "florence2"],
        [py, "-u", "analyze.py", "--exp", "exp3", "--captioner", "florence2", "--vqa-backend", "florence2"],
        [py, "-u", "analyze.py", "--exp", "exp3", "--captioner", "florence2", "--vqa-backend", "florence2",
         "--output-tag", "sigma08_fixed", "--fix-theta-star", "0.8", "--fix-theta-det", "0.8"],
    ]
    for cmd in commands:
        print("RUN", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=mc, env=ENV, check=True)


def main():
    candidate_pool()
    entry = find_success()
    splice(entry)
    run_defense()
    print("DONE", entry["question_id"], flush=True)


if __name__ == "__main__":
    main()
