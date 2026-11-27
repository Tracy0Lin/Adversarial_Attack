"""PGD on Florence-2-large-ft (white-box = victim). Writes BLIP-compatible attack log + PNGs."""
import argparse
import json
import os
import sys
from pathlib import Path

import gc
import ruamel.yaml as yaml
import torch
from PIL import Image

from common import DATA_ROOT, BLIP_ROOT, write_json

sys.path.insert(0, str(BLIP_ROOT))
os.chdir(BLIP_ROOT)
from data.utils import pre_question
from florence_engine import FlorenceEngine, normalize_answer

# Parent E2E re-runs this process to clear a poisoned CUDA context after OOM.
CUDA_RESTART_EXIT = 75


def _recoverable_cuda_error(err: BaseException) -> bool:
    if isinstance(err, torch.cuda.OutOfMemoryError):
        return True
    if isinstance(err, RuntimeError):
        msg = str(err).lower()
        return any(
            s in msg
            for s in ("out of memory", "cublas", "cuda error", "memory allocation")
        )
    return False


def _cuda_recover():
    gc.collect()
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize()
    except RuntimeError:
        pass
    torch.cuda.empty_cache()


def load_ann_index(config):
    """Image + question per qid; prefer screening_log.jsonl (low RAM) over full vqa_val.json."""
    screen_path = DATA_ROOT / "screening_log.jsonl"
    if screen_path.is_file():
        out = {}
        with open(screen_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                qid = int(rec["question_id"])
                out[qid] = {
                    "question_id": qid,
                    "image": rec["image"],
                    "question": rec["question"],
                }
        if out:
            return out
    ann_path = Path(config["ann_root"]) / "vqa_val.json"
    anns = json.load(open(ann_path, "r", encoding="utf-8"))
    return {int(a["question_id"]): a for a in anns if a.get("dataset") == "vqa"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment_root", required=True)
    p.add_argument("--config", default=str(BLIP_ROOT / "configs" / "vqa.yaml"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--num_samples", type=int, default=1000)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--alpha", type=float, default=0.002)
    p.add_argument("--epsilon", type=float, default=0.125, help="L_inf on processor pixel_values scale")
    p.add_argument("--pool", default=str(DATA_ROOT / "eligible_pool.json"))
    p.add_argument("--ans_table", default=str(DATA_ROOT / "florence_correct_ans_table.json"))
    p.add_argument("--exclude_log", nargs="*", default=[], help="question_ids to skip (other experiments)")
    args = p.parse_args()

    os.chdir(BLIP_ROOT)
    root = Path(args.experiment_root).resolve()
    vis = root / "visual_results"
    vis.mkdir(parents=True, exist_ok=True)
    log_path = root / "attack_dual_defense_log.json"

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    vqa_root = Path(config["vqa_root"])
    ann_by_qid = load_ann_index(config)
    pool = json.load(open(args.pool, "r", encoding="utf-8"))["question_ids"]
    ans_table = json.load(open(args.ans_table, "r", encoding="utf-8"))

    logs = []
    if log_path.is_file():
        try:
            logs = json.load(open(log_path, "r", encoding="utf-8"))
        except json.JSONDecodeError:
            logs = []

    done = {int(x["question_id"]) for x in logs}
    for ex in args.exclude_log:
        ex_path = Path(ex)
        if ex_path.is_file():
            for row in json.load(open(ex_path, "r", encoding="utf-8")):
                done.add(int(row["question_id"]))
    engine = FlorenceEngine(args.device)

    for qid in pool:
        if len(logs) >= args.num_samples:
            break
        if int(qid) in done:
            continue
        ann = ann_by_qid.get(int(qid))
        if not ann:
            continue
        question = pre_question(ann["question"])
        correct = normalize_answer(ans_table.get(str(qid), ""))
        if not correct:
            continue

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            pil = Image.open(vqa_root / ann["image"]).convert("RGB")
            ori_path = vis / f"{qid}_1_ori.png"
            atk_path = vis / f"{qid}_2_atk.png"
            pil.save(ori_path)

            input_ids, ori_px = engine.encode_vqa(pil, question)
            ori_px = ori_px.detach()
            adv_px = ori_px.clone().requires_grad_(True)

            for _ in range(args.steps):
                if adv_px.grad is not None:
                    adv_px.grad.zero_()
                loss = engine.vqa_teacher_loss(adv_px, input_ids, correct)
                loss.backward()
                with torch.no_grad():
                    step = args.alpha * adv_px.grad.sign()
                    adv_px = adv_px + step
                    adv_px = torch.max(torch.min(adv_px, ori_px + args.epsilon), ori_px - args.epsilon)
                adv_px = adv_px.detach().requires_grad_(True)
                del loss

            with torch.no_grad():
                size = (pil.width, pil.height)
                pred_clean = engine.vqa_on_pixels(input_ids, ori_px, question, size, num_beams=3)
                adv_pil = engine.pixels_to_pil(adv_px)
                adv_pil.save(atk_path)
                # Probe re-encodes the saved PNG. Pixel-space beam search can disagree
                # with that re-ask, so success follows the file that later stages read.
                pred_adv = engine.vqa(Image.open(atk_path).convert("RGB"), question, num_beams=3)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as err:
            if not _recoverable_cuda_error(err):
                raise
            print(f"[cuda-restart] qid={qid}: {err}", flush=True)
            _cuda_recover()
            sys.exit(CUDA_RESTART_EXIT)
        except Exception as err:
            print(f"[skip] qid={qid} failed: {err}", flush=True)
            _cuda_recover()
            continue

        entry = {
            "question_id": int(qid),
            "question_original": question,
            "question_attack": question,
            "ground_truth": correct,
            "original_prediction": normalize_answer(pred_clean) or correct,
            "adversarial_prediction": normalize_answer(pred_adv),
            "attack_success": normalize_answer(pred_adv) != correct,
            "attack_type": "FlorencePGD",
            "pgd_steps": args.steps,
            "pgd_alpha": args.alpha,
            "pgd_epsilon": args.epsilon,
        }
        logs.append(entry)
        write_json(log_path, logs)
        print(f"[{len(logs)}/{args.num_samples}] qid={qid} success={entry['attack_success']}", flush=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"Saved {log_path} ({len(logs)} entries)", flush=True)


if __name__ == "__main__":
    main()
