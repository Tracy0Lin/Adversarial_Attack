"""Scan VQAv2 val with Florence VQA; append screening_log.jsonl; build eligible_pool.json."""
import argparse
import json
import os
import sys
from pathlib import Path

import ruamel.yaml as yaml
from PIL import Image

from common import BLIP_ROOT, DATA_ROOT

sys.path.insert(0, str(BLIP_ROOT))
os.chdir(BLIP_ROOT)
from data.utils import pre_question
from florence_engine import FlorenceEngine, normalize_answer
from vqa_match import blip_style_correct


def answer_weights(ann):
    w = {}
    for a in ann["answer"]:
        w[a] = w.get(a, 0.0) + 1.0 / len(ann["answer"])
    return list(w.keys()), list(w.values())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(BLIP_ROOT / "configs" / "vqa.yaml"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--max_pool", type=int, default=5000)
    p.add_argument("--max_scan", type=int, default=None)
    args = p.parse_args()

    os.chdir(BLIP_ROOT)
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    ann_path = Path(config["ann_root"]) / "vqa_val.json"
    anns = json.load(open(ann_path, "r", encoding="utf-8"))
    vqa_root = Path(config["vqa_root"])

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    screen_path = DATA_ROOT / "screening_log.jsonl"
    pool_path = DATA_ROOT / "eligible_pool.json"
    ans_table_path = DATA_ROOT / "florence_correct_ans_table.json"

    engine = FlorenceEngine(args.device)
    eligible = []
    ans_table = {}
    done_ids = set()
    if screen_path.is_file():
        for line in screen_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            done_ids.add(int(rec["question_id"]))
            if rec.get("eligible"):
                eligible.append(int(rec["question_id"]))
                ans_table[str(rec["question_id"])] = rec["pred"]

    n_scan = 0
    with open(screen_path, "a", encoding="utf-8") as slog:
        for ann in anns:
            if ann.get("dataset") != "vqa":
                continue
            qid = int(ann["question_id"])
            if qid in done_ids:
                continue
            n_scan += 1
            if args.max_scan and n_scan > args.max_scan:
                break
            if len(eligible) >= args.max_pool:
                break

            image_path = vqa_root / ann["image"]
            question = pre_question(ann["question"])
            answers, weights = answer_weights(ann)
            pil = Image.open(image_path).convert("RGB")

            pred_raw = engine.vqa(pil, question)
            pred_norm = normalize_answer(pred_raw)
            batch_ans = [[a] for a in answers]
            batch_w = [[w] for w in weights]
            ok = blip_style_correct(pred_norm, batch_ans, batch_w)

            rec = {
                "question_id": qid,
                "question": question,
                "image": ann["image"],
                "pred": pred_norm,
                "pred_raw": pred_raw,
                "eligible": ok,
                "annotated_answers": answers,
            }
            slog.write(json.dumps(rec, ensure_ascii=False) + "\n")
            slog.flush()

            if ok:
                eligible.append(qid)
                ans_table[str(qid)] = pred_norm

            if n_scan % 25 == 0:
                print(f"scan={n_scan} eligible={len(eligible)}", flush=True)

    pool_path.write_text(json.dumps({"question_ids": eligible}, indent=2), encoding="utf-8")
    ans_table_path.write_text(json.dumps(ans_table, indent=2), encoding="utf-8")
    print(f"Done. new_scan={n_scan} total_eligible={len(eligible)}", flush=True)


if __name__ == "__main__":
    main()
