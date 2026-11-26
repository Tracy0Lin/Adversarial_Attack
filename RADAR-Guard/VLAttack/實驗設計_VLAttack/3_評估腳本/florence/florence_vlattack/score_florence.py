"""Ask Florence (num_beams=3) on the saved VLAttack PNGs and fill attack_success."""
import json
import sys
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "florence_victim"))
from florence_engine import FlorenceEngine, normalize_answer


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--experiment_root", required=True)
    args = p.parse_args()
    root = Path(args.experiment_root)
    log_path = root / "attack_dual_defense_log.json"
    vis = root / "visual_results"
    logs = json.loads(log_path.read_text(encoding="utf-8"))
    todo = [row for row in logs if "adversarial_prediction" not in row]
    print(f"score todo={len(todo)} / {len(logs)}", flush=True)
    if not todo:
        return
    engine = FlorenceEngine("cuda")
    for i, row in enumerate(todo, 1):
        qid = row["question_id"]
        question = row["question_original"]
        gt = normalize_answer(row["ground_truth"])
        clean = Image.open(vis / f"{qid}_1_ori.png").convert("RGB")
        adv = Image.open(vis / f"{qid}_2_atk.png").convert("RGB")
        pred_clean = normalize_answer(engine.vqa(clean, question, num_beams=3))
        pred_adv = normalize_answer(engine.vqa(adv, question, num_beams=3))
        row["ground_truth"] = gt
        row["original_prediction"] = pred_clean
        row["adversarial_prediction"] = pred_adv
        row["attack_success"] = pred_adv != gt
        if i % 20 == 0 or i == len(todo):
            log_path.write_text(json.dumps(logs, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[score] {i}/{len(todo)} qid={qid} success={row['attack_success']}", flush=True)
    log_path.write_text(json.dumps(logs, ensure_ascii=False, indent=2), encoding="utf-8")
    n_ok = sum(1 for row in logs if row.get("attack_success"))
    print(f"scored success={n_ok}/{len(logs)}", flush=True)


if __name__ == "__main__":
    main()
