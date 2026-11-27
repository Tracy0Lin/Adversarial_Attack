"""Sanity check: does re-inferring the saved PNGs reproduce the logged predictions?

Confirms that inverting save_adv_image()'s ImageNet de-normalisation recovers the
exact tensor the black-box VQA model saw during the attack. Run before trusting
any delta_ASR number out of sss_denoise.py.
"""
import argparse
import json
from pathlib import Path

from PIL import Image

from sss_denoise import QualityScorer, VQAScorer, gaussian_denoise, vqa_config

parser = argparse.ArgumentParser()
parser.add_argument("--run_dir", required=True)
parser.add_argument("--device", default="cpu")
parser.add_argument("--n", type=int, default=2)
parser.add_argument("--sigma", type=float, default=0.5)
args = parser.parse_args()

run_dir = Path(args.run_dir).resolve()
visual = run_dir / "visual_results"
with open(run_dir / "attack_dual_defense_log.json", "r", encoding="utf-8") as f:
    logs = json.load(f)

cfg = vqa_config()
vqa = VQAScorer(cfg["pretrained"], Path(cfg["ann_root"]) / "answer_list.json", args.device, cfg["k_test"])
quality = QualityScorer(args.device)

ok_ori = ok_atk = checked = 0
for item in logs[: args.n]:
    qid = item["question_id"]
    ori_p, atk_p = visual / f"{qid}_1_ori.png", visual / f"{qid}_2_atk.png"
    if not (ori_p.is_file() and atk_p.is_file()):
        print(f"[skip] {qid} missing png")
        continue
    ori, atk = Image.open(ori_p).convert("RGB"), Image.open(atk_p).convert("RGB")
    q = item["question_original"]
    p_ori, p_atk = vqa.predict(ori, q), vqa.predict(atk, q)
    den = gaussian_denoise(atk, 5, args.sigma)
    p_den = vqa.predict(den, q)
    m = quality.metrics(den, atk)
    checked += 1
    ok_ori += p_ori == item["original_prediction"]
    ok_atk += p_atk == item["adversarial_prediction"]
    print(f"\nqid={qid}  q='{q}'  gt='{item['ground_truth']}'")
    print(f"  ori  logged={item['original_prediction']!r} re-infer={p_ori!r} match={p_ori == item['original_prediction']}")
    print(f"  atk  logged={item['adversarial_prediction']!r} re-infer={p_atk!r} match={p_atk == item['adversarial_prediction']}")
    print(f"  denoised(5,{args.sigma}) pred={p_den!r} rescued={p_den == item['original_prediction']}")
    print(f"  IQP={QualityScorer.iqp(m):.4f} ssim={m['ssim']:.4f} lpips={m['lpips']:.4f} "
          f"brisque={m['brisque']:.2f} lap_ratio={m['lap_ratio']:.4f} clip={m['clip_sim']:.4f}")

print(f"\n=== {checked} samples: ori match {ok_ori}/{checked}, atk match {ok_atk}/{checked} ===")
