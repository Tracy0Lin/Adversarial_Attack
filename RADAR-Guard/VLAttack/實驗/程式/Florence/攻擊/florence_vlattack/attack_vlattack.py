"""Same image attack as attack_vqa.py --method VLAttack, scored later by Florence.

Active path in adv_attack_blip_vla.evaluate: BLIP-pretrain BSA PGD,
L-inf 0.125, alpha 0.002, 100 steps. The text attack in that file is commented out,
so the question is unchanged. Success is not decided here.
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.utils import save_image

HERE = Path(__file__).resolve().parent
BLIP_ROOT = HERE.parents[1]
REPO = BLIP_ROOT.parent
sys.path.insert(0, str(REPO / "cleverhans"))
sys.path.insert(0, str(BLIP_ROOT))
os.chdir(BLIP_ROOT)

from data.utils import pre_question
from models.blip_pretrain import blip_pretrain
import cleverhans.torch.attacks.BLIP.projected_gradient_descent as pgd

IMG_ROOT = Path(os.environ.get("VLATTACK_IMG_ROOT", r"C:\Users\tracy_aboikg6\OneDrive\Desktop\program\VLAttack_Data\vqa_images"))
POOL = BLIP_ROOT / "output" / "VQA" / "florence_victim" / "eligible_pool.json"
SCREEN = BLIP_ROOT / "output" / "VQA" / "florence_victim" / "screening_log.jsonl"
ANS = BLIP_ROOT / "output" / "VQA" / "florence_victim" / "florence_correct_ans_table.json"
WEIGHTS = Path(os.environ.get("VLATTACK_BLIP_PRETRAIN", r"C:\Users\tracy_aboikg6\OneDrive\Desktop\program\VLAttack_Data\models\model_base.pth"))
CUDA_RESTART_EXIT = 75


def save_adv_image(img_tensor, path):
    """Same ImageNet de-norm as adv_attack_blip_vla.save_adv_image."""
    mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device).view(1, 3, 1, 1)
    img = img_tensor * std + mean
    save_image(img, path)


def load_questions():
    out = {}
    with open(SCREEN, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[int(rec["question_id"])] = rec
    return out


def read_log(path):
    if not path.is_file():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def write_log(path, rows):
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def make_transform():
    return transforms.Compose([
        transforms.Resize((480, 480), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def pgd_fn(model, question):
    def _fn(x):
        if random.random() > 0.5:
            orig = x.shape[-2:]
            rnd = random.randint(int(orig[0] * 0.9), orig[0])
            resized = F.interpolate(x, size=(rnd, rnd), mode="bilinear")
            pad_top = random.randint(0, orig[0] - rnd)
            pad_left = random.randint(0, orig[1] - rnd)
            x = F.pad(resized, (pad_left, orig[1] - rnd - pad_left, pad_top, orig[0] - rnd - pad_top))
        img_feats_list, txt_feats_list = model.Gen_feats(x, question)
        return [torch.cat(txt_feats_list, axis=0), torch.cat(img_feats_list, axis=0)]
    return _fn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment_root", required=True)
    p.add_argument("--num_samples", type=int, default=1000)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--alpha", type=float, default=0.002)
    p.add_argument("--epsilon", type=float, default=0.125)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    root = Path(args.experiment_root)
    vis = root / "visual_results"
    vis.mkdir(parents=True, exist_ok=True)
    log_path = root / "attack_dual_defense_log.json"
    logs = read_log(log_path)
    done = {int(x["question_id"]) for x in logs}

    pool = json.loads(POOL.read_text(encoding="utf-8"))["question_ids"]
    screen = load_questions()
    ans = json.loads(ANS.read_text(encoding="utf-8"))
    transform = make_transform()

    print(f"loading BLIP pretrain {WEIGHTS}", flush=True)
    model = blip_pretrain(
        pretrained=str(WEIGHTS), image_size=480, vit="base",
        vit_grad_ckpt=True, vit_ckpt_layer=0, queue_size=1,
    ).to("cuda").eval()

    for qid in pool:
        if len(logs) >= args.num_samples:
            break
        qid = int(qid)
        if qid in done:
            continue
        rec = screen.get(qid)
        if not rec:
            continue
        correct = str(ans.get(str(qid), "")).strip()
        if not correct:
            continue
        image_path = IMG_ROOT / rec["image"]
        if not image_path.is_file():
            continue
        question = pre_question(rec["question"])
        try:
            torch.cuda.empty_cache()
            pil = Image.open(image_path).convert("RGB")
            ori = transform(pil).unsqueeze(0).to("cuda")
            attack = pgd_fn(model, question)
            with torch.no_grad():
                img_list, txt_list = model.Gen_feats(ori, question)
                ori_img_feats = torch.cat(img_list, axis=0)
                ori_txt_feats = torch.cat(txt_list, axis=0)
            torch.set_grad_enabled(True)
            adv, _ = pgd.projected_gradient_descent(
                attack, ori.clone(), args.epsilon, args.alpha, args.steps,
                np.inf, clip_min=-1.0, clip_max=1.0,
                y=[ori_txt_feats, ori_img_feats, None, None, None],
                time=0, ori_x=ori, method="BSA",
            )
            torch.set_grad_enabled(False)
            save_adv_image(ori, vis / f"{qid}_1_ori.png")
            save_adv_image(adv, vis / f"{qid}_2_atk.png")
        except RuntimeError as err:
            msg = str(err).lower()
            if "out of memory" in msg or "cuda" in msg:
                print(f"[cuda-restart] qid={qid}: {err}", flush=True)
                torch.cuda.empty_cache()
                sys.exit(CUDA_RESTART_EXIT)
            print(f"[skip] qid={qid}: {err}", flush=True)
            continue

        logs.append({
            "question_id": qid,
            "question_original": question,
            "question_attack": question,
            "ground_truth": correct,
            "attack_type": "VLAttack",
            "pgd_steps": args.steps,
            "pgd_alpha": args.alpha,
            "pgd_epsilon": args.epsilon,
            "image": rec["image"],
        })
        write_log(log_path, logs)
        print(f"[{len(logs)}/{args.num_samples}] qid={qid}", flush=True)
        del ori, adv
        torch.cuda.empty_cache()

    print(f"Saved {log_path} ({len(logs)} entries)", flush=True)


if __name__ == "__main__":
    main()
