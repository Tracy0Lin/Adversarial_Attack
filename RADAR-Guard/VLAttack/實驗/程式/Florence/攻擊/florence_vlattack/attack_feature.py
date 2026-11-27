"""Feature-distance PGD on pretrained Florence-2-large.

Pushes DaViT block features away from the original image. The question and
the fine-tuned VQA model are not used. Images are scored later.
"""
import argparse
import gc
import json
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.utils import save_image

HERE = Path(__file__).resolve().parent
SIDE = Path(os.environ.get("LOCALAPPDATA", "")) / "vlattack_florence_pkgs"
if SIDE.is_dir() and str(SIDE) not in sys.path:
    sys.path.insert(0, str(SIDE))

IMG_ROOT = Path(os.environ.get("VLATTACK_IMG_ROOT", r"C:\Users\tracy_aboikg6\OneDrive\Desktop\program\VLAttack_Data\vqa_images"))
POOL = HERE.parent / "VQA" / "florence_victim" / "eligible_pool.json"
SCREEN = HERE.parent / "VQA" / "florence_victim" / "screening_log.jsonl"
ANS = HERE.parent / "VQA" / "florence_victim" / "florence_correct_ans_table.json"
PRETRAIN = "microsoft/Florence-2-large"
CUDA_RESTART_EXIT = 75


def load_questions():
    out = {}
    with open(SCREEN, encoding="utf-8") as f:
        for line in f:
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


def maybe_resize(x):
    """Same 50% random resize-and-pad as the BLIP VLAttack step."""
    if random.random() > 0.5:
        return x
    _, _, height, width = x.shape
    rnd = random.randint(int(height * 0.9), height)
    resized = F.interpolate(x, size=(rnd, rnd), mode="bilinear", align_corners=False)
    pad_top = random.randint(0, height - rnd)
    pad_left = random.randint(0, width - rnd)
    return F.pad(resized, (pad_left, width - rnd - pad_left, pad_top, height - rnd - pad_top))


def block_feats(vision, pixel_values, augment):
    x = maybe_resize(pixel_values) if augment else pixel_values
    size = (x.size(2), x.size(3))
    feats = []
    for conv, block in zip(vision.convs, vision.blocks):
        x, size = conv(x, size)
        for sub in block:
            x, size = sub(x, size)
            feats.append(x)
    return feats


def bsa_loss(adv_feats, ori_feats):
    loss = adv_feats[0].new_zeros(())
    for adv, ori in zip(adv_feats, ori_feats):
        loss = loss + (-F.cosine_similarity(adv.float(), ori.float(), dim=-1)).sum()
    return loss


def pixels_to_unit(pixel_values, mean, std):
    x = pixel_values.detach().float().cpu()[0]
    x = (x * std + mean).clamp(0.0, 1.0)
    return x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment_root", required=True)
    p.add_argument("--num_samples", type=int, default=20)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--alpha", type=float, default=0.002)
    p.add_argument("--epsilon", type=float, default=0.125)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    from transformers import AutoModelForCausalLM, AutoProcessor

    root = Path(args.experiment_root)
    vis = root / "visual_results"
    vis.mkdir(parents=True, exist_ok=True)
    log_path = root / "attack_dual_defense_log.json"
    logs = read_log(log_path)
    done = {int(row["question_id"]) for row in logs}
    skip_path = root / "cuda_skip.json"
    skipped = set()
    if skip_path.is_file():
        skipped = {int(x) for x in json.loads(skip_path.read_text(encoding="utf-8"))}

    pool = json.loads(POOL.read_text(encoding="utf-8"))["question_ids"]
    screen = load_questions()
    ans = json.loads(ANS.read_text(encoding="utf-8"))

    print(f"loading {PRETRAIN}", flush=True)
    processor = AutoProcessor.from_pretrained(PRETRAIN, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        PRETRAIN, trust_remote_code=True, torch_dtype=torch.float16,
    ).to("cuda").eval()
    vision = model.vision_tower
    ip = processor.image_processor
    mean = torch.tensor(ip.image_mean, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(ip.image_std, dtype=torch.float32).view(3, 1, 1)

    for qid in pool:
        if len(logs) >= args.num_samples:
            break
        qid = int(qid)
        if qid in done or qid in skipped:
            continue
        rec = screen.get(qid)
        correct = str(ans.get(str(qid), "")).strip()
        if not rec or not correct:
            continue
        image_path = IMG_ROOT / rec["image"]
        if not image_path.is_file():
            continue
        try:
            torch.cuda.empty_cache()
            pil = Image.open(image_path).convert("RGB")
            ori = processor(images=pil, return_tensors="pt")["pixel_values"].to("cuda").float()
            with torch.no_grad():
                ori_feats = [feat.detach() for feat in block_feats(vision, ori.half(), augment=False)]
            eta = torch.empty_like(ori).uniform_(-args.epsilon, args.epsilon)
            adv = (ori + eta).clamp(ori - args.epsilon, ori + args.epsilon)
            adv = adv.detach().requires_grad_(True)
            last_loss = None
            for step in range(args.steps):
                if adv.grad is not None:
                    adv.grad = None
                feats = block_feats(vision, adv.half(), augment=True)
                loss = bsa_loss(feats, ori_feats)
                loss.backward()
                grad = adv.grad
                if grad is None:
                    raise RuntimeError("feature gradient is missing")
                with torch.no_grad():
                    stepped = adv + args.alpha * grad.sign()
                    adv_next = torch.max(torch.min(stepped, ori + args.epsilon), ori - args.epsilon)
                last_loss = float(loss.detach())
                del feats, loss, grad
                adv = adv_next.detach().requires_grad_(True)
                if step == 0 or step == args.steps - 1:
                    print(f"  qid={qid} step={step} loss={last_loss:.1f}", flush=True)
            save_image(pixels_to_unit(ori, mean, std), vis / f"{qid}_1_ori.png")
            save_image(pixels_to_unit(adv, mean, std), vis / f"{qid}_2_atk.png")
        except RuntimeError as err:
            msg = str(err).lower()
            if "out of memory" in msg or "cuda" in msg:
                print(f"[cuda-skip] qid={qid}: {err}", flush=True)
                skipped.add(qid)
                skip_path.write_text(json.dumps(sorted(skipped)), encoding="utf-8")
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                sys.exit(CUDA_RESTART_EXIT)
            print(f"[skip] qid={qid}: {err}", flush=True)
            continue

        logs.append({
            "question_id": qid,
            "question_original": rec["question"].strip(),
            "question_attack": rec["question"].strip(),
            "ground_truth": correct,
            "attack_type": "FlorenceVLAttack",
            "surrogate": PRETRAIN,
            "pgd_steps": args.steps,
            "pgd_alpha": args.alpha,
            "pgd_epsilon": args.epsilon,
            "image": rec["image"],
        })
        write_log(log_path, logs)
        print(f"[{len(logs)}/{args.num_samples}] qid={qid}", flush=True)
        del ori, adv
        gc.collect()
        torch.cuda.empty_cache()

    print(f"Saved {log_path} ({len(logs)} entries)", flush=True)


if __name__ == "__main__":
    main()
