"""Step 1: pretrained Florence block features, then -ft VQA vs ground_truth.

Loads one checkpoint at a time. Times a few feature-distance backward steps
so later attack estimates use a measured seconds-per-step.
"""
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).resolve().parent
VICTIM = HERE.parent / "florence_victim"
sys.path.insert(0, str(VICTIM))

SIDE = Path(os.environ.get("LOCALAPPDATA", "")) / "vlattack_florence_pkgs"
if SIDE.is_dir() and str(SIDE) not in sys.path:
    sys.path.insert(0, str(SIDE))

IMG_ROOT = Path(r"C:\Users\tracy_aboikg6\OneDrive\Desktop\program\VLAttack_Data\vqa_images")
SCREEN = HERE.parent / "VQA" / "florence_victim" / "screening_log.jsonl"
ANS = HERE.parent / "VQA" / "florence_victim" / "florence_correct_ans_table.json"
PRETRAIN = "microsoft/Florence-2-large"
FT = "microsoft/Florence-2-large-ft"


def stage_feats(vision, pixel_values):
    feats = []
    x = pixel_values
    size = (x.size(2), x.size(3))
    for conv, block in zip(vision.convs, vision.blocks):
        x, size = conv(x, size)
        x, size = block(x, size)
        feats.append(x)
    return feats


def cosine_push(adv_feats, ori_feats):
    loss = adv_feats[0].new_zeros(())
    for adv, ori in zip(adv_feats, ori_feats):
        a = adv.flatten(1).float()
        b = ori.flatten(1).float()
        loss = loss + (-F.cosine_similarity(a, b, dim=1)).sum()
    return loss


def main():
    from transformers import AutoModelForCausalLM, AutoProcessor

    screen = {}
    with open(SCREEN, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            screen[int(rec["question_id"])] = rec
    ans = json.loads(ANS.read_text(encoding="utf-8"))
    qids = [262148000, 262148001, 262148002, 393225000, 393225001]

    print("load", PRETRAIN, flush=True)
    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(PRETRAIN, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        PRETRAIN, trust_remote_code=True, torch_dtype=torch.float16,
    ).to("cuda").eval()
    vision = model.vision_tower
    print(f"pretrain_load_s {time.perf_counter() - t0:.1f}", flush=True)
    print(f"depths {vision.embed_dims} blocks {[len(b) for b in vision.blocks]}", flush=True)

    rec = screen[qids[0]]
    pil = Image.open(IMG_ROOT / rec["image"]).convert("RGB")
    px = processor(images=pil, return_tensors="pt")["pixel_values"].to("cuda")
    print("pixel", tuple(px.shape), px.dtype, flush=True)

    with torch.no_grad():
        ori = [f.detach() for f in stage_feats(vision, px.half())]
    for i, f in enumerate(ori):
        print(f"stage{i}", tuple(f.shape), f.dtype, flush=True)

    adv = px.detach().float().requires_grad_(True)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    n_steps = 5
    for step in range(n_steps):
        if adv.grad is not None:
            adv.grad = None
        feats = stage_feats(vision, adv.half())
        loss = cosine_push(feats, ori)
        loss.backward()
        with torch.no_grad():
            adv = (adv + 0.002 * adv.grad.sign()).clamp(px.float() - 0.125, px.float() + 0.125)
        adv = adv.detach().requires_grad_(True)
        print(f"step {step} loss {float(loss):.4f}", flush=True)
        del feats, loss
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t1
    peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"backward_steps {n_steps} elapsed_s {elapsed:.2f} sec_per_step {elapsed / n_steps:.3f} peak_mib {peak:.0f}", flush=True)

    del adv, ori, model, vision, processor
    gc.collect()
    torch.cuda.empty_cache()

    os.environ["FLORENCE_MODEL_ID"] = FT
    from florence_engine import FlorenceEngine, normalize_answer

    print("load", FT, flush=True)
    t2 = time.perf_counter()
    engine = FlorenceEngine("cuda")
    print(f"ft_load_s {time.perf_counter() - t2:.1f}", flush=True)
    ok = 0
    for qid in qids:
        rec = screen[qid]
        pil = Image.open(IMG_ROOT / rec["image"]).convert("RGB")
        tq = time.perf_counter()
        pred = engine.vqa(pil, rec["question"], num_beams=3)
        gt = normalize_answer(ans.get(str(qid), ""))
        match = normalize_answer(pred) == gt
        ok += int(match)
        print(
            f"vqa qid={qid} beams=3 pred={pred!r} gt={gt!r} match={match} "
            f"sec={time.perf_counter() - tq:.2f}",
            flush=True,
        )
    print(f"vqa_match {ok}/{len(qids)}", flush=True)


if __name__ == "__main__":
    main()
