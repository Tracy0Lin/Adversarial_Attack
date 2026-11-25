#!/usr/bin/env python3

import gc
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT = Path(
    "/workspace/projects/couser/"
    "AttackVLM_Official_Reproduction"
)
EXP = (
    PROJECT /
    "formal_tests/attackvlm_three_scenarios_v1/"
    "vitgpt2/experiment2_stress"
)
OUT = EXP / "tcvad/stage2_medium"
OLD_ROOT = (
    PROJECT /
    "formal_tests/paper_vitgpt2_coco1000_v1/"
    "formal1000_v1"
)

ROUTING = (
    EXP /
    "tcvad/routing/experiment2_risk_routing_all2000.json"
)
CAPTIONS = (
    EXP /
    "tcvad/stage1/captions/"
    "vitgpt2_21captions_experiment2_1000pairs.json"
)
OLD_RESULTS = OLD_ROOT / "sss_bilateral_strong/results.json"
HELPER_PATH = (
    PROJECT /
    "work_code/"
    "run_small_caption_sss_bilateral_strong_formal1000.py"
)

FILTER = {
    "name": "bilateral_strong",
    "method": "bilateral",
    "diameter": 7,
    "sigma_color": 50.0,
    "sigma_space": 7.0,
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


spec = importlib.util.spec_from_file_location(
    "sss_helper",
    HELPER_PATH,
)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)

samples = {
    r["selection_id"]: r
    for r in json.loads(
        (EXP / "experiment2_manifest.json").read_text()
    )["samples"]
}

routing = json.loads(ROUTING.read_text())["records"]
medium = [
    r for r in routing
    if r["risk_level_T1"] == "medium"
]
assert len(medium) == 430

caption_lookup = {
    (r["image_id"], r["label"]): r["base_caption"]
    for r in json.loads(CAPTIONS.read_text())["records"]
}

old_result = json.loads(OLD_RESULTS.read_text())
old_lookup = {
    (r["image_id"], r["label"]): r
    for r in old_result["medium_records"]
}

records = []
pending = []
reused_old = 0
resumed_new = 0

for route in sorted(
    medium,
    key=lambda r: (r["image_id"], r["label"]),
):
    image_id = route["image_id"]
    label = route["label"]
    sample = samples[image_id]
    input_path = Path(route["source_path"])
    input_sha = sha256(input_path)

    output_path = (
        OUT / "filtered" / label / f"{image_id}.png"
    )
    state_path = (
        OUT / "state/per_record" / label /
        f"{image_id}.json"
    )

    original_caption = caption_lookup[(image_id, label)]
    old_key = (sample["original_image_id"], label)
    old = old_lookup.get(old_key)

    old_valid = (
        sample["origin"] == "paper_vitgpt2_coco1000_v1"
        and old is not None
        and old["input_sha256"] == input_sha
        and Path(old["output_path"]).is_file()
        and sha256(old["output_path"]) == old["output_sha256"]
    )

    common = {
        **route,
        "origin": sample["origin"],
        "original_image_id": sample["original_image_id"],
        "input_path": str(input_path),
        "input_sha256": input_sha,
        "filter": FILTER,
        "target_caption": sample["target_caption"],
        "original_caption": original_caption,
    }

    if old_valid:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not output_path.exists():
            output_path.symlink_to(
                Path(old["output_path"]).resolve()
            )
        assert sha256(output_path) == old["output_sha256"]

        records.append({
            **common,
            "output_path": str(output_path),
            "output_sha256": old["output_sha256"],
            "caption": old["caption"],
            "reused_from_old_experiment": True,
        })
        reused_old += 1
        continue

    if state_path.is_file():
        state = json.loads(state_path.read_text())
        valid = (
            state.get("input_sha256") == input_sha
            and Path(state.get("output_path", "")).is_file()
            and sha256(state["output_path"])
            == state.get("output_sha256")
        )
        if valid:
            records.append(state)
            resumed_new += 1
            continue

    pending.append((common, output_path, state_path))

print("Medium total:", len(medium), flush=True)
print("Reused old:", reused_old, flush=True)
print("Resumed new:", resumed_new, flush=True)
print("Need generation:", len(pending), flush=True)

if pending:
    assert torch.cuda.is_available(), "GPU unavailable"

    mfii = json.loads(
        (OLD_ROOT / "mfii/results.json").read_text()
    )
    mftt = json.loads(
        (OLD_ROOT / "mftt/results.json").read_text()
    )

    model, processor, tokenizer, caption_fn, _ = (
        helper.load_caption_model(
            "vitgpt2",
            "nlpconnect/vit-gpt2-image-captioning",
            mfii["revision"],
            mfii["caption_diagnostic_preset"],
            mftt,
            mfii["image_mean"],
            mfii["image_std"],
        )
    )

    for number, (common, output_path, state_path) in enumerate(
        pending,
        1,
    ):
        helper.filter_image(
            common["input_path"],
            output_path,
        )
        with Image.open(output_path) as image:
            filtered_caption = caption_fn(image.convert("RGB"))

        assert filtered_caption
        state = {
            **common,
            "output_path": str(output_path),
            "output_sha256": sha256(output_path),
            "caption": filtered_caption,
            "reused_from_old_experiment": False,
        }
        dump(state, state_path)
        records.append(state)

        if number % 25 == 0 or number == len(pending):
            print(
                f"STAGE2 {number}/{len(pending)}",
                flush=True,
            )

    del model, processor, tokenizer, caption_fn
    gc.collect()
    torch.cuda.empty_cache()

assert len(records) == 430

texts = []
for record in records:
    texts.extend([
        record["caption"],
        record["target_caption"],
        record["original_caption"],
    ])

lookup = helper.clip_text_embeddings(texts, "cuda")

for record in records:
    filtered = lookup[record["caption"]]
    target = lookup[record["target_caption"]]
    original = lookup[record["original_caption"]]

    target_similarity = float(filtered @ target)
    preservation = float(filtered @ original)

    record["filtered_target_similarity"] = target_similarity
    record["filtered_target_success_ge_0_80"] = (
        target_similarity >= 0.80
    )
    record["original_to_filtered_caption_similarity"] = (
        preservation
    )

records.sort(key=lambda r: (r["image_id"], r["label"]))

risk = {
    label: Counter(
        r["risk_level_T1"]
        for r in routing
        if r["label"] == label
    )
    for label in ["clean", "mf_tt"]
}

medium_attack = [
    r for r in records
    if r["label"] == "mf_tt"
]
medium_clean = [
    r for r in records
    if r["label"] == "clean"
]

remaining = sum(
    r["filtered_target_success_ge_0_80"]
    for r in medium_attack
)
low_passed = risk["mf_tt"]["low"]
high_blocked = risk["mf_tt"]["high"]
final_success = low_passed + remaining

summary = {
    "initial_successful_attacks": 1000,
    "risk_routing": {
        label: {
            band: int(risk[label][band])
            for band in ["low", "medium", "high"]
        }
        for label in ["clean", "mf_tt"]
    },
    "medium_records": len(records),
    "reused_old_records": reused_old,
    "new_or_resumed_records": len(records) - reused_old,
    "high_attacks_blocked": int(high_blocked),
    "medium_attacks_neutralized": int(
        len(medium_attack) - remaining
    ),
    "medium_attacks_remaining": int(remaining),
    "low_attacks_passed": int(low_passed),
    "final_successful_attacks": int(final_success),
    "final_asr": final_success / 1000,
    "relative_asr_reduction": 1 - final_success / 1000,
    "clean_high_blocked": int(risk["clean"]["high"]),
    "clean_medium_processed": len(medium_clean),
    "clean_medium_caption_preservation_mean": float(
        np.mean([
            r["original_to_filtered_caption_similarity"]
            for r in medium_clean
        ])
    ),
}

payload = {
    "status": "complete",
    "model": "ViT-GPT2",
    "checkpoint": "nlpconnect/vit-gpt2-image-captioning",
    "protocol": {
        "target_success_threshold": 0.80,
        "risk_thresholds": {
            "low": "score < 0.30",
            "medium": "0.30 <= score < 0.80",
            "high": "score >= 0.80",
        },
        "repair": FILTER,
    },
    "summary": summary,
    "medium_records": records,
}

result_path = OUT / "results.json"
dump(payload, result_path)

print(json.dumps(summary, indent=2), flush=True)
print("Results:", result_path, flush=True)
print("VITGPT2_EXPERIMENT2_STAGE2_COMPLETE", flush=True)
