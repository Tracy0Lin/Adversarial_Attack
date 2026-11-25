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
BASE = (
    PROJECT
    / "formal_tests/attackvlm_three_scenarios_v1/"
      "git_large_coco"
)
STAGE1 = (
    BASE
    / "experiment2_stress/tcvad/stage1"
)
OUT = (
    BASE
    / "experiment2_stress/tcvad/stage2_medium"
)

ROUTING = (
    STAGE1
    / "manifests/git_experiment2_routing_manifest.json"
)
CAPTIONS = (
    STAGE1
    / "captions/git_large_coco_21captions_1000pairs.json"
)
OLD_ROOT = (
    PROJECT
    / "formal_tests/paper_git_large_coco1000_v1/"
      "formal1000_v1"
)
HELPER_PATH = (
    PROJECT
    / "work_code/"
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
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(block)
    return digest.hexdigest()


def dump(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


spec = importlib.util.spec_from_file_location(
    "git_sss_helper",
    HELPER_PATH,
)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)

if helper.FILTER != FILTER:
    raise RuntimeError(
        f"bilateral_strong 設定不一致："
        f"{helper.FILTER}"
    )

routing_payload = json.loads(
    ROUTING.read_text(encoding="utf-8")
)
routing = routing_payload["records"]

if len(routing) != 2000:
    raise RuntimeError(
        f"Routing 應為 2000 筆，實際為 "
        f"{len(routing)}。"
    )

medium = [
    row for row in routing
    if row["risk_level_T1"] == "medium"
]

medium_counts = Counter(
    row["label"] for row in medium
)

if medium_counts != {
    "clean": 304,
    "mf_tt": 741,
}:
    raise RuntimeError(
        f"Medium 數量不符：{dict(medium_counts)}"
    )

caption_payload = json.loads(
    CAPTIONS.read_text(encoding="utf-8")
)
caption_lookup = {
    (
        str(row["image_id"]).zfill(4),
        row["label"],
    ): row["base_caption"]
    for row in caption_payload["records"]
}

if len(caption_lookup) != 2000:
    raise RuntimeError(
        f"Caption lookup 數量不符："
        f"{len(caption_lookup)}"
    )

records = []
pending = []
resumed = 0

for route in sorted(
    medium,
    key=lambda row: (
        row["image_id"],
        row["label"],
    ),
):
    image_id = str(route["image_id"]).zfill(4)
    label = route["label"]
    input_path = Path(route["input_path"])
    input_sha = sha256(input_path)

    output_path = (
        OUT
        / "filtered"
        / label
        / f"{image_id}.png"
    )
    state_path = (
        OUT
        / "state/per_record"
        / label
        / f"{image_id}.json"
    )

    original_caption = caption_lookup[
        (image_id, label)
    ]

    common = {
        **route,
        "image_id": image_id,
        "input_path": str(input_path),
        "input_sha256": input_sha,
        "filter": FILTER,
        "original_caption": original_caption,
    }

    if state_path.is_file():
        state = json.loads(
            state_path.read_text(encoding="utf-8")
        )
        state_output = Path(
            state.get("output_path", "")
        )

        valid = (
            state.get("input_sha256") == input_sha
            and state.get("filter") == FILTER
            and state_output.is_file()
            and sha256(state_output)
            == state.get("output_sha256")
            and bool(state.get("caption"))
        )

        if valid:
            records.append(state)
            resumed += 1
            continue

    pending.append(
        (common, output_path, state_path)
    )

print("Medium total:", len(medium), flush=True)
print("Medium clean:", medium_counts["clean"], flush=True)
print("Medium MF-tt:", medium_counts["mf_tt"], flush=True)
print("Resumed:", resumed, flush=True)
print("Need generation:", len(pending), flush=True)

if pending:
    if not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable")

    mfii = json.loads(
        (OLD_ROOT / "mfii/results.json").read_text(
            encoding="utf-8"
        )
    )
    mftt = json.loads(
        (OLD_ROOT / "mftt/results.json").read_text(
            encoding="utf-8"
        )
    )

    (
        model,
        processor,
        tokenizer,
        caption_fn,
        _,
    ) = helper.load_caption_model(
        "git_large_coco",
        "microsoft/git-large-coco",
        mfii["revision"],
        mfii["caption_diagnostic_preset"],
        mftt,
        mfii["image_mean"],
        mfii["image_std"],
    )

    for number, item in enumerate(
        pending,
        1,
    ):
        common, output_path, state_path = item

        helper.filter_image(
            common["input_path"],
            output_path,
        )

        with Image.open(output_path) as image:
            filtered_caption = caption_fn(
                image.convert("RGB")
            )

        if not filtered_caption:
            raise RuntimeError(
                f"{common['image_id']} 產生空描述。"
            )

        state = {
            **common,
            "output_path": str(output_path),
            "output_sha256": sha256(output_path),
            "caption": filtered_caption,
        }

        dump(state, state_path)
        records.append(state)

        if (
            number % 25 == 0
            or number == len(pending)
        ):
            print(
                f"STAGE2 {number}/{len(pending)}",
                flush=True,
            )

    del model, processor, tokenizer, caption_fn
    gc.collect()
    torch.cuda.empty_cache()

if len(records) != 1045:
    raise RuntimeError(
        f"Stage-2 紀錄應為 1045，實際為 "
        f"{len(records)}。"
    )

if not torch.cuda.is_available():
    raise RuntimeError(
        "CLIP 文字評估需要 GPU。"
    )

texts = []
for record in records:
    texts.extend(
        [
            record["caption"],
            record["target_caption"],
            record["original_caption"],
        ]
    )

lookup = helper.clip_text_embeddings(
    texts,
    "cuda",
)

for record in records:
    filtered = lookup[record["caption"]]
    target = lookup[record["target_caption"]]
    original = lookup[record["original_caption"]]

    target_similarity = float(
        filtered @ target
    )
    preservation = float(
        filtered @ original
    )

    record[
        "filtered_target_similarity"
    ] = target_similarity
    record[
        "filtered_target_success_ge_0_80"
    ] = target_similarity >= 0.80
    record[
        "original_to_filtered_caption_similarity"
    ] = preservation

records.sort(
    key=lambda row: (
        row["image_id"],
        row["label"],
    )
)

risk = {
    label: Counter(
        row["risk_level_T1"]
        for row in routing
        if row["label"] == label
    )
    for label in ["clean", "mf_tt"]
}

medium_attack = [
    row for row in records
    if row["label"] == "mf_tt"
]
medium_clean = [
    row for row in records
    if row["label"] == "clean"
]

if len(medium_attack) != 741:
    raise RuntimeError(len(medium_attack))
if len(medium_clean) != 304:
    raise RuntimeError(len(medium_clean))

remaining = sum(
    row["filtered_target_success_ge_0_80"]
    for row in medium_attack
)
neutralized = len(medium_attack) - remaining
low_passed = int(risk["mf_tt"]["low"])
high_blocked = int(risk["mf_tt"]["high"])
final_success = low_passed + remaining

clean_preservations = [
    row[
        "original_to_filtered_caption_similarity"
    ]
    for row in medium_clean
]
clean_below_080 = sum(
    score < 0.80
    for score in clean_preservations
)

summary = {
    "initial_successful_attacks": 1000,
    "risk_routing": {
        label: {
            band: int(risk[label][band])
            for band in [
                "low",
                "medium",
                "high",
            ]
        }
        for label in ["clean", "mf_tt"]
    },
    "medium_records": len(records),
    "resumed_records": resumed,
    "generated_this_run": len(pending),
    "high_attacks_blocked": high_blocked,
    "medium_attacks_neutralized": int(
        neutralized
    ),
    "medium_attacks_remaining": int(
        remaining
    ),
    "low_attacks_passed": low_passed,
    "final_successful_attacks": int(
        final_success
    ),
    "final_asr": final_success / 1000,
    "relative_asr_reduction": (
        1 - final_success / 1000
    ),
    "clean_low_passed": int(
        risk["clean"]["low"]
    ),
    "clean_high_blocked": int(
        risk["clean"]["high"]
    ),
    "clean_medium_processed": len(
        medium_clean
    ),
    "clean_medium_caption_preservation_mean": (
        float(np.mean(clean_preservations))
    ),
    "clean_medium_preservation_below_0_80": (
        int(clean_below_080)
    ),
}

payload = {
    "status": "complete",
    "model": "GIT-large-COCO",
    "checkpoint": "microsoft/git-large-coco",
    "protocol": {
        "target_success_threshold": 0.80,
        "risk_thresholds": {
            "low": "score < 0.30",
            "medium": (
                "0.30 <= score < 0.80"
            ),
            "high": "score >= 0.80",
        },
        "repair": FILTER,
        "clean_preservation_note": (
            "CLIP ViT-B/32 cosine between "
            "the deterministic caption before "
            "and after filtering"
        ),
    },
    "summary": summary,
    "medium_records": records,
}

result_path = OUT / "results.json"
dump(payload, result_path)

print(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    )
)
print("Results:", result_path)
print("GIT_EXPERIMENT2_STAGE2_COMPLETE")
