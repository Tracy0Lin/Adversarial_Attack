import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from lavis.models import load_model_and_preprocess


SSS_ROOT = Path(
    "/workspace/projects/couser/AttackVLM_Official_Reproduction/"
    "formal_tests/paper_coco1000_v1/tcvad/experiment2_sss_blip"
)

INPUT_MANIFEST = (
    SSS_ROOT
    / "formal/stage2_alone/state/"
      "sss_stage2_alone_all2000_images_manifest.json"
)

CAPTION_DIR = (
    SSS_ROOT / "formal/stage2_alone/captions"
)
STATE_DIR = (
    SSS_ROOT
    / "formal/stage2_alone/state/caption_records"
)
COMBINED_FILE = (
    CAPTION_DIR
    / "sss_stage2_alone_all2000_21captions.json"
)

OUTPUT_SIZE = (384, 384)
ROTATION_DEGREES = [
    -5, -4, -3, -2, -1,
    1, 2, 3, 4, 5,
]
RESCALE_FACTORS = [
    0.90, 0.92, 0.94, 0.96, 0.98,
    1.02, 1.04, 1.06, 1.08, 1.10,
]
CAPTION_ORDER = (
    ["base"]
    + [
        f"rotate_{'m' if degree < 0 else 'p'}"
        f"{abs(degree)}"
        for degree in ROTATION_DEGREES
    ]
    + [
        "rescale_" + f"{factor:.2f}".replace(
            ".", "p"
        )
        for factor in RESCALE_FACTORS
    ]
)

BATCH_SIZE = 21
SEED = 2026

CAPTION_DIR.mkdir(parents=True, exist_ok=True)
STATE_DIR.mkdir(parents=True, exist_ok=True)


def atomic_json_dump(payload, path):
    temporary = path.with_suffix(path.suffix + ".tmp")

    with temporary.open(
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(
            payload,
            stream,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    os.replace(temporary, path)


def state_path_for(record):
    return (
        STATE_DIR
        / (
            f"fold_{int(record['fold'])}_"
            f"{record['image_id']}_"
            f"{record['label']}_"
            f"{record['config_name']}.json"
        )
    )


def valid_saved_record(path, expected):
    if not path.exists():
        return None

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as stream:
            payload = json.load(stream)

        if (
            str(payload.get("image_id"))
            != str(expected["image_id"])
        ):
            return None

        if (
            str(payload.get("label"))
            != str(expected["label"])
        ):
            return None

        if (
            str(payload.get("config_name"))
            != str(expected["config_name"])
        ):
            return None

        if (
            str(payload.get("sha256"))
            != str(expected["sha256"])
        ):
            return None

        captions = payload.get("captions")

        if not isinstance(captions, list):
            return None

        if len(captions) != 21:
            return None

        if any(
            not str(caption).strip()
            for caption in captions
        ):
            return None

        if payload.get("caption_order") != CAPTION_ORDER:
            return None

        return payload

    except Exception:
        return None


def load_image(path):
    with Image.open(path) as opened:
        return opened.convert("RGB").resize(
            OUTPUT_SIZE,
            Image.Resampling.BICUBIC,
        )


def rotate_image(image, degree):
    return image.rotate(
        degree,
        resample=Image.Resampling.BICUBIC,
        expand=False,
    )


def rescale_image(image, factor):
    width, height = image.size

    scaled_width = max(
        1,
        int(round(width * factor)),
    )
    scaled_height = max(
        1,
        int(round(height * factor)),
    )

    resized = image.resize(
        (scaled_width, scaled_height),
        Image.Resampling.BICUBIC,
    )

    return resized.resize(
        OUTPUT_SIZE,
        Image.Resampling.BICUBIC,
    )


def make_variants(image):
    variants = [("base", image)]

    for degree in ROTATION_DEGREES:
        name = (
            f"rotate_"
            f"{'m' if degree < 0 else 'p'}"
            f"{abs(degree)}"
        )
        variants.append(
            (name, rotate_image(image, degree))
        )

    for factor in RESCALE_FACTORS:
        name = (
            "rescale_"
            + f"{factor:.2f}".replace(".", "p")
        )
        variants.append(
            (name, rescale_image(image, factor))
        )

    if [
        name for name, _ in variants
    ] != CAPTION_ORDER:
        raise RuntimeError(
            "轉換順序與 CAPTION_ORDER 不一致。"
        )

    return variants


def generate_retry(
    model,
    vis_processor,
    image,
    device,
    max_attempts=3,
):
    for _ in range(max_attempts):
        tensor = (
            vis_processor(image)
            .unsqueeze(0)
            .to(device)
        )

        generated = model.generate({
            "image": tensor
        })

        if generated:
            caption = str(
                generated[0]
            ).strip()

            if caption:
                return caption

    return ""


with INPUT_MANIFEST.open(
    "r",
    encoding="utf-8",
) as stream:
    manifest = json.load(stream)

records = manifest["records"]

if len(records) != 2000:
    raise RuntimeError(
        f"應有 2000 筆，實際為 "
        f"{len(records)}。"
    )

if sum(
    record["label"] == "clean"
    for record in records
) != 1000:
    raise RuntimeError("Clean 數量不是 1000。")

if sum(
    record["label"] == "mf_tt"
    for record in records
) != 1000:
    raise RuntimeError("MF-tt 數量不是 1000。")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

pending_records = []
saved_lookup = {}

for record in records:
    state_path = state_path_for(record)
    saved = valid_saved_record(
        state_path,
        record,
    )

    key = (
        int(record["fold"]),
        str(record["image_id"]),
        str(record["label"]),
    )

    if saved is None:
        pending_records.append(record)
    else:
        saved_lookup[key] = saved

print("Device:", device, flush=True)

if device.type == "cuda":
    print(
        "GPU:",
        torch.cuda.get_device_name(0),
        flush=True,
    )

print("Total records:", len(records), flush=True)
print(
    "Already valid:",
    len(saved_lookup),
    flush=True,
)
print(
    "Pending:",
    len(pending_records),
    flush=True,
)

if pending_records:
    model, vis_processors, _ = (
        load_model_and_preprocess(
            name="blip_caption",
            model_type="base_coco",
            is_eval=True,
            device=device,
        )
    )

    model.eval()
    vis_processor = vis_processors["eval"]
    started_at = time.time()

    with torch.inference_mode():
        for pending_index, record in enumerate(
            pending_records,
            start=1,
        ):
            image_path = Path(record["output_path"])

            if not image_path.exists():
                raise FileNotFoundError(image_path)

            image = load_image(image_path)
            variants = make_variants(image)

            captions_by_name = {}

            for start in range(
                0,
                len(variants),
                BATCH_SIZE,
            ):
                batch_variants = variants[
                    start:start + BATCH_SIZE
                ]

                batch_tensor = torch.stack([
                    vis_processor(variant_image)
                    for _, variant_image
                    in batch_variants
                ]).to(device)

                generated = model.generate({
                    "image": batch_tensor
                })

                if len(generated) != len(
                    batch_variants
                ):
                    raise RuntimeError(
                        f"{record['image_id']}："
                        "BLIP 輸出數量不一致。"
                    )

                for (
                    transform_name,
                    variant_image,
                ), generated_caption in zip(
                    batch_variants,
                    generated,
                ):
                    caption = str(
                        generated_caption
                    ).strip()

                    if not caption:
                        caption = generate_retry(
                            model=model,
                            vis_processor=vis_processor,
                            image=variant_image,
                            device=device,
                        )

                    if not caption:
                        raise RuntimeError(
                            f"{record['image_id']} "
                            f"{record['label']} "
                            f"{transform_name}："
                            "Caption 仍為空。"
                        )

                    captions_by_name[
                        transform_name
                    ] = caption

            captions = [
                captions_by_name[name]
                for name in CAPTION_ORDER
            ]

            output_record = dict(record)
            output_record.update({
                "caption_order": CAPTION_ORDER,
                "captions": captions,
            })

            state_path = state_path_for(record)

            atomic_json_dump(
                output_record,
                state_path,
            )

            key = (
                int(record["fold"]),
                str(record["image_id"]),
                str(record["label"]),
            )
            saved_lookup[key] = output_record

            if (
                pending_index % 10 == 0
                or pending_index
                == len(pending_records)
            ):
                elapsed = time.time() - started_at
                rate = (
                    pending_index
                    / max(elapsed, 1e-9)
                )
                remaining = (
                    len(pending_records)
                    - pending_index
                )
                eta_minutes = (
                    remaining
                    / max(rate, 1e-9)
                    / 60.0
                )

                print(
                    f"Completed pending: "
                    f"{pending_index}/"
                    f"{len(pending_records)}"
                    f" | total valid: "
                    f"{len(saved_lookup)}/2000"
                    f" | rate: {rate:.3f} "
                    f"records/s"
                    f" | ETA: "
                    f"{eta_minutes:.1f} min",
                    flush=True,
                )

combined_records = []

for record in records:
    key = (
        int(record["fold"]),
        str(record["image_id"]),
        str(record["label"]),
    )

    saved = saved_lookup.get(key)

    if saved is None:
        saved = valid_saved_record(
            state_path_for(record),
            record,
        )

    if saved is None:
        raise RuntimeError(
            f"{key} 缺少有效 Caption 結果。"
        )

    combined_records.append(saved)

total_captions = sum(
    len(record["captions"])
    for record in combined_records
)
empty_captions = sum(
    not str(caption).strip()
    for record in combined_records
    for caption in record["captions"]
)

expected_summary = {
    "num_records": 2000,
    "num_clean": 1000,
    "num_mf_tt": 1000,
    "captions_per_record": 21,
    "total_captions": 42000,
    "empty_caption_count": 0,
}

actual_summary = {
    "num_records": len(combined_records),
    "num_clean": sum(
        record["label"] == "clean"
        for record in combined_records
    ),
    "num_mf_tt": sum(
        record["label"] == "mf_tt"
        for record in combined_records
    ),
    "captions_per_record": 21,
    "total_captions": total_captions,
    "empty_caption_count": empty_captions,
}

if actual_summary != expected_summary:
    raise RuntimeError(
        f"完整性檢查失敗：{actual_summary}"
    )

combined_payload = {
    "schema_version": "1.0",
    "experiment": (
        "SSS Stage-2-alone all-2000 "
        "bilateral-strong 21 captions"
    ),
    "caption_model": {
        "name": "blip_caption",
        "model_type": "base_coco",
        "decoding": "deterministic default generation",
    },
    "output_size": list(OUTPUT_SIZE),
    "rotation_degrees": ROTATION_DEGREES,
    "rescale_factors": RESCALE_FACTORS,
    "caption_order": CAPTION_ORDER,
    "summary": actual_summary,
    "records": combined_records,
}

atomic_json_dump(
    combined_payload,
    COMBINED_FILE,
)

print("\nRecords:", actual_summary["num_records"])
print("Clean:", actual_summary["num_clean"])
print("MF-tt:", actual_summary["num_mf_tt"])
print(
    "Captions per record:",
    actual_summary["captions_per_record"],
)
print(
    "Total captions:",
    actual_summary["total_captions"],
)
print(
    "Empty captions:",
    actual_summary["empty_caption_count"],
)
print("Combined output:", COMBINED_FILE)
print("SSS_STAGE2_ALONE_CAPTIONS_OK")
