import hashlib
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


SSS_ROOT = Path(
    "/workspace/projects/couser/AttackVLM_Official_Reproduction/"
    "formal_tests/paper_coco1000_v1/tcvad/experiment2_sss_blip"
)

INPUT_MANIFEST = (
    SSS_ROOT
    / "formal/manifests/"
      "sss_formal_all2000_manifest.json"
)
GRID_PATH = (
    SSS_ROOT
    / "formal/configs/"
      "sss_formal_ssa_grid.json"
)

OUTPUT_DIR = (
    SSS_ROOT
    / "formal/stage2_alone/images"
)
STATE_DIR = (
    SSS_ROOT
    / "formal/stage2_alone/state"
)
OUTPUT_MANIFEST = (
    STATE_DIR
    / "sss_stage2_alone_all2000_images_manifest.json"
)

OUTPUT_SIZE = (384, 384)
CONFIG_NAME = "bilateral_strong"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STATE_DIR.mkdir(parents=True, exist_ok=True)


def atomic_json_dump(payload, path):
    temporary = path.with_suffix(path.suffix + ".tmp")

    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(
            payload,
            stream,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    os.replace(temporary, path)


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def valid_output(path):
    if not path.exists():
        return False

    try:
        with Image.open(path) as image:
            image.load()
            return (
                image.mode == "RGB"
                and image.size == OUTPUT_SIZE
            )
    except Exception:
        return False


def load_input(path):
    with Image.open(path) as opened:
        return opened.convert("RGB").resize(
            OUTPUT_SIZE,
            Image.Resampling.BICUBIC,
        )


def apply_bilateral(image, parameters):
    array = np.asarray(
        image,
        dtype=np.uint8,
    )

    filtered = cv2.bilateralFilter(
        array,
        d=int(parameters["diameter"]),
        sigmaColor=float(
            parameters["sigma_color"]
        ),
        sigmaSpace=float(
            parameters["sigma_space"]
        ),
    )

    return Image.fromarray(filtered)


def atomic_save_png(image, path):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_name(
        path.stem + ".tmp.png"
    )

    image.save(
        temporary,
        format="PNG",
        optimize=False,
    )

    os.replace(temporary, path)


with INPUT_MANIFEST.open(
    "r",
    encoding="utf-8",
) as stream:
    input_payload = json.load(stream)

with GRID_PATH.open(
    "r",
    encoding="utf-8",
) as stream:
    grid_payload = json.load(stream)

records = input_payload["records"]

if len(records) != 2000:
    raise RuntimeError(
        f"輸入應為 2000 筆，實際為 "
        f"{len(records)}。"
    )

matching_configs = [
    config
    for config in grid_payload["configs"]
    if config["name"] == CONFIG_NAME
]

if len(matching_configs) != 1:
    raise RuntimeError(
        f"{CONFIG_NAME} 設定數量不是 1。"
    )

config = matching_configs[0]
parameters = config["parameters"]

expected_parameters = {
    "diameter": 7,
    "sigma_color": 50.0,
    "sigma_space": 7.0,
}

if parameters != expected_parameters:
    raise RuntimeError(
        f"bilateral_strong 參數不符："
        f"{parameters}"
    )

started_at = time.time()
created = 0
reused = 0
output_records = []

for index, record in enumerate(records, start=1):
    source_path = Path(record["source_path"])

    if not source_path.exists():
        raise FileNotFoundError(source_path)

    output_path = (
        OUTPUT_DIR
        / f"fold_{int(record['fold'])}"
        / str(record["image_id"])
        / str(record["label"])
        / f"{CONFIG_NAME}.png"
    )

    if valid_output(output_path):
        reused += 1
    else:
        input_image = load_input(source_path)

        output_image = apply_bilateral(
            input_image,
            parameters,
        )

        if (
            output_image.mode != "RGB"
            or output_image.size != OUTPUT_SIZE
        ):
            raise RuntimeError(
                f"{record['image_id']} 輸出格式錯誤。"
            )

        atomic_save_png(
            output_image,
            output_path,
        )

        created += 1

    output_record = dict(record)
    output_record.update({
        "config_name": CONFIG_NAME,
        "method": "bilateral",
        "parameters": parameters,
        "output_path": str(output_path),
        "sha256": sha256_file(output_path),
    })

    output_records.append(output_record)

    if (
        index % 50 == 0
        or index == len(records)
    ):
        elapsed = time.time() - started_at
        rate = index / max(elapsed, 1e-9)
        remaining = len(records) - index
        eta_minutes = (
            remaining / max(rate, 1e-9) / 60.0
        )

        print(
            f"Completed: {index}/{len(records)}"
            f" | created: {created}"
            f" | reused: {reused}"
            f" | rate: {rate:.2f} images/s"
            f" | ETA: {eta_minutes:.1f} min",
            flush=True,
        )

clean_count = sum(
    record["label"] == "clean"
    for record in output_records
)
attack_count = sum(
    record["label"] == "mf_tt"
    for record in output_records
)

if clean_count != 1000:
    raise RuntimeError(
        f"Clean 應為 1000，實際為 {clean_count}。"
    )

if attack_count != 1000:
    raise RuntimeError(
        f"MF-tt 應為 1000，實際為 {attack_count}。"
    )

if not all(
    valid_output(Path(record["output_path"]))
    for record in output_records
):
    raise RuntimeError(
        "存在未通過 RGB/384×384 驗證的影像。"
    )

output_payload = {
    "schema_version": "1.0",
    "experiment": (
        "SSS Stage-2-alone all-2000 "
        "bilateral-strong images"
    ),
    "output_size": list(OUTPUT_SIZE),
    "config": config,
    "summary": {
        "num_records": len(output_records),
        "num_clean": clean_count,
        "num_mf_tt": attack_count,
        "created_this_run": created,
        "reused_this_run": reused,
        "all_rgb": True,
        "all_384x384": True,
    },
    "records": output_records,
}

atomic_json_dump(
    output_payload,
    OUTPUT_MANIFEST,
)

print("\nRecords:", len(output_records))
print("Clean:", clean_count)
print("MF-tt:", attack_count)
print("Created:", created)
print("Reused:", reused)
print("Manifest:", OUTPUT_MANIFEST)
print("SSS_STAGE2_ALONE_IMAGES_OK")
