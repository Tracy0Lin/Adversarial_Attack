"""Copied formal functions without batch execution. See source_provenance.json."""

from PIL import Image
import torchvision.transforms.functional as TF

OUTPUT_SIZE = (384, 384)

ROTATION_DEGREES = [
    -5, -4, -3, -2, -1,
     1,  2,  3,  4,  5,
]

RESCALE_FACTORS = [
    0.90, 0.92, 0.94, 0.96, 0.98,
    1.02, 1.04, 1.06, 1.08, 1.10,
]

ROTATION_NAMES = [
    f"rotate_m{abs(degree)}" if degree < 0
    else f"rotate_p{degree}"
    for degree in ROTATION_DEGREES
]

RESCALE_NAMES = [
    f"rescale_{int(round(scale * 100)):03d}"
    for scale in RESCALE_FACTORS
]

TRANSFORM_NAMES = ROTATION_NAMES + RESCALE_NAMES

CAPTION_ORDER = ["base"] + TRANSFORM_NAMES

def fixed_rescale(image, scale):
    width, height = image.size

    new_width = max(10, int(width * scale))
    new_height = max(10, int(height * scale))

    resized = image.resize((new_width, new_height))
    return resized.resize(OUTPUT_SIZE)

def make_variants(image_path):
    with Image.open(image_path) as opened:
        source_size = list(opened.size)
        source_mode = opened.mode
        source_format = opened.format

        base_image = (
            opened.convert("RGB")
            .resize(OUTPUT_SIZE)
        )

    variants = [("base", base_image)]

    for degree in ROTATION_DEGREES:
        name = (
            f"rotate_m{abs(degree)}"
            if degree < 0
            else f"rotate_p{degree}"
        )

        variants.append((
            name,
            TF.rotate(base_image, degree),
        ))

    for scale in RESCALE_FACTORS:
        name = f"rescale_{int(round(scale * 100)):03d}"

        variants.append((
            name,
            fixed_rescale(base_image, scale),
        ))

    actual_order = [name for name, _ in variants]

    if actual_order != CAPTION_ORDER:
        raise RuntimeError(
            f"轉換順序錯誤：{actual_order}"
        )

    return {
        "source_size": source_size,
        "source_mode": source_mode,
        "source_format": source_format,
        "variants": variants,
    }
