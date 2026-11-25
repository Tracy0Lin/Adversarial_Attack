"""Copied formal functions without batch execution. See source_provenance.json."""

import cv2
import numpy as np
from PIL import Image

OUTPUT_SIZE = (384, 384)

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
