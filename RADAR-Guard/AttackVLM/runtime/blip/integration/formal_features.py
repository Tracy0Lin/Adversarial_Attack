"""Copied formal functions without batch execution. See source_provenance.json."""

import numpy as np

FEATURE_NAMES = [
    "d_center",
    "d_iqr",
    "d_tail",
    "d_q95",
]

def normalize_rows(array):
    array = np.asarray(array, dtype=np.float64)
    norms = np.linalg.norm(
        array,
        axis=1,
        keepdims=True,
    )

    if np.any(norms <= 1e-12):
        raise RuntimeError("發現零向量 embedding。")

    return array / norms

def calculate_group_features(
    all_embeddings,
    all_names,
    selected_names,
):
    name_to_index = {
        name: index
        for index, name in enumerate(all_names)
    }

    indices = [
        name_to_index[name]
        for name in selected_names
    ]

    unit_embeddings = normalize_rows(
        all_embeddings[indices]
    )

    mean_embedding = unit_embeddings.mean(axis=0)
    center_norm = np.linalg.norm(mean_embedding)

    if center_norm <= 1e-12:
        raise RuntimeError("Caption embedding 中心為零向量。")

    unit_center = mean_embedding / center_norm

    cosine_values = unit_embeddings @ unit_center

    # 這裡是數值截斷，不是 CLIP 模型。
    safe_cosine_values = np.clip(
        cosine_values,
        -1.0,
        1.00,
    )

    angular_distances = (
        np.arccos(safe_cosine_values) /
        np.pi
    )

    if not np.all(np.isfinite(angular_distances)):
        raise RuntimeError("角距離出現 NaN 或 Inf。")

    q25, q50, q75, q90, q95 = np.quantile(
        angular_distances,
        [0.25, 0.50, 0.75, 0.90, 0.95],
        method="linear",
    )

    features = {
        "d_center": float(
            np.mean(angular_distances)
        ),
        "d_iqr": float(q75 - q25),
        "d_tail": float(q90 - q50),
        "d_q95": float(q95),
    }

    for name, value in features.items():
        if not np.isfinite(value):
            raise RuntimeError(
                f"{name} 出現非有限值。"
            )

        if value < -1e-12 or value > 1.0 + 1e-12:
            raise RuntimeError(
                f"{name} 超出 [0,1]：{value}"
            )

    return {
        "caption_names": selected_names,
        "num_captions": len(selected_names),
        "features": features,
        "normalized_angular_distances": [
            float(value)
            for value in angular_distances
        ],
        "distance_summary": {
            "min": float(np.min(angular_distances)),
            "q25": float(q25),
            "median": float(q50),
            "q75": float(q75),
            "q90": float(q90),
            "q95": float(q95),
            "max": float(np.max(angular_distances)),
        },
    }
