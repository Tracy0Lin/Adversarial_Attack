import csv
import json
import os
from collections import Counter
from pathlib import Path

import clip
import numpy as np
import torch
import torch.nn.functional as F


EXPERIMENT_ROOT = Path(
    "formal_tests/attackvlm_three_scenarios_v1/blip/experiment2_stress/tcvad/stage1_candidate434"
)

CAPTIONS_FILE = (
    EXPERIMENT_ROOT /
    "captions/blip_21captions_candidate434.json"
)

FEATURE_JSON = (
    EXPERIMENT_ROOT /
    "features/blip_clip_vitb32_angular_features_candidate434.json"
)

FEATURE_CSV = (
    EXPERIMENT_ROOT /
    "features/blip_clip_vitb32_angular_features_candidate434.csv"
)

SUMMARY_JSON = (
    EXPERIMENT_ROOT /
    "results/blip_clip_vitb32_feature_distribution_summary.json"
)

CLIP_MODEL_NAME = "ViT-B/32"
BATCH_SIZE = 256
SEED = 20260729

ROTATION_NAMES = [
    "rotate_m5",
    "rotate_m4",
    "rotate_m3",
    "rotate_m2",
    "rotate_m1",
    "rotate_p1",
    "rotate_p2",
    "rotate_p3",
    "rotate_p4",
    "rotate_p5",
]

RESIZE_NAMES = [
    "rescale_090",
    "rescale_092",
    "rescale_094",
    "rescale_096",
    "rescale_098",
    "rescale_102",
    "rescale_104",
    "rescale_106",
    "rescale_108",
    "rescale_110",
]

TRANSFORM_NAMES = ROTATION_NAMES + RESIZE_NAMES
CAPTION_ORDER = ["base"] + TRANSFORM_NAMES

GROUP_NAMES = {
    "features_all": CAPTION_ORDER,
    "features_rotation": ["base"] + ROTATION_NAMES,
    "features_resize": ["base"] + RESIZE_NAMES,
}

FEATURE_NAMES = [
    "d_center",
    "d_iqr",
    "d_tail",
    "d_q95",
]


def atomic_json_dump(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = Path(str(path) + ".tmp")

    with open(temporary_path, "w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )

    os.replace(temporary_path, path)


def atomic_csv_dump(rows, fieldnames, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = Path(str(path) + ".tmp")

    with open(
        temporary_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    os.replace(temporary_path, path)


def get_record_captions(record):
    if record.get("caption_order") != CAPTION_ORDER:
        raise RuntimeError(
            f"{record.get('image_id')} "
            f"{record.get('label')} caption_order 錯誤。"
        )

    transform_captions = record.get(
        "transform_captions",
        {},
    )

    if list(transform_captions) != TRANSFORM_NAMES:
        raise RuntimeError(
            f"{record.get('image_id')} "
            f"{record.get('label')} 轉換名稱或順序錯誤。"
        )

    captions = [
        str(record.get("base_caption", "")).strip()
    ]

    captions.extend(
        str(transform_captions[name]).strip()
        for name in TRANSFORM_NAMES
    )

    if len(captions) != 21:
        raise RuntimeError(
            f"{record.get('image_id')} "
            f"{record.get('label')} captions 不是 21 個。"
        )

    if any(not caption for caption in captions):
        raise RuntimeError(
            f"{record.get('image_id')} "
            f"{record.get('label')} 存在空 caption。"
        )

    return captions


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


def describe(values):
    values = np.asarray(values, dtype=np.float64)

    q25, median, q75 = np.quantile(
        values,
        [0.25, 0.50, 0.75],
        method="linear",
    )

    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "std_sample": float(np.std(values, ddof=1)),
        "min": float(np.min(values)),
        "q25": float(q25),
        "median": float(median),
        "q75": float(q75),
        "max": float(np.max(values)),
    }


torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


print("===== 讀取 Caption JSON =====", flush=True)
print("輸入：", CAPTIONS_FILE, flush=True)

with open(CAPTIONS_FILE, "r", encoding="utf-8") as file:
    caption_payload = json.load(file)

input_summary = caption_payload.get("summary", {})

expected_summary_values = {
    "num_image_ids": 434,
    "num_records": 868,
    "num_captions_per_record": 21,
    "total_captions": 18228,
    "empty_caption_count": 0,
}

for name, expected_value in expected_summary_values.items():
    actual_value = input_summary.get(name)

    if actual_value != expected_value:
        raise RuntimeError(
            f"Caption JSON 的 {name} 應為 "
            f"{expected_value}，目前為 {actual_value}。"
        )

if caption_payload.get("caption_order") != CAPTION_ORDER:
    raise RuntimeError("合併 JSON 的 caption_order 錯誤。")

records = caption_payload.get("records", [])

if len(records) != 868:
    raise RuntimeError(
        f"影像紀錄應為 868，目前為 {len(records)}。"
    )


record_caption_lists = []
all_captions = []
pair_labels = {}

for record in records:
    image_id = str(record.get("image_id", ""))
    record_label = str(record.get("label", ""))

    if record_label not in {"clean", "mf_tt"}:
        raise RuntimeError(
            f"{image_id} 出現未知類別：{record_label}"
        )

    captions = get_record_captions(record)

    record_caption_lists.append(captions)
    all_captions.extend(captions)

    pair_labels.setdefault(image_id, []).append(
        record_label
    )


if len(pair_labels) != 434:
    raise RuntimeError(
        f"Image ID 應為 434，目前為 {len(pair_labels)}。"
    )

for image_id, labels in pair_labels.items():
    if sorted(labels) != ["clean", "mf_tt"]:
        raise RuntimeError(
            f"{image_id} 不是一組 Clean／MF-tt 配對："
            f"{labels}"
        )


label_counts = Counter(
    str(record["label"])
    for record in records
)

if label_counts != Counter({
    "clean": 434,
    "mf_tt": 434,
}):
    raise RuntimeError(
        f"類別數量錯誤：{dict(label_counts)}"
    )


unique_captions = list(dict.fromkeys(all_captions))

print("影像紀錄：", len(records), flush=True)
print("Captions 總數：", len(all_captions), flush=True)
print("不重複 captions：", len(unique_captions), flush=True)
print("Clean：", label_counts["clean"], flush=True)
print("MF-tt：", label_counts["mf_tt"], flush=True)


device = "cuda" if torch.cuda.is_available() else "cpu"

print("\n===== 載入 CLIP 文字編碼器 =====", flush=True)
print("CLIP 模型：", CLIP_MODEL_NAME, flush=True)
print("Device：", device, flush=True)

if device == "cuda":
    print(
        "GPU：",
        torch.cuda.get_device_name(0),
        flush=True,
    )

clip_model, _ = clip.load(
    CLIP_MODEL_NAME,
    device=device,
    jit=False,
)

clip_model.eval()
caption_to_embedding = {}


print("\n===== 編碼 Captions =====", flush=True)

with torch.inference_mode():
    for start in range(
        0,
        len(unique_captions),
        BATCH_SIZE,
    ):
        batch_captions = unique_captions[
            start:start + BATCH_SIZE
        ]

        tokens = clip.tokenize(
            batch_captions,
            truncate=True,
        ).to(device)

        batch_embeddings = clip_model.encode_text(
            tokens
        ).float()

        batch_embeddings = F.normalize(
            batch_embeddings,
            p=2,
            dim=-1,
        )

        batch_embeddings = (
            batch_embeddings
            .cpu()
            .numpy()
            .astype(np.float64)
        )

        # 再以 float64 正規化一次，降低 arccos 數值誤差。
        batch_embeddings = normalize_rows(
            batch_embeddings
        )

        for caption, embedding in zip(
            batch_captions,
            batch_embeddings,
        ):
            caption_to_embedding[caption] = embedding

        completed = min(
            start + len(batch_captions),
            len(unique_captions),
        )

        print(
            f"CLIP 進度："
            f"{completed}/{len(unique_captions)}",
            flush=True,
        )


if len(caption_to_embedding) != len(unique_captions):
    raise RuntimeError("Caption embedding 數量不一致。")


print("\n===== 計算四項角距離特徵 =====", flush=True)

output_records = []
csv_rows = []

for record, captions in zip(
    records,
    record_caption_lists,
):
    image_id = str(record["image_id"])
    record_label = str(record["label"])
    numeric_label = 0 if record_label == "clean" else 1

    embeddings = np.stack([
        caption_to_embedding[caption]
        for caption in captions
    ])

    group_results = {}

    for group_name, selected_names in GROUP_NAMES.items():
        group_results[group_name] = (
            calculate_group_features(
                all_embeddings=embeddings,
                all_names=CAPTION_ORDER,
                selected_names=selected_names,
            )
        )

    output_record = {
        "image_id": image_id,
        "caption_model": "BLIP base_coco",
        "attack_type": (
            "clean"
            if record_label == "clean"
            else "AttackVLM_MF-tt"
        ),
        "record_label": record_label,
        "attack_target_model": (
            None
            if record_label == "clean"
            else "BLIP"
        ),
        "label": numeric_label,
        "source_path": record.get("source_path"),
        **group_results,
    }

    output_records.append(output_record)

    csv_row = {
        "image_id": image_id,
        "caption_model": "BLIP base_coco",
        "attack_type": output_record["attack_type"],
        "attack_target_model": (
            ""
            if output_record["attack_target_model"] is None
            else output_record["attack_target_model"]
        ),
        "label": numeric_label,
        "source_path": record.get("source_path", ""),
    }

    for feature_name in FEATURE_NAMES:
        csv_row[feature_name] = (
            group_results["features_all"]
            ["features"][feature_name]
        )

        csv_row[
            f"rotation_{feature_name}"
        ] = (
            group_results["features_rotation"]
            ["features"][feature_name]
        )

        csv_row[
            f"resize_{feature_name}"
        ] = (
            group_results["features_resize"]
            ["features"][feature_name]
        )

    csv_rows.append(csv_row)


if len(output_records) != 868:
    raise RuntimeError("輸出特徵紀錄不是 868 筆。")


feature_payload = {
    "schema_version": "1.0",
    "experiment": (
        "TC-VAD Experiment 1 BLIP "
        "normalized angular features"
    ),
    "caption_source": str(CAPTIONS_FILE),
    "caption_model": {
        "name": "blip_caption",
        "model_type": "base_coco",
    },
    "text_encoder": {
        "name": "OpenAI CLIP",
        "model": CLIP_MODEL_NAME,
        "embedding_normalization": "L2",
    },
    "distance": {
        "name": "normalized_angular_distance",
        "formula": (
            "arccos(clip(cosine,-1,1))/pi"
        ),
        "range": [0.0, 1.0],
        "dtype_for_statistics": "float64",
    },
    "quantile_method": "linear",
    "feature_order": FEATURE_NAMES,
    "svm_input_group": "features_all",
    "group_definitions": {
        "features_all": (
            "base + 10 rotation + 10 resize"
        ),
        "features_rotation": (
            "base + 10 rotation; "
            "group-specific normalized center"
        ),
        "features_resize": (
            "base + 10 resize; "
            "group-specific normalized center"
        ),
    },
    "summary": {
        "num_image_ids": 434,
        "num_records": 868,
        "num_clean": 434,
        "num_mf_tt": 434,
        "captions_encoded": len(all_captions),
        "unique_captions_encoded": len(unique_captions),
    },
    "records": output_records,
}

atomic_json_dump(
    feature_payload,
    FEATURE_JSON,
)


csv_fieldnames = [
    "image_id",
    "caption_model",
    "attack_type",
    "attack_target_model",
    "label",
    "d_center",
    "d_iqr",
    "d_tail",
    "d_q95",
    "rotation_d_center",
    "rotation_d_iqr",
    "rotation_d_tail",
    "rotation_d_q95",
    "resize_d_center",
    "resize_d_iqr",
    "resize_d_tail",
    "resize_d_q95",
    "source_path",
]

atomic_csv_dump(
    csv_rows,
    csv_fieldnames,
    FEATURE_CSV,
)


print("\n===== 四項特徵分布：features_all =====")

distribution_summary = {}
pair_lookup = {
    (
        record["image_id"],
        record["record_label"],
    ): record
    for record in output_records
}

paired_ids = sorted(pair_labels)

for feature_name in FEATURE_NAMES:
    clean_values = np.array([
        record["features_all"]["features"][feature_name]
        for record in output_records
        if record["record_label"] == "clean"
    ], dtype=np.float64)

    mf_tt_values = np.array([
        record["features_all"]["features"][feature_name]
        for record in output_records
        if record["record_label"] == "mf_tt"
    ], dtype=np.float64)

    paired_differences = np.array([
        pair_lookup[
            (image_id, "mf_tt")
        ]["features_all"]["features"][feature_name]
        -
        pair_lookup[
            (image_id, "clean")
        ]["features_all"]["features"][feature_name]
        for image_id in paired_ids
    ], dtype=np.float64)

    clean_stats = describe(clean_values)
    mf_tt_stats = describe(mf_tt_values)
    difference_stats = describe(paired_differences)

    tolerance = 1e-12

    num_mf_tt_greater = int(np.sum(
        paired_differences > tolerance
    ))

    num_clean_greater = int(np.sum(
        paired_differences < -tolerance
    ))

    num_equal = int(
        len(paired_differences)
        - num_mf_tt_greater
        - num_clean_greater
    )

    distribution_summary[feature_name] = {
        "clean": clean_stats,
        "mf_tt": mf_tt_stats,
        "paired_mf_tt_minus_clean": {
            **difference_stats,
            "mf_tt_greater_count": num_mf_tt_greater,
            "clean_greater_count": num_clean_greater,
            "equal_count": num_equal,
        },
    }

    print(
        f"{feature_name:<10}｜"
        f"Clean mean={clean_stats['mean']:.6f}, "
        f"median={clean_stats['median']:.6f}｜"
        f"MF-tt mean={mf_tt_stats['mean']:.6f}, "
        f"median={mf_tt_stats['median']:.6f}｜"
        f"MF-tt > Clean："
        f"{num_mf_tt_greater}/434"
    )


summary_payload = {
    "experiment": (
        "TC-VAD Experiment 1 BLIP "
        "feature distribution summary"
    ),
    "feature_source": str(FEATURE_JSON),
    "feature_group": "features_all",
    "feature_order": FEATURE_NAMES,
    "distribution": distribution_summary,
}

atomic_json_dump(
    summary_payload,
    SUMMARY_JSON,
)


sample_id = "0001"

if (
    (sample_id, "clean") in pair_lookup
    and (sample_id, "mf_tt") in pair_lookup
):
    print(f"\n===== 樣本 {sample_id} 的 features_all =====")

    for record_label in ["clean", "mf_tt"]:
        features = pair_lookup[
            (sample_id, record_label)
        ]["features_all"]["features"]

        print(
            f"{record_label}: "
            + ", ".join(
                f"{name}={features[name]:.6f}"
                for name in FEATURE_NAMES
            )
        )


print("\n===== 輸出檔案 =====")
print("完整特徵 JSON：", FEATURE_JSON)
print("SVM 輸入 CSV：", FEATURE_CSV)
print("分布摘要 JSON：", SUMMARY_JSON)

print("\n===== 完整性驗證 =====")
print("Image IDs：", len(pair_labels))
print("特徵紀錄：", len(output_records))
print("Clean：", label_counts["clean"])
print("MF-tt：", label_counts["mf_tt"])
print("每筆 features_all 維度：", len(FEATURE_NAMES))
print("NaN／Inf：0")
print("\nANGULAR_FEATURES_OK")
