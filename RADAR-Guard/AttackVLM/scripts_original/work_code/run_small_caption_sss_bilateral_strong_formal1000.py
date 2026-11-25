"""Apply the frozen bilateral_strong repair to Medium-risk records."""
import ast
import csv
import gc
import hashlib
import json
import math
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    GitForCausalLM,
    ViTImageProcessor,
    VisionEncoderDecoderModel,
)


PROJECT = Path("/workspace/projects/couser/AttackVLM_Official_Reproduction")
MODELS = {
    "vitgpt2": (
        "ViT-GPT2",
        "nlpconnect/vit-gpt2-image-captioning",
        "paper_vitgpt2_coco1000_v1",
    ),
    "git_large_coco": (
        "GIT-large-COCO",
        "microsoft/git-large-coco",
        "paper_git_large_coco1000_v1",
    ),
}
FILTER = {
    "name": "bilateral_strong",
    "method": "bilateral",
    "diameter": 7,
    "sigma_color": 50.0,
    "sigma_space": 7.0,
}
LOW_THRESHOLD = 0.30
HIGH_THRESHOLD = 0.80
TARGET_THRESHOLD = 0.80


def dump(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def risk_band(score):
    if score < LOW_THRESHOLD:
        return "low"
    if score < HIGH_THRESHOLD:
        return "medium"
    return "high"


def load_git_helper(path, cache_enabled):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {
        "_git_batch_inputs",
        "_generate_git_cached",
        "_generate_git_cached_impl",
    }
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in functions} == names
    scope = {
        "torch": torch,
        "_GIT_CACHE_ENABLED": bool(cache_enabled),
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), scope)
    return scope


def load_caption_model(
    slug,
    checkpoint,
    revision,
    preset,
    attack_report,
    expected_mean,
    expected_std,
):
    if slug == "vitgpt2":
        from transformers import AutoTokenizer

        model = VisionEncoderDecoderModel.from_pretrained(
            checkpoint, revision=revision
        ).eval().to("cuda")
        processor = ViTImageProcessor.from_pretrained(
            checkpoint, revision=revision
        )
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint, revision=revision
        )
        image_processor = processor
        generate = model.generate
        helper_sha = None
    else:
        model = GitForCausalLM.from_pretrained(
            checkpoint, revision=revision
        ).eval().to("cuda")
        processor = AutoProcessor.from_pretrained(
            checkpoint, revision=revision
        )
        tokenizer = processor.tokenizer
        image_processor = processor.image_processor
        helper_path = PROJECT / "work_code/run_git_large_coco_mftt_pilot5.py"
        helper = load_git_helper(
            helper_path,
            attack_report["git_vision_cache_enabled"],
        )
        generate = lambda **kwargs: helper["_generate_git_cached"](
            model, **kwargs
        )
        helper_sha = digest(helper_path)

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    assert image_processor.do_normalize
    assert image_processor.image_mean == expected_mean
    assert image_processor.image_std == expected_std

    mean = torch.tensor(
        image_processor.image_mean, device="cuda"
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        image_processor.image_std, device="cuda"
    ).view(1, 3, 1, 1)

    def caption(image):
        rgb = np.array(image.convert("RGB"), copy=True)
        assert rgb.shape == (224, 224, 3)
        pixels = (
            torch.from_numpy(rgb)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .to("cuda")
            .contiguous(memory_format=torch.channels_last)
        )
        with torch.inference_mode():
            ids = generate(
                pixel_values=(pixels / 255.0 - mean) / std,
                **preset,
            )
        assert ids.shape[0] == 1
        return tokenizer.batch_decode(ids, skip_special_tokens=True)[0].strip()

    return model, processor, tokenizer, caption, helper_sha


def filter_image(input_path, output_path):
    image = Image.open(input_path).convert("RGB")
    array = np.asarray(image, dtype=np.uint8)
    assert array.shape == (224, 224, 3)
    filtered = cv2.bilateralFilter(
        array,
        d=FILTER["diameter"],
        sigmaColor=FILTER["sigma_color"],
        sigmaSpace=FILTER["sigma_space"],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    Image.fromarray(filtered, mode="RGB").save(temporary, format="PNG")
    temporary.replace(output_path)


def clip_text_embeddings(texts, device):
    import clip

    model, _ = clip.load("ViT-B/32", device=device)
    model.eval()
    unique = sorted(set(texts))
    lookup = {}
    for start in range(0, len(unique), 256):
        batch = unique[start : start + 256]
        tokens = clip.tokenize(batch, truncate=True).to(device)
        with torch.inference_mode():
            vectors = model.encode_text(tokens)
            vectors = vectors / vectors.norm(dim=-1, keepdim=True)
        for text, vector in zip(batch, vectors.detach().cpu().float().numpy()):
            lookup[text] = vector
        print(f"CLIP text: {min(start + 256, len(unique))}/{len(unique)}", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return lookup


def main():
    slug = sys.argv[1]
    display, checkpoint, folder = MODELS[slug]
    root = PROJECT / "formal_tests" / folder / "formal1000_v1"
    mfii_path = root / "mfii/results.json"
    attack_path = root / "mftt/results.json"
    risk_path = root / "defense/svm_group5fold/oof_predictions.csv"
    output = root / "sss_bilateral_strong"
    state_dir = output / "state/per_record"
    result_path = output / "results.json"
    protocol_path = output / "frozen_protocol.json"
    helper_path = PROJECT / "work_code/run_git_large_coco_mftt_pilot5.py"
    source_config = (
        PROJECT
        / "formal_tests/paper_coco1000_v1/tcvad/experiment2_sss_blip/"
        "formal/configs/sss_formal_ssa_grid.json"
    )

    mfii = json.loads(mfii_path.read_text(encoding="utf-8"))
    attack = json.loads(attack_path.read_text(encoding="utf-8"))
    assert mfii["status"] == "complete"
    assert attack["status"] == "complete"
    assert mfii["checkpoint"] == attack["checkpoint"] == checkpoint
    assert mfii["revision"] == attack["revision"]
    assert len(attack["records"]) == 1000
    revision = attack["revision"]
    preset = attack["evaluation_caption_preset"]
    assert preset == {
        "do_sample": False,
        "num_beams": 4,
        "max_new_tokens": 32,
    }
    attack_rows = {
        Path(row["image_key"]).name: row for row in attack["records"]
    }
    assert len(attack_rows) == 1000

    risk_rows = []
    with risk_path.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            image_id = str(row["image_id"])
            label = "mf_tt" if int(row["y_true"]) == 1 else "clean"
            score = float(row["sigmoid_margin_score"])
            risk_rows.append(
                {
                    "image_id": image_id,
                    "label": label,
                    "risk_score_T1": score,
                    "risk_band": risk_band(score),
                    "fold": int(row["fold"]),
                }
            )
    assert len(risk_rows) == 2000
    assert len({(row["image_id"], row["label"]) for row in risk_rows}) == 2000
    routing = {
        label: Counter(
            row["risk_band"] for row in risk_rows if row["label"] == label
        )
        for label in ["clean", "mf_tt"]
    }

    protocol = {
        "model": display,
        "checkpoint": checkpoint,
        "revision": revision,
        "mfii_results": str(mfii_path),
        "mfii_results_sha256": digest(mfii_path),
        "attack_results": str(attack_path),
        "attack_results_sha256": digest(attack_path),
        "oof_risk_predictions": str(risk_path),
        "oof_risk_predictions_sha256": digest(risk_path),
        "source_sss_grid": str(source_config),
        "source_sss_grid_sha256": digest(source_config),
        "repair": FILTER,
        "risk_thresholds": {
            "low_upper_exclusive": LOW_THRESHOLD,
            "high_lower_inclusive": HIGH_THRESHOLD,
        },
        "target_success_threshold": TARGET_THRESHOLD,
        "caption_decoding": preset,
        "driver_sha256": digest(__file__),
        "git_generation_adapter": str(helper_path) if slug == "git_large_coco" else None,
        "git_generation_adapter_sha256": digest(helper_path) if slug == "git_large_coco" else None,
        "routing_policy": {
            "low": "pass unchanged",
            "medium": "bilateral_strong then caption",
            "high": "block",
        },
    }
    source_grid = json.loads(source_config.read_text(encoding="utf-8"))
    frozen_filter = next(
        item for item in source_grid["configs"] if item["name"] == FILTER["name"]
    )
    assert frozen_filter == {
        "name": "bilateral_strong",
        "method": "bilateral",
        "parameters": {
            "diameter": FILTER["diameter"],
            "sigma_color": FILTER["sigma_color"],
            "sigma_space": FILTER["sigma_space"],
        },
    }
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text(encoding="utf-8")) == protocol
    else:
        output.mkdir(parents=True, exist_ok=True)
        dump(protocol, protocol_path)

    medium = [row for row in risk_rows if row["risk_band"] == "medium"]
    print("Model:", display, flush=True)
    print("Routing clean:", dict(routing["clean"]), flush=True)
    print("Routing MF-tt:", dict(routing["mf_tt"]), flush=True)
    print("Medium records:", len(medium), flush=True)

    valid = {}
    for item in medium:
        record_path = state_dir / item["label"] / f"{item['image_id']}.json"
        if not record_path.is_file():
            continue
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
            input_row = attack_rows[item["image_id"]]
            input_path = Path(
                input_row["clean_reference"]
                if item["label"] == "clean"
                else input_row["mftt_image"]
            )
            assert record["input_sha256"] == digest(input_path)
            assert record["filter"] == FILTER
            assert record["caption"]
            assert Path(record["output_path"]).is_file()
            assert record["output_sha256"] == digest(record["output_path"])
            valid[(item["image_id"], item["label"])] = record
        except Exception:
            continue

    print(f"Resume valid: {len(valid)}/{len(medium)}", flush=True)
    pending = [
        row for row in medium if (row["image_id"], row["label"]) not in valid
    ]

    if pending:
        assert torch.cuda.is_available()
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        model, processor, tokenizer, caption, helper_sha = load_caption_model(
            slug,
            checkpoint,
            revision,
            preset,
            attack,
            mfii["image_mean"],
            mfii["image_std"],
        )
        started = time.time()
        for index, item in enumerate(pending, start=1):
            image_id = item["image_id"]
            label = item["label"]
            attack_row = attack_rows[image_id]
            input_path = Path(
                attack_row["clean_reference"]
                if label == "clean"
                else attack_row["mftt_image"]
            )
            assert input_path.is_file()
            output_path = output / "filtered" / label / f"{image_id}.png"
            filter_image(input_path, output_path)
            text = caption(Image.open(output_path).convert("RGB"))
            assert text
            record = {
                **item,
                "input_path": str(input_path),
                "input_sha256": digest(input_path),
                "output_path": str(output_path),
                "output_sha256": digest(output_path),
                "filter": FILTER,
                "caption": text,
                "target_caption": attack_row["target_caption"],
                "original_caption": (
                    attack_row["clean_caption"]
                    if label == "clean"
                    else attack_row["mftt_caption"]
                ),
            }
            record_path = state_dir / label / f"{image_id}.json"
            dump(record, record_path)
            valid[(image_id, label)] = record
            elapsed = time.time() - started
            print(
                f"[{index:04d}/{len(pending):04d}] {label}/{image_id} | "
                f"{elapsed / index:.2f} sec/record | {text}",
                flush=True,
            )
        del caption, tokenizer, processor, model
        gc.collect()
        torch.cuda.empty_cache()

    assert len(valid) == len(medium)
    texts = []
    for row in attack_rows.values():
        texts.extend(
            [
                row["target_caption"],
                row["clean_caption"],
                row["mftt_caption"],
            ]
        )
    texts.extend(record["caption"] for record in valid.values())
    lookup = clip_text_embeddings(texts, "cuda")

    medium_records = []
    for item in medium:
        record = valid[(item["image_id"], item["label"])]
        attack_row = attack_rows[item["image_id"]]
        target = attack_row["target_caption"]
        filtered_caption = record["caption"]
        original_caption = record["original_caption"]
        target_similarity = float(lookup[filtered_caption] @ lookup[target])
        caption_similarity = float(lookup[filtered_caption] @ lookup[original_caption])
        enriched = {
            **record,
            "filtered_target_similarity": target_similarity,
            "filtered_target_success_ge_0_80": bool(
                target_similarity >= TARGET_THRESHOLD
            ),
            "original_to_filtered_caption_similarity": caption_similarity,
        }
        medium_records.append(enriched)

    medium_attack = {
        row["image_id"]: row for row in medium_records if row["label"] == "mf_tt"
    }
    medium_clean = [row for row in medium_records if row["label"] == "clean"]
    attack_risk = {
        row["image_id"]: row for row in risk_rows if row["label"] == "mf_tt"
    }
    detailed = []
    transition = Counter()
    before_success_count = 0
    final_success_count = 0
    successful_routing = Counter()

    for image_id in sorted(attack_rows):
        row = attack_rows[image_id]
        risk = attack_risk[image_id]
        before_similarity = float(row["mftt_target_similarity"])
        before_success = before_similarity >= TARGET_THRESHOLD
        band = risk["risk_band"]
        if before_success:
            before_success_count += 1
            successful_routing[band] += 1

        if band == "high":
            action = "blocked"
            after_caption = None
            after_similarity = None
            after_success = False
        elif band == "low":
            action = "passed_unchanged"
            after_caption = row["mftt_caption"]
            after_similarity = before_similarity
            after_success = before_success
        else:
            action = "bilateral_strong"
            repaired = medium_attack[image_id]
            after_caption = repaired["caption"]
            after_similarity = float(repaired["filtered_target_similarity"])
            after_success = after_similarity >= TARGET_THRESHOLD
            if before_success and after_success:
                transition["persistent"] += 1
            elif before_success and not after_success:
                transition["neutralized"] += 1
            elif not before_success and after_success:
                transition["new_after"] += 1
            else:
                transition["not_successful"] += 1
        final_success_count += int(after_success)
        detailed.append(
            {
                "image_id": image_id,
                "risk_score_T1": risk["risk_score_T1"],
                "risk_band": band,
                "action": action,
                "target_caption": row["target_caption"],
                "before_caption": row["mftt_caption"],
                "before_target_similarity": before_similarity,
                "before_success_ge_0_80": before_success,
                "after_caption": after_caption,
                "after_target_similarity": after_similarity,
                "after_success_ge_0_80": after_success,
                "clean_target_similarity": float(row["clean_target_similarity"]),
            }
        )

    medium_before_success = sum(
        float(attack_rows[image_id]["mftt_target_similarity"]) >= TARGET_THRESHOLD
        for image_id in medium_attack
    )
    medium_after_success = sum(
        row["filtered_target_success_ge_0_80"] for row in medium_attack.values()
    )
    delta_asr = (
        (medium_before_success / max(len(medium_attack), 1))
        - (medium_after_success / max(len(medium_attack), 1))
    ) / max(medium_before_success / max(len(medium_attack), 1), 1e-12)
    clean_caption_preservation = [
        row["original_to_filtered_caption_similarity"] for row in medium_clean
    ]
    attack_caption_preservation = [
        row["original_to_filtered_caption_similarity"]
        for row in medium_records
        if row["label"] == "mf_tt"
    ]
    clean_already_target = sum(
        float(row["clean_target_similarity"]) >= TARGET_THRESHOLD
        for row in attack_rows.values()
    )
    induced_before = sum(
        float(row["clean_target_similarity"]) < TARGET_THRESHOLD
        and float(row["mftt_target_similarity"]) >= TARGET_THRESHOLD
        for row in attack_rows.values()
    )
    induced_after = sum(
        detail["clean_target_similarity"] < TARGET_THRESHOLD
        and detail["after_success_ge_0_80"]
        for detail in detailed
    )
    result = {
        "status": "complete",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": display,
        "protocol": protocol,
        "routing": {
            label: {band: int(routing[label][band]) for band in ["low", "medium", "high"]}
            for label in ["clean", "mf_tt"]
        },
        "successful_attack_routing_before": {
            band: int(successful_routing[band]) for band in ["low", "medium", "high"]
        },
        "medium": {
            "clean_records": len(medium_clean),
            "mf_tt_records": len(medium_attack),
            "mf_tt_success_before": int(medium_before_success),
            "mf_tt_success_after": int(medium_after_success),
            "asr_before": medium_before_success / max(len(medium_attack), 1),
            "asr_after": medium_after_success / max(len(medium_attack), 1),
            "delta_asr": float(delta_asr),
            "transitions": dict(transition),
            "clean_caption_preservation_mean": float(np.mean(clean_caption_preservation)),
            "mf_tt_caption_preservation_mean": float(np.mean(attack_caption_preservation)),
        },
        "full_pipeline": {
            "raw_target_success_before": int(before_success_count),
            "raw_asr_before": before_success_count / 1000.0,
            "raw_target_success_after": int(final_success_count),
            "raw_asr_after": final_success_count / 1000.0,
            "raw_delta_asr": (
                (before_success_count - final_success_count)
                / max(before_success_count, 1)
            ),
            "clean_already_target_success_before": int(clean_already_target),
            "attack_induced_success_before": int(induced_before),
            "attack_induced_success_after": int(induced_after),
            "clean_high_false_blocks": int(routing["clean"]["high"]),
            "clean_medium_processed": int(routing["clean"]["medium"]),
            "mf_tt_high_blocked": int(routing["mf_tt"]["high"]),
            "mf_tt_medium_repaired": int(routing["mf_tt"]["medium"]),
            "mf_tt_low_passed": int(routing["mf_tt"]["low"]),
        },
        "medium_records": medium_records,
        "attack_records": detailed,
    }
    dump(result, result_path)
    print("\n===== RESULT =====", flush=True)
    print("Model:", display, flush=True)
    print("Successful attacks by band:", dict(successful_routing), flush=True)
    print(
        f"Medium ASR: {result['medium']['asr_before']:.6f} -> "
        f"{result['medium']['asr_after']:.6f}",
        flush=True,
    )
    print("Medium transitions:", dict(transition), flush=True)
    print(
        f"Full ASR: {result['full_pipeline']['raw_asr_before']:.6f} -> "
        f"{result['full_pipeline']['raw_asr_after']:.6f}",
        flush=True,
    )
    print(
        f"Induced successes: {induced_before} -> {induced_after}",
        flush=True,
    )
    print("Results:", result_path, flush=True)
    print(display + "_SSS_BILATERAL_STRONG_FORMAL1000_PASS", flush=True)


if __name__ == "__main__":
    main()
