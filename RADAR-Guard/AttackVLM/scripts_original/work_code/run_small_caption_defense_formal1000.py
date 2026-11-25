"""Generate formal 21-caption data and CLIP angular features for one model."""
import ast
import gc
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import transformers
from PIL import Image
from transformers import (
    AutoProcessor,
    GitForCausalLM,
    ViTImageProcessor,
    VisionEncoderDecoderModel,
)


PROJECT = Path("/workspace/projects/couser/AttackVLM_Official_Reproduction")
BLIP_SCRIPTS = PROJECT / "formal_tests/paper_coco1000_v1/tcvad/experiment1_blip/scripts"
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


def dump(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def adapt(path, overrides, slug, display, checkpoint, audit_path):
    source = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for name in overrides:
        matches = [
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
        ]
        assert len(matches) == 1, "Unexpected source assignment: " + name
    tree.body = [
        node
        for node in tree.body
        if not (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id in overrides
                for target in node.targets
            )
        )
        and not (isinstance(node, ast.ImportFrom) and node.module == "lavis.models")
    ]

    class FormalAdapter(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str):
                node.value = node.value.replace("BLIP base_coco", display)
                node.value = node.value.replace("blip_caption", slug)
                node.value = node.value.replace("base_coco", checkpoint)
                node.value = node.value.replace("BLIP", display).replace("blip", slug)
            return node

    tree = FormalAdapter().visit(tree)
    ast.fix_missing_locations(tree)
    code = ast.unparse(tree) + "\n"
    compile(code, str(audit_path), "exec")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(code, encoding="utf-8")
    return code


def verify_complete(report, out):
    assert report["status"] == "complete"
    for field in ["captions_file", "feature_json", "feature_csv", "feature_summary"]:
        path = Path(report[field])
        assert path.is_file(), "Missing completed output: " + str(path)
        assert digest(path) == report[field + "_sha256"], "Completed output changed: " + str(path)
    assert report["total_captions"] == 42000
    assert report["feature_shape"] == [2000, 4]
    assert report["base_caption_mismatch_count"] == len(report["base_caption_mismatches"])
    assert report["base_caption_match_count"] + report["base_caption_mismatch_count"] == 2000
    print(report["model"] + "_21CAPTIONS_FEATURES_FORMAL1000_PASS", flush=True)


def main():
    slug = sys.argv[1]
    display, checkpoint, folder = MODELS[slug]
    root = PROJECT / "formal_tests" / folder
    ii_dir = root / "formal1000_v1/mfii"
    tt_dir = root / "formal1000_v1/mftt"
    out = root / "formal1000_v1/defense"
    ii_path = ii_dir / "results.json"
    tt_path = tt_dir / "results.json"
    ii = json.loads(ii_path.read_text(encoding="utf-8"))
    tt = json.loads(tt_path.read_text(encoding="utf-8"))
    assert ii["status"] == tt["status"] == "complete"
    assert ii["checkpoint"] == tt["checkpoint"] == checkpoint
    assert ii.get("revision") == tt.get("revision") and ii.get("revision")
    assert len(ii["records"]) == len(tt["records"]) == 1000
    ii_rows = sorted(ii["records"], key=lambda row: row["image_key"])
    tt_rows = sorted(tt["records"], key=lambda row: row["image_key"])
    ids = [Path(row["image_key"]).name for row in ii_rows]
    assert len(set(ids)) == 1000
    assert ids == [Path(row["image_key"]).name for row in tt_rows]
    for first, second in zip(ii_rows, tt_rows):
        assert first["image_key"] == second["image_key"]
        assert digest(first["clean_reference"]) == first["clean_reference_sha256"]
        assert digest(first["mfii_image"]) == first["mfii_image_sha256"]
        assert digest(second["mftt_image"]) == second["mftt_image_sha256"]
        assert Path(first["clean_reference"]).resolve() == Path(second["clean_reference"]).resolve()
        assert Path(first["mfii_image"]).resolve() == Path(second["mfii_image"]).resolve()
    preset = tt["evaluation_caption_preset"]
    assert preset == ii["caption_diagnostic_preset"] == {
        "do_sample": False,
        "num_beams": 4,
        "max_new_tokens": 32,
    }
    caption_source = BLIP_SCRIPTS / "generate_blip_21captions_all.py"
    feature_source = BLIP_SCRIPTS / "extract_clip_angular_features.py"
    helper_path = PROJECT / "work_code/run_git_large_coco_mftt_pilot5.py"
    protocol = {
        "model": display,
        "checkpoint": checkpoint,
        "revision": ii["revision"],
        "mfii_results": str(ii_path),
        "mfii_results_sha256": digest(ii_path),
        "mftt_results": str(tt_path),
        "mftt_results_sha256": digest(tt_path),
        "caption_source": str(caption_source),
        "caption_source_sha256": digest(caption_source),
        "feature_source": str(feature_source),
        "feature_source_sha256": digest(feature_source),
        "driver_sha256": digest(__file__),
        "decoding": preset,
        "seed": 20260729,
        "output_size": [224, 224],
        "num_image_ids": 1000,
        "labels": ["clean", "mf_tt"],
        "captions_per_image": 21,
        "base_caption_recheck_policy": 'record exact-text differences; current-run captions remain authoritative',
        "feature_order": ["d_center", "d_iqr", "d_tail", "d_q95"],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    if slug == "git_large_coco":
        protocol["git_generation_adapter"] = str(helper_path)
        protocol["git_generation_adapter_sha256"] = digest(helper_path)
        protocol["git_vision_cache_enabled"] = bool(tt["git_vision_cache_enabled"])
    config = out / "frozen_protocol.json"
    if out.exists():
        assert config.is_file(), "Existing defense output has no frozen protocol"
        assert json.loads(config.read_text(encoding="utf-8")) == protocol, "Frozen defense protocol changed"
    else:
        out.mkdir(parents=True)
        dump(protocol, config)
    report_path = out / "results.json"
    if report_path.exists():
        old_report = json.loads(report_path.read_text(encoding="utf-8"))
        if old_report.get("status") == "complete":
            verify_complete(old_report, out)
            return
    captions_file = out / "captions/21captions_1000pairs.json"
    features_file = out / "features/angular_features_1000pairs.json"
    csv_file = out / "features/angular_features_1000pairs.csv"
    summary_file = out / "results/feature_distribution_summary.json"
    report = {
        "status": "running",
        "model": display,
        "checkpoint": checkpoint,
        "revision": ii["revision"],
        "mfii_formal": str(ii_dir),
        "mftt_formal": str(tt_dir),
        "paired_image_ids": ids,
        "output_size": [224, 224],
        "decoding": preset,
        "seed": 20260729,
        "scope": "Formal COCO1000: 21 captions and angular features for clean and MF-tt.",
        "original_caption_script": str(caption_source),
        "original_caption_sha256": digest(caption_source),
        "original_feature_script": str(feature_source),
        "original_feature_sha256": digest(feature_source),
        "base_caption_mismatches": [],
    }
    dump(report, report_path)
    print("Model:", display, flush=True)
    print("Output:", out, flush=True)
    print("Input validation: 1000 clean/MF-ii/MF-tt records and PNG hashes PASS", flush=True)
    try:
        assert torch.cuda.is_available(), "GPU unavailable"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        revision = ii["revision"]
        if slug == "vitgpt2":
            from transformers import AutoTokenizer

            victim = VisionEncoderDecoderModel.from_pretrained(
                checkpoint, revision=revision
            ).eval().to("cuda")
            processor = ViTImageProcessor.from_pretrained(checkpoint, revision=revision)
            tokenizer = AutoTokenizer.from_pretrained(checkpoint, revision=revision)
            image_processor = processor
            generate = victim.generate
        else:
            victim = GitForCausalLM.from_pretrained(
                checkpoint, revision=revision
            ).eval().to("cuda")
            processor = AutoProcessor.from_pretrained(checkpoint, revision=revision)
            tokenizer = processor.tokenizer
            image_processor = processor.image_processor
            helper_tree = ast.parse(helper_path.read_text(encoding="utf-8"))
            names = {"_git_batch_inputs", "_generate_git_cached", "_generate_git_cached_impl"}
            functions = [
                node
                for node in helper_tree.body
                if isinstance(node, ast.FunctionDef) and node.name in names
            ]
            assert {node.name for node in functions} == names, "Need fixed GIT batch adapter"
            helper_scope = {
                "torch": torch,
                "_GIT_CACHE_ENABLED": bool(tt["git_vision_cache_enabled"]),
            }
            exec(
                compile(ast.Module(body=functions, type_ignores=[]), str(helper_path), "exec"),
                helper_scope,
            )
            generate = lambda **kwargs: helper_scope["_generate_git_cached"](
                victim, **kwargs
            )
            report["git_generation_adapter_sha256"] = digest(helper_path)
            report["git_vision_cache_enabled"] = helper_scope["_GIT_CACHE_ENABLED"]
        for parameter in victim.parameters():
            parameter.requires_grad_(False)
        assert image_processor.do_normalize
        assert image_processor.image_mean == ii["image_mean"]
        assert image_processor.image_std == ii["image_std"]
        mean = torch.tensor(image_processor.image_mean, device="cuda").view(1, 3, 1, 1)
        std = torch.tensor(image_processor.image_std, device="cuda").view(1, 3, 1, 1)
        session_queries = 0

        def raw_pixels(image):
            rgb = np.array(image.convert("RGB"), copy=True)
            assert rgb.shape == (224, 224, 3)
            return torch.from_numpy(rgb).permute(2, 0, 1).float()

        class CaptionAdapter:
            def eval(self):
                return self

            def generate(self, samples):
                nonlocal session_queries
                pixels = samples["image"].contiguous(memory_format=torch.channels_last)
                session_queries += int(pixels.shape[0])
                ids_out = generate(
                    pixel_values=(pixels / 255.0 - mean) / std,
                    **preset,
                )
                assert ids_out.shape[0] == pixels.shape[0]
                return tokenizer.batch_decode(ids_out, skip_special_tokens=True)

        def load_adapter(**kwargs):
            return CaptionAdapter(), {"eval": raw_pixels}, {}

        overrides = {
            "EXPERIMENT_ROOT": out,
            "CLEAN_DIR": ii_dir / "clean_reference/class0",
            "MF_TT_DIR": tt_dir / "images/class0",
            "PER_ID_DIR": out / "captions/per_id",
            "COMBINED_FILE": captions_file,
            "OUTPUT_SIZE": (224, 224),
            "BATCH_SIZE": 1,
        }
        code = adapt(
            caption_source,
            overrides,
            slug,
            display,
            checkpoint,
            out / "audits/generate_21captions_adapted.py",
        )
        scope = {
            **overrides,
            "__name__": "__caption_formal__",
            "load_model_and_preprocess": load_adapter,
        }
        exec(
            compile(code, str(out / "audits/generate_21captions_adapted.py"), "exec"),
            scope,
        )
        payload = json.loads(captions_file.read_text(encoding="utf-8"))
        assert payload["summary"] == {
            "num_image_ids": 1000,
            "num_records": 2000,
            "num_captions_per_record": 21,
            "num_transform_captions_per_record": 20,
            "total_captions": 42000,
            "empty_caption_count": 0,
        }
        assert payload["paired_image_ids"] == ids
        expected = {(Path(row["image_key"]).name, "clean"): row["clean_caption"] for row in tt_rows}
        expected.update(
            {(Path(row["image_key"]).name, "mf_tt"): row["mftt_caption"] for row in tt_rows}
        )
        mismatches = [
            {
                "image_id": row["image_id"],
                "label": row["label"],
                "expected": expected[(row["image_id"], row["label"])],
                "actual": row["base_caption"],
            }
            for row in payload["records"]
            if row["base_caption"] != expected[(row["image_id"], row["label"])]
        ]
        report["base_caption_mismatches"] = mismatches
        report["base_caption_mismatch_count"] = len(mismatches)
        report["base_caption_match_count"] = 2000 - len(mismatches)
        report["base_caption_recheck_policy"] = 'record exact-text differences; current-run captions remain authoritative'
        report["caption_image_queries_this_session"] = session_queries
        report["caption_image_queries_committed"] = 42000
        dump(report, report_path)
        # Exact-text differences are retained in results.json for audit.
        payload["caption_model"].update(
            checkpoint=checkpoint,
            revision=revision,
            decoding=preset,
        )
        payload["native_resolution_note"] = (
            "Original fixed transform functions on native 224x224 attack-coordinate images."
        )
        dump(payload, captions_file)
        print(
            f"BASE_CAPTION_RECHECK_RECORDED: "
            f"{report['base_caption_match_count']}/2000 exact; "
            f"{report['base_caption_mismatch_count']} differences",
            flush=True,
        )
        report["rotation_degrees"] = payload["rotation_degrees"]
        report["rescale_factors"] = payload["rescale_factors"]
        del scope, load_adapter, CaptionAdapter, generate, victim, processor, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        feature_overrides = {
            "EXPERIMENT_ROOT": out,
            "CAPTIONS_FILE": captions_file,
            "FEATURE_JSON": features_file,
            "FEATURE_CSV": csv_file,
            "SUMMARY_JSON": summary_file,
        }
        code = adapt(
            feature_source,
            feature_overrides,
            slug,
            display,
            checkpoint,
            out / "audits/extract_features_adapted.py",
        )
        feature_scope = {**feature_overrides, "__name__": "__feature_formal__"}
        exec(
            compile(code, str(out / "audits/extract_features_adapted.py"), "exec"),
            feature_scope,
        )
        features = json.loads(features_file.read_text(encoding="utf-8"))
        assert features["summary"]["num_records"] == 2000
        assert features["summary"]["num_clean"] == 1000
        assert features["summary"]["num_mf_tt"] == 1000
        assert features["feature_order"] == ["d_center", "d_iqr", "d_tail", "d_q95"]
        matrix = np.array(
            [
                [row["features_all"]["features"][name] for name in features["feature_order"]]
                for row in features["records"]
            ]
        )
        assert matrix.shape == (2000, 4) and np.isfinite(matrix).all()
        report.update(
            status="complete",
            captions_file=str(captions_file),
            captions_file_sha256=digest(captions_file),
            feature_json=str(features_file),
            feature_json_sha256=digest(features_file),
            feature_csv=str(csv_file),
            feature_csv_sha256=digest(csv_file),
            feature_summary=str(summary_file),
            feature_summary_sha256=digest(summary_file),
            total_captions=42000,
            feature_shape=[2000, 4],
        )
        dump(report, report_path)
        print("Results:", report_path, flush=True)
        print(display + "_21CAPTIONS_FEATURES_FORMAL1000_PASS", flush=True)
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        dump(report, report_path)
        raise


if __name__ == "__main__":
    main()
