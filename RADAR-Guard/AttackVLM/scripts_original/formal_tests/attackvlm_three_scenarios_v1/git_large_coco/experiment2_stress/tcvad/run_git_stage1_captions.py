#!/usr/bin/env python3

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoProcessor,
    GitForCausalLM,
)

PROJECT = Path(
    "/workspace/projects/couser/"
    "AttackVLM_Official_Reproduction"
)
SOURCE = (
    PROJECT /
    "formal_tests/paper_coco1000_v1/tcvad/"
    "experiment1_blip/scripts/"
    "generate_blip_21captions_all.py"
)

CHECKPOINT = "microsoft/git-large-coco"
REVISION = "b644b0b41274649f87de8846fa15f9a22dad7583"
HELPER = (
    PROJECT /
    "work_code/run_git_large_coco_mftt_pilot5.py"
)
PRESET = {
    "do_sample": False,
    "num_beams": 4,
    "max_new_tokens": 32,
}


def adapt(source_path, overrides, count, audit_path):
    tree = ast.parse(source_path.read_text(encoding="utf-8"))

    tree.body = [
        node for node in tree.body
        if not (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id in overrides
                for target in node.targets
            )
        )
        and not (
            isinstance(node, ast.ImportFrom)
            and node.module == "lavis.models"
        )
    ]

    replacements = {
        1000: count,
        2000: count * 2,
        42000: count * 42,
    }

    class Adapter(ast.NodeTransformer):
        def visit_Constant(self, node):
            if type(node.value) is int and node.value in replacements:
                node.value = replacements[node.value]

            if isinstance(node.value, str):
                node.value = (
                    node.value
                    .replace("BLIP base_coco", "GIT-large-COCO")
                    .replace("blip_caption", "git_large_coco")
                    .replace("base_coco", CHECKPOINT)
                    .replace("BLIP", "GIT-large-COCO")
                    .replace("blip", "git_large_coco")
                )
            return node

    tree = Adapter().visit(tree)
    ast.fix_missing_locations(tree)
    code = ast.unparse(tree) + "\n"

    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(code, encoding="utf-8")
    compile(code, str(audit_path), "exec")
    return code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-dir", required=True)
    parser.add_argument("--attack-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--count", required=True, type=int)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "GPU unavailable"

    clean_dir = Path(args.clean_dir).resolve()
    attack_dir = Path(args.attack_dir).resolve()
    output_root = Path(args.output_root).resolve()

    assert len(list(clean_dir.iterdir())) == args.count
    assert len(list(attack_dir.iterdir())) == args.count

    torch.manual_seed(20260729)
    torch.cuda.manual_seed_all(20260729)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("Loading GIT-large-COCO...", flush=True)

    model = GitForCausalLM.from_pretrained(
        CHECKPOINT,
        revision=REVISION,
    ).eval().to("cuda")

    processor = AutoProcessor.from_pretrained(
        CHECKPOINT,
        revision=REVISION,
    )
    tokenizer = processor.tokenizer
    image_processor = processor.image_processor

    helper_tree = ast.parse(
        HELPER.read_text(encoding="utf-8")
    )
    helper_names = {
        "_git_batch_inputs",
        "_generate_git_cached",
        "_generate_git_cached_impl",
    }
    helper_functions = [
        node for node in helper_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in helper_names
    ]

    if {
        node.name for node in helper_functions
    } != helper_names:
        raise RuntimeError(
            "找不到完整的 GIT batch adapter。"
        )

    helper_scope = {
        "torch": torch,
        "_GIT_CACHE_ENABLED": True,
    }
    exec(
        compile(
            ast.Module(
                body=helper_functions,
                type_ignores=[],
            ),
            str(HELPER),
            "exec",
        ),
        helper_scope,
    )

    def generate(**kwargs):
        return helper_scope["_generate_git_cached"](
            model,
            **kwargs,
        )

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    mean = torch.tensor(
        image_processor.image_mean,
        device="cuda",
    ).view(1, 3, 1, 1)

    std = torch.tensor(
        image_processor.image_std,
        device="cuda",
    ).view(1, 3, 1, 1)

    class CaptionAdapter:
        def eval(self):
            return self

        def generate(self, samples):
            pixels = samples["image"].contiguous(
                memory_format=torch.channels_last
            )
            ids = generate(
                pixel_values=(pixels / 255.0 - mean) / std,
                **PRESET,
            )
            return tokenizer.batch_decode(
                ids,
                skip_special_tokens=True,
            )

    def raw_pixels(image):
        rgb = np.array(image.convert("RGB"), copy=True)
        assert rgb.shape == (224, 224, 3)
        return torch.from_numpy(rgb).permute(2, 0, 1).float()

    def load_adapter(**kwargs):
        return CaptionAdapter(), {"eval": raw_pixels}, {}

    combined = (
        output_root /
        f"captions/git_large_coco_21captions_{args.count}pairs.json"
    )
    overrides = {
        "EXPERIMENT_ROOT": output_root,
        "CLEAN_DIR": clean_dir,
        "MF_TT_DIR": attack_dir,
        "PER_ID_DIR": output_root / "captions/per_id",
        "COMBINED_FILE": combined,
        "OUTPUT_SIZE": (224, 224),
        "BATCH_SIZE": 1,
    }

    audit = output_root / "audits/generate_21captions_adapted.py"
    code = adapt(SOURCE, overrides, args.count, audit)

    scope = {
        **overrides,
        "__name__": "__caption_subset__",
        "load_model_and_preprocess": load_adapter,
    }
    exec(compile(code, str(audit), "exec"), scope)

    result = json.loads(combined.read_text(encoding="utf-8"))
    summary = result["summary"]

    assert summary["num_image_ids"] == args.count
    assert summary["num_records"] == args.count * 2
    assert summary["total_captions"] == args.count * 42
    assert summary["empty_caption_count"] == 0

    print("Output:", combined, flush=True)
    print("Summary:", summary, flush=True)
    print("GIT_STAGE1_CAPTIONS_OK", flush=True)


if __name__ == "__main__":
    main()
