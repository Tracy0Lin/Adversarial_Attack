"""Run the already-tested MF-tt adapter on COCO1000 with image commits."""
import ast
import base64
import hashlib
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from PIL import Image


MODELS = {
    "vitgpt2": ("paper_vitgpt2_coco1000_v1", "run_vitgpt2_mftt_pilot5.py",
                "mftt5_20261006T163747Z_24697", "nlpconnect/vit-gpt2-image-captioning"),
    "git_large_coco": ("paper_git_large_coco1000_v1", "run_git_large_coco_mftt_pilot5.py",
                       "mftt5_20261006T173421Z_45148", "microsoft/git-large-coco"),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def pack_rng():
    n = np.random.get_state()
    def encode(t):
        return base64.b64encode(t.cpu().numpy().tobytes()).decode("ascii")
    return dict(python=random.getstate(), numpy=[n[0], n[1].tolist(), int(n[2]), int(n[3]), float(n[4])],
                cpu=encode(torch.random.get_rng_state()),
                cuda=[encode(t) for t in torch.cuda.get_rng_state_all()])


def restore_rng(state):
    def tuples(x):
        return tuple(tuples(v) for v in x) if isinstance(x, list) else x
    def decode(x):
        return torch.from_numpy(np.frombuffer(base64.b64decode(x), dtype=np.uint8).copy())
    random.setstate(tuples(state["python"]))
    n = state["numpy"]
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.random.set_rng_state(decode(state["cpu"]))
    assert len(state["cuda"]) == torch.cuda.device_count()
    torch.cuda.set_rng_state_all([decode(x) for x in state["cuda"]])


def import_helpers(path, slug):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {"adapt_legacy", "sampling_preset"}
    if slug == "git_large_coco":
        names.update({"_git_batch_inputs", "_generate_git_cached", "_generate_git_cached_impl"})
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in functions} == names, "Missing tested adapter functions"
    ns = dict(torch=torch, np=np, Path=Path, ast=ast, hashlib=hashlib, importlib=importlib)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), ns)
    return ns


def split_legacy(adapted):
    tree = ast.parse(adapted)
    entry = [n for n in tree.body if isinstance(n, ast.If)
             and ast.dump(n.test) == ast.dump(ast.parse("__name__ == '__main__'", mode="eval").body)]
    assert len(entry) == 1
    entry = entry[0]
    loops = [n for n in entry.body if isinstance(n, ast.For)
             and isinstance(n.target, ast.Tuple) and getattr(n.target.elts[0], "id", None) == "i"]
    assert len(loops) == 1
    loop = loops[0]
    assert entry.body[-1] is loop, "Unexpected work after the original image loop"
    first = loop.body[0]
    assert isinstance(first, ast.If) and len(first.body) == 1 and isinstance(first.body[0], ast.Break)
    body = ast.fix_missing_locations(ast.Module(body=loop.body[1:], type_ignores=[]))
    position = tree.body.index(entry)
    tree.body[position:position + 1] = entry.body[:entry.body.index(loop)]
    ast.fix_missing_locations(tree)
    return tree, body


def png(path):
    with Image.open(path) as opened:
        array = np.array(opened.convert("RGB"), copy=True)
    assert array.shape == (224, 224, 3), "Unexpected PNG size: " + str(path)
    return array


def check_budget(clean_path, adv_path):
    diff = np.abs(png(adv_path).astype(np.int16) - png(clean_path).astype(np.int16))
    assert int(diff.max()) <= 8, "PNG exceeds clean +/-8: " + str(adv_path)
    return int(diff.max()), float(diff.mean())


def read_commits(folder, rows, protocol_hash, image_field):
    expected = [r["image_key"] for r in rows]
    result = []
    for path in sorted(folder.glob("*.json")):
        commit = json.loads(path.read_text(encoding="utf-8"))
        assert commit["protocol_sha256"] == protocol_hash, "Commit protocol changed"
        record = commit["record"]
        assert len(result) < len(rows) and record["image_key"] == expected[len(result)], "Non-prefix commits"
        assert sha(record[image_field]) == record[image_field + "_sha256"], "Saved PNG changed"
        result.append(commit)
    return result


def main():
    project, slug = Path(sys.argv[1]), sys.argv[2]
    folder, adapter_name, pilot_name, checkpoint = MODELS[slug]
    root = project / "formal_tests" / folder
    ii_dir, out = root / "formal1000_v1/mfii", root / "formal1000_v1/mftt"
    ii_path = ii_dir / "results.json"
    previous = json.loads(ii_path.read_text(encoding="utf-8"))
    rows = sorted(previous["records"], key=lambda r: r["image_key"])
    assert previous["status"] == "complete" and len(rows) == 1000
    assert previous["checkpoint"] == checkpoint and previous["revision"]
    assert previous["torch"] == torch.__version__ and previous["transformers"] == transformers.__version__
    assert previous["args"]["steps"] == 100 and previous["args"]["epsilon"] == 8
    assert previous["args"]["alpha"] == 1.0 and previous["args"]["seed"] == 2023
    keys = [r["image_key"] for r in rows]
    assert len(set(keys)) == 1000
    manifest_path = project / "formal_tests/paper_coco1000_v1/manifests/coco1000_attack_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = {str(s["sample_id"]): s for s in manifest["samples"]}
    assert len(samples) == len(manifest["samples"]) == 1000
    targets = [samples[Path(k).name]["target_caption"].strip() for k in keys]
    assert all(t and "\n" not in t and "\r" not in t for t in targets)

    print("Checking MF-ii inputs:", slug, flush=True)
    for number, r in enumerate(rows, 1):
        key = r["image_key"]
        assert Path(samples[Path(key).name]["source_experiment_path"]).resolve() == Path(r["source"]).resolve()
        for field in ["source", "target", "clean_reference", "mfii_image"]:
            assert sha(r[field]) == r[field + "_sha256"], "MF-ii input changed: " + r[field]
        for field, subfolder in [("clean_reference", "clean_reference"), ("mfii_image", "images")]:
            assert Path(r[field]).resolve() == (ii_dir / subfolder / (key + ".png")).resolve()
        check_budget(r["clean_reference"], r["mfii_image"])
        if number % 100 == 0:
            print("INPUT_VERIFIED:", number, "/1000", flush=True)

    for subfolder in ["images", "clean_reference"]:
        found = sorted(p.relative_to(ii_dir / subfolder).with_suffix("").as_posix()
                       for p in (ii_dir / subfolder).rglob("*")
                       if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"})
        assert found == keys, "MF-ii folder does not match ordered records"

    adapter_path = project / "work_code" / adapter_name
    helpers = import_helpers(adapter_path, slug)
    pilot = json.loads((root / pilot_name / "results.json").read_text(encoding="utf-8"))
    assert pilot["status"] == "complete" and pilot["checkpoint"] == checkpoint
    assert pilot["revision"] == previous["revision"]
    assert pilot["torch"] == torch.__version__ and pilot["transformers"] == transformers.__version__
    sampled, lavis_source, lavis_hash = helpers["sampling_preset"]()
    deterministic = previous["caption_diagnostic_preset"]
    assert sampled == pilot["attack_caption_preset"]
    assert deterministic == pilot["evaluation_caption_preset"]
    legacy_path = project / "work_code/mf_tt_query.py"
    assert sha(legacy_path) == pilot["legacy_sha256"], "Original RGF changed after the pilot"
    assert lavis_hash == pilot["lavis_defaults_sha256"], "LAVIS sampling defaults changed"
    adapted = helpers["adapt_legacy"](legacy_path.read_text(encoding="utf-8"))
    setup, attack_body = split_legacy(adapted)
    cache_enabled = bool(pilot.get("git_vision_cache_enabled", False))

    protocol = dict(checkpoint=checkpoint, revision=previous["revision"],
                    mfii_results_sha256=sha(ii_path), manifest_sha256=sha(manifest_path),
                    adapter_sha256=sha(adapter_path), legacy_sha256=sha(legacy_path),
                    driver_sha256=sha(__file__), lavis_defaults_sha256=lavis_hash,
                    torch=torch.__version__, transformers=transformers.__version__,
                    attack_caption_preset=sampled, evaluation_caption_preset=deterministic,
                    git_vision_cache_enabled=cache_enabled, seed=2023,
                    steps=8, num_query=100, num_sub_query=50, sigma=8, epsilon=8, alpha=1)
    protocol_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    config = out / "frozen_protocol.json"
    if out.exists():
        assert config.is_file(), "Existing output has no frozen protocol"
        assert json.loads(config.read_text(encoding="utf-8")) == protocol, "Formal protocol changed"
    else:
        out.mkdir(parents=True)
        write_json(config, protocol)

    attack_dir, eval_dir = out / "per_image_attack", out / "per_image_eval"
    attack_dir.mkdir(exist_ok=True); eval_dir.mkdir(exist_ok=True)
    attacks = read_commits(attack_dir, rows, protocol_hash, "mftt_image")
    evaluations = read_commits(eval_dir, rows, protocol_hash, "mftt_image")
    assert len(evaluations) <= len(attacks)
    for a, e in zip(attacks, evaluations):
        assert a["record"]["mftt_image_sha256"] == e["record"]["mftt_image_sha256"]

    counter_path = out / "query_counts.json"
    counts = json.loads(counter_path.read_text()) if counter_path.exists() else dict(attack_images=0, evaluation_images=0)
    assert counts["attack_images"] >= len(attacks) * 816
    assert counts["evaluation_images"] >= len(evaluations) * 4

    report = dict(status="running", checkpoint=checkpoint, revision=previous["revision"],
                  input_pilot=str(ii_dir), manifest=str(manifest_path), legacy_script=str(legacy_path),
                  legacy_sha256=sha(legacy_path), lavis_defaults_source=lavis_source,
                  lavis_defaults_sha256=lavis_hash, torch=torch.__version__, transformers=transformers.__version__,
                  attack_caption_preset=sampled, evaluation_caption_preset=deterministic,
                  git_vision_cache_enabled=cache_enabled, protocol=protocol,
                  query_counts=counts, query_count_note="Requested image queries, including interrupted attempts; counters saved before generation.",
                  probe_policy="Original RGF: clamp probes to RGB range; output projected to clean +/-8",
                  seed=2023, epsilon=8, alpha=1, steps=8, num_query=100, num_sub_query=50, sigma=8,
                  attack_completed=len(attacks), evaluation_completed=len(evaluations),
                  records=[e["record"] for e in evaluations])

    def save_report():
        report["attack_completed"] = len(attacks)
        report["evaluation_completed"] = len(evaluations)
        write_json(out / "results.json", report)

    save_report()
    print("Model:", checkpoint, "\nOutput:", out, flush=True)
    print("RESUME: attack", len(attacks), "/1000; evaluation", len(evaluations), "/1000", flush=True)
    if len(evaluations) == 1000:
        report["status"] = "complete"; save_report()
        print("MFTT_" + slug.upper() + "_FORMAL1000_PASS", flush=True)
        return

    assert torch.cuda.is_available(), "GPU unavailable"
    print("Loading victim:", checkpoint, flush=True)
    revision = previous["revision"]
    if slug == "vitgpt2":
        from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
        model = VisionEncoderDecoderModel.from_pretrained(checkpoint, revision=revision).eval().to("cuda")
        processor = ViTImageProcessor.from_pretrained(checkpoint, revision=revision)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, revision=revision)
        image_processor = processor
    else:
        from transformers import GitForCausalLM, AutoProcessor
        model = GitForCausalLM.from_pretrained(checkpoint, revision=revision).eval().to("cuda")
        processor = AutoProcessor.from_pretrained(checkpoint, revision=revision)
        tokenizer, image_processor = processor.tokenizer, processor.image_processor
        helpers["_GIT_CACHE_ENABLED"] = cache_enabled

    for parameter in model.parameters(): parameter.requires_grad_(False)
    assert image_processor.image_mean == previous["image_mean"]
    assert image_processor.image_std == previous["image_std"]
    assert image_processor.do_normalize
    mean = torch.tensor(image_processor.image_mean, device="cuda").view(1, 3, 1, 1)
    std = torch.tensor(image_processor.image_std, device="cuda").view(1, 3, 1, 1)
    assert bool((std > 0).all())
    phase = "attack_images"

    def caption(pixels, sampled_mode):
        assert pixels.shape[1:] == (3, 224, 224)
        counts[phase] += int(pixels.shape[0])
        write_json(counter_path, counts)
        options = dict(pixel_values=(pixels / 255.0 - mean) / std)
        options.update(sampled if sampled_mode else deterministic)
        with torch.inference_mode():
            ids = model.generate(**options) if slug == "vitgpt2" else helpers["_generate_git_cached"](model, **options)
        texts = [t.strip() for t in tokenizer.batch_decode(ids, skip_special_tokens=True)]
        assert len(texts) == pixels.shape[0]
        return texts

    def i2t(args, txt_processors, victim, image): return caption(image, True)

    mfii_text, target_text = out / "mfii_captions.txt", out / "target_captions.txt"
    write_contents = [(mfii_text, "\n".join(" ".join(r["mfii_caption"].split()) for r in rows) + "\n"),
                      (target_text, "\n".join(targets) + "\n")]
    for path, text in write_contents:
        if path.exists(): assert path.read_text(encoding="utf-8") == text, "Caption input changed"
        else: path.write_text(text, encoding="utf-8")

    (out / "best_captions.txt").write_text(
        "\n".join(" ".join(a["record"]["historical_best_sampled_caption"].split()) for a in attacks)
        + ("\n" if attacks else ""), encoding="utf-8")
    (out / "legacy_adapted.py").write_text(adapted, encoding="utf-8")
    (out / "attack_body.py").write_text(ast.unparse(attack_body) + "\n", encoding="utf-8")

    namespace = {"__name__": "__main__", "__file__": str(legacy_path),
                 "_vit_i2t": i2t, "_git_i2t": i2t,
                 "load_model_and_preprocess": lambda **kwargs: (model, {}, {})}
    sys.argv = [str(legacy_path), "--model_name", slug, "--model_type", checkpoint,
                "--batch_size", "1", "--num_workers", "0", "--num_samples", "1000", "--input_res", "224",
                "--alpha", "1", "--epsilon", "8", "--steps", "8", "--num_query", "100",
                "--num_sub_query", "50", "--sigma", "8", "--data_path", str(ii_dir / "images"),
                "--clean_dir", str(ii_dir / "clean_reference"), "--text_path", str(mfii_text),
                "--target_text_path", str(target_text), "--image_output", str(out / "images"),
                "--output", str(out / "best_captions")]

    try:
        exec(compile(setup, str(legacy_path), "exec"), namespace)
        attack_code = compile(attack_body, str(legacy_path), "exec")
        iterator = iter(zip(namespace["data_loader"], namespace["clean_data_loader"]))
        already = len(attacks)
        start = time.perf_counter()

        for i, ((image, label, path), (image_clean, clean_label, clean_path)) in enumerate(iterator):
            assert i < 1000
            r, key = rows[i], rows[i]["image_key"]
            assert Path(path[0]).resolve() == Path(r["mfii_image"]).resolve()
            assert Path(clean_path[0]).resolve() == Path(r["clean_reference"]).resolve()
            if i < already: continue
            if i == already and attacks: restore_rng(attacks[-1]["rng_state"])
            namespace.update(i=i, image=image, image_clean=image_clean, path=path)
            count_before = counts["attack_images"]
            exec(attack_code, namespace)
            assert counts["attack_images"] - count_before == 816, "Attack query count changed"
            final_path = out / "images" / (key + ".png")
            maximum, average = check_budget(r["clean_reference"], final_path)
            record = dict(image_key=key, mftt_image=str(final_path), mftt_image_sha256=sha(final_path),
                          max_pixel_difference=maximum, mean_pixel_difference=average,
                          historical_best_sampled_caption=namespace["best_caption"],
                          historical_best_sampled_similarity=float(namespace["query_attack_results"][i]),
                          attack_image_queries=816)
            commit = dict(protocol_sha256=protocol_hash, record=record, rng_state=pack_rng())
            write_json(attack_dir / (str(i + 1).zfill(4) + ".json"), commit)
            attacks.append(commit); save_report()
            elapsed = time.perf_counter() - start
            average_seconds = elapsed / (len(attacks) - already)
            print(f"ATTACK_COMMIT {slug} {i+1}/1000 | maxdiff={maximum} | "
                  f"mean={average_seconds:.1f}s/image | ETA={(1000-i-1)*average_seconds/3600:.2f}h", flush=True)

        assert len(attacks) == 1000
        phase = "evaluation_images"
        clip, clip_model = namespace["clip"], namespace["clip_img_model_vitb32"]

        def score(text, target):
            with torch.inference_mode():
                features = clip_model.encode_text(clip.tokenize([text, target]).to("cuda"))
                features = features / features.norm(dim=1, keepdim=True)
            return float((features[0] * features[1]).sum())

        def read_tensor(path):
            array = png(path)
            return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to("cuda").float()

        restore_rng(evaluations[-1]["rng_state"] if evaluations else attacks[-1]["rng_state"])

        for i in range(len(evaluations), 1000):
            r, a, target = rows[i], attacks[i]["record"], targets[i]
            before = counts["evaluation_images"]
            caps = [caption(read_tensor(p), False)[0] for p in [r["clean_reference"], r["mfii_image"], a["mftt_image"]]]
            sims = [score(c, target) for c in caps]
            fresh = caption(read_tensor(a["mftt_image"]), True)[0]
            assert counts["evaluation_images"] - before == 4
            record = dict(a, source_coco_image_id=samples[Path(r["image_key"]).name]["source_coco_image_id"],
                          target_caption=target, mfii_image=r["mfii_image"], clean_reference=r["clean_reference"],
                          clean_caption=caps[0], mfii_caption=caps[1], mftt_caption=caps[2],
                          clean_target_similarity=sims[0], mfii_target_similarity=sims[1], mftt_target_similarity=sims[2],
                          fresh_saved_sampled_caption=fresh, fresh_saved_sampled_similarity=score(fresh, target),
                          prior_clean_caption_matches=caps[0] == r["clean_caption"],
                          prior_mfii_caption_matches=caps[1] == r["mfii_caption"])
            commit = dict(protocol_sha256=protocol_hash, record=record, rng_state=pack_rng())
            write_json(eval_dir / (str(i + 1).zfill(4) + ".json"), commit)
            evaluations.append(commit); report["records"].append(record); save_report()
            print(f"EVAL {slug} {i+1}/1000 {r['image_key']}: "
                  f"clean={sims[0]:.6f} MF-ii={sims[1]:.6f} MF-tt={sims[2]:.6f}", flush=True)

        report["status"] = "complete"; save_report()
    except BaseException as error:
        report.update(status="failed", error=repr(error)); save_report(); raise

    for label, field in [("Clean", "clean_target_similarity"), ("MF-ii", "mfii_target_similarity"),
                         ("MF-tt", "mftt_target_similarity")]:
        passed = sum(r[field] >= 0.80 for r in report["records"])
        print(f"Fixed decoding, target similarity >=0.80: {label} {passed}/1000", flush=True)
    print("Base caption exact matches: Clean", sum(r["prior_clean_caption_matches"] for r in report["records"]),
          "/1000; MF-ii", sum(r["prior_mfii_caption_matches"] for r in report["records"]), "/1000", flush=True)
    print("Query counts:", counts, flush=True)
    print("Results:", out / "results.json", flush=True)
    print("MFTT_" + slug.upper() + "_FORMAL1000_PASS", flush=True)


if __name__ == "__main__":
    main()
