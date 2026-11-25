import ast
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
import transformers
from PIL import Image
from transformers import GitForCausalLM, AutoProcessor
_GIT_CACHE_ENABLED = True

def _git_batch_inputs(model, generation_kwargs):
    kwargs = dict(generation_kwargs)
    pixels = kwargs['pixel_values']
    assert pixels.ndim == 4 and pixels.shape[0] > 0
    if kwargs.get('input_ids') is None:
        config = kwargs.get('generation_config') or model.generation_config
        bos = kwargs.get('bos_token_id')
        if bos is None:
            bos = config.bos_token_id
        if bos is None:
            bos = model.config.bos_token_id
        assert bos is not None, 'Missing GIT beginning-of-sequence token'
        kwargs['input_ids'] = torch.full((pixels.shape[0], 1), int(bos), dtype=torch.long, device=pixels.device)
    assert kwargs['input_ids'].shape[0] == pixels.shape[0], 'GIT input batch mismatch'
    return kwargs

def _generate_git_cached(model, **generation_kwargs):
    kwargs = _git_batch_inputs(model, generation_kwargs)
    ids = _generate_git_cached_impl(model, **kwargs)
    config = kwargs.get('generation_config') or model.generation_config
    returns = int(kwargs.get('num_return_sequences', config.num_return_sequences))
    assert ids.ndim == 2 and ids.shape[0] == kwargs['pixel_values'].shape[0] * returns, 'GIT did not return the expected number of captions'
    return ids

def _generate_git_cached_impl(model, **generation_kwargs):
    """Reuse identical vision outputs only within one generation call."""
    if not _GIT_CACHE_ENABLED:
        return model.generate(**generation_kwargs)
    encoder = model.git.image_encoder
    original_forward = encoder.forward
    cache = {}

    def cached_forward(*args, **kwargs):
        pixels = kwargs.get('pixel_values', args[0] if args else None)
        options = tuple(args[1:]) + tuple(((k, v) for (k, v) in sorted(kwargs.items()) if k != 'pixel_values'))
        option_values = list(args[1:]) + [v for (k, v) in kwargs.items() if k != 'pixel_values']
        simple = all((isinstance(v, (str, bool, int, float, type(None))) for v in option_values))
        if pixels is None or encoder.training or torch.is_grad_enabled() or (not simple):
            cache.clear()
            return original_forward(*args, **kwargs)
        identity = (tuple(pixels.shape), pixels.dtype, pixels.device)
        if cache.get('identity') == identity and cache.get('options') == options and torch.equal(pixels, cache['pixels']):
            return cache['output']
        output = original_forward(*args, **kwargs)
        cache.update(identity=identity, options=options, pixels=pixels.detach().clone(), output=output)
        return output
    encoder.forward = cached_forward
    try:
        return model.generate(**generation_kwargs)
    finally:
        encoder.forward = original_forward

def adapt_legacy(source):
    """Change the caption interface and resolution; keep the original RGF body."""
    tree = ast.parse(source)
    edits = []
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_i2t']
    assert len(functions) == 1, 'Expected one original _i2t function'
    edits.append((functions[0], '_i2t = _git_i2t'))
    for n in tree.body:
        if isinstance(n, ast.ImportFrom) and n.module in {'lavis.common.gradcam', 'lavis.models'}:
            edits.append((n, '# LAVIS caption interface replaced by GIT-large-COCO.'))
        if isinstance(n, ast.Import) and len(n.names) == 1 and (n.names[0].name == 'wandb'):
            edits.append((n, '# This pilot does not enable wandb.'))
    resolution = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.If):
            continue
        test = ast.dump(n.test, include_attributes=False)
        expected = ast.dump(ast.parse("args.model_name == 'blip2_opt'", mode='eval').body)
        if test == expected:
            resolution.append(n)
    assert len(resolution) == 1, 'Original victim-resolution switch changed'
    edits.append((resolution[0], 'args.input_res = 224'))
    lines = source.splitlines(keepends=True)
    for (n, replacement) in sorted(edits, key=lambda e: e[0].lineno, reverse=True):
        lines[n.lineno - 1:n.end_lineno] = [' ' * n.col_offset + replacement + '\n']
    adapted = ''.join(lines).replace('Loading LAVIS models:', 'Loading GIT-large-COCO adapter:')
    caption_write = "f.write(best_caption.strip() + '\\n')"
    single_line_write = "f.write(' '.join(best_caption.split()) + '\\n')"
    assert adapted.count(caption_write) == 1, 'Original caption logging statement changed'
    adapted = adapted.replace(caption_write, single_line_write)
    ast.parse(adapted)
    return adapted

def sampling_preset():
    """Read BLIP's generate_from_encoder bridge and actual nucleus call."""
    spec = importlib.util.find_spec('lavis')
    assert spec is not None and spec.submodule_search_locations, 'Cannot locate LAVIS'
    root = Path(next(iter(spec.submodule_search_locations)))
    caption_path = root / 'models/blip_models/blip_caption.py'
    decoder_path = root / 'models/med.py'
    caption_source = caption_path.read_text(encoding='utf-8')
    decoder_source = decoder_path.read_text(encoding='utf-8')
    cls = next((n for n in ast.parse(caption_source).body if isinstance(n, ast.ClassDef) and n.name == 'BlipCaption'))
    caption_method = next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'generate'))
    helpers = [n for n in ast.walk(ast.parse(decoder_source)) if isinstance(n, ast.FunctionDef) and n.name == 'generate_from_encoder']
    assert len(helpers) == 1, 'Expected one generate_from_encoder helper'
    helper = helpers[0]

    def defaults(fn):
        params = fn.args.posonlyargs + fn.args.args
        result = {a.arg: ast.literal_eval(d) for (a, d) in zip(params[len(params) - len(fn.args.defaults):], fn.args.defaults)}
        result.update({a.arg: ast.literal_eval(d) for (a, d) in zip(fn.args.kwonlyargs, fn.args.kw_defaults) if d is not None})
        return result

    def resolve(node, env):
        return env[node.id] if isinstance(node, ast.Name) else ast.literal_eval(node)
    caption_env = defaults(caption_method)
    caption_env.update(use_nucleus_sampling=True, num_captions=1)
    bridges = [n for n in ast.walk(caption_method) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and (n.func.attr == 'generate_from_encoder')]
    assert len(bridges) == 1, "Expected BLIP's decoder bridge"
    helper_env = defaults(helper)
    for kw in bridges[0].keywords:
        if kw.arg in helper_env:
            helper_env[kw.arg] = resolve(kw.value, caption_env)
    assert helper_env['use_nucleus_sampling'] is True
    branches = [n for n in ast.walk(helper) if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and (n.test.id == 'use_nucleus_sampling')]
    assert len(branches) == 1, 'Expected the actual nucleus branch in med.py'
    calls = [n for stmt in branches[0].body for n in ast.walk(stmt) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and (n.func.attr == 'generate')]
    assert len(calls) == 1, 'Expected one nucleus generation call'
    preset = {'do_sample': True, 'num_beams': 1, 'num_return_sequences': 1}
    allowed = {'do_sample', 'top_p', 'top_k', 'temperature', 'max_length', 'max_new_tokens', 'min_length', 'min_new_tokens', 'repetition_penalty', 'length_penalty', 'num_beams', 'num_return_sequences', 'early_stopping'}
    for kw in calls[0].keywords:
        if kw.arg in allowed:
            preset[kw.arg] = resolve(kw.value, helper_env)
    assert preset['do_sample'] is True and preset['num_return_sequences'] == 1
    assert 'top_p' in preset and ('max_length' in preset or 'max_new_tokens' in preset)
    print('Installed BLIP sampling settings:', preset, flush=True)
    digest = hashlib.sha256(caption_source.encode() + b'\x00' + decoder_source.encode()).hexdigest()
    return (preset, {'caption': str(caption_path), 'decoder': str(decoder_path)}, digest)

def resume_saved_evaluation(project, pilot, out):
    import clip
    import random
    import shutil
    previous = json.loads((pilot / 'results.json').read_text(encoding='utf-8'))
    report_path = out / 'results.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    rows = sorted(previous['records'], key=lambda r: r['image_key'])
    assert previous['status'] == 'complete' and len(rows) == 5
    assert report['checkpoint'] == previous['checkpoint']
    assert Path(report['input_pilot']).resolve() == pilot.resolve()
    assert report.get('revision') == previous.get('revision')
    assert report['query_counts']['attack_images'] == 4080
    assert torch.cuda.is_available(), 'GPU unavailable'
    keys = [r['image_key'] for r in rows]
    found = sorted((str(p.relative_to(out / 'images').with_suffix('')) for p in (out / 'images').rglob('*.png')))
    assert found == keys, 'Saved MF-tt images do not match the five input IDs'
    manifest = json.loads(Path(report['manifest']).read_text(encoding='utf-8'))
    samples = {s['sample_id']: s for s in manifest['samples']}
    targets = [samples[Path(k).name]['target_caption'].strip() for k in keys]
    assert (out / 'target_captions.txt').read_text(encoding='utf-8').splitlines() == targets
    raw_best_path = out / 'best_captions.txt'
    assert raw_best_path.is_file(), 'Original caption log is missing'
    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    backup = out / ('results_before_saved_eval_' + stamp + '_' + str(os.getpid()) + '.json')
    shutil.copy2(report_path, backup)
    report['previous_error'] = report.pop('error', None)
    report['status'] = 'evaluating_saved_images'
    report['evaluation_resumed'] = True
    report['attack_rerun'] = False
    report['saved_evaluation_seed'] = 2023
    report['historical_caption_text_path'] = str(raw_best_path)
    report['historical_score_note'] = 'Ambiguous multiline caption log retained; original in-memory best scores unavailable. Saved PNGs evaluated independently.'
    report['query_counts']['evaluation_images'] = 0
    report['records'] = []

    def save_report():
        temporary = out / 'results.tmp'
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
        temporary.replace(report_path)
    save_report()
    print('Evaluate saved images:', out, flush=True)
    print('Previous report backup:', backup, flush=True)
    (checkpoint, revision) = (previous['checkpoint'], previous.get('revision'))
    model = GitForCausalLM.from_pretrained(checkpoint, revision=revision).eval().to('cuda')
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    processor = AutoProcessor.from_pretrained(checkpoint, revision=revision)
    tokenizer = processor.tokenizer
    assert processor.image_processor.image_mean == previous['image_mean']
    assert processor.image_processor.image_std == previous['image_std']
    mean = torch.tensor(processor.image_processor.image_mean, device='cuda').view(1, 3, 1, 1)
    std = torch.tensor(processor.image_processor.image_std, device='cuda').view(1, 3, 1, 1)
    assert processor.image_processor.do_normalize and bool((std > 0).all())
    (clip_model, _) = clip.load('ViT-B/32', device='cuda', jit=False)
    deterministic = previous['caption_diagnostic_preset']
    sampled = report['attack_caption_preset']
    assert deterministic == report['evaluation_caption_preset']
    random.seed(2023)
    np.random.seed(2023)
    torch.manual_seed(2023)
    torch.cuda.manual_seed_all(2023)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    def read_image(path):
        with Image.open(path) as im:
            rgb = np.array(im.convert('RGB'), copy=True)
        assert rgb.shape == (224, 224, 3)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to('cuda').float()
        return (rgb, tensor)

    def caption(pixels, preset):
        report['query_counts']['evaluation_images'] += pixels.shape[0]
        with torch.inference_mode():
            ids = _generate_git_cached(model, pixel_values=(pixels / 255.0 - mean) / std, **preset)
        return tokenizer.batch_decode(ids, skip_special_tokens=True)[0].strip()

    def score(text, target):
        with torch.inference_mode():
            feats = clip_model.encode_text(clip.tokenize([text, target]).to('cuda'))
            feats = feats / feats.norm(dim=1, keepdim=True)
            return float((feats[0] * feats[1]).sum())
    print('\nSaved-image evaluation: CLIP ViT-B/32 text cosine', flush=True)
    try:
        for (i, (r, target)) in enumerate(zip(rows, targets)):
            key = r['image_key']
            (clean_rgb, clean) = read_image(r['clean_reference'])
            (_, mfii) = read_image(r['mfii_image'])
            final_path = out / 'images' / (key + '.png')
            (final_rgb, final) = read_image(final_path)
            diff = np.abs(final_rgb.astype(np.int16) - clean_rgb.astype(np.int16))
            assert int(diff.max()) <= 8, 'Saved image exceeds total clean-image budget'
            caps = [caption(img, deterministic) for img in [clean, mfii, final]]
            sims = [score(c, target) for c in caps]
            fresh = caption(final, sampled)
            report['records'].append(dict(image_key=key, source_coco_image_id=samples[Path(key).name]['source_coco_image_id'], target_caption=target, mfii_image=r['mfii_image'], mftt_image=str(final_path), max_pixel_difference=int(diff.max()), mean_pixel_difference=float(diff.mean()), clean_caption=caps[0], mfii_caption=caps[1], mftt_caption=caps[2], clean_target_similarity=sims[0], mfii_target_similarity=sims[1], mftt_target_similarity=sims[2], historical_best_sampled_caption=None, historical_best_sampled_similarity=None, fresh_saved_sampled_caption=fresh, fresh_saved_sampled_similarity=score(fresh, target), prior_clean_caption_matches=caps[0] == r['clean_caption'], prior_mfii_caption_matches=caps[1] == r['mfii_caption']))
            save_report()
            print(f'{key}: clean={sims[0]:.6f} MF-ii={sims[1]:.6f} MF-tt={sims[2]:.6f}; maxdiff={diff.max()}', flush=True)
            print('Target:', target, flush=True)
            print('MF-tt:', json.dumps(caps[2], ensure_ascii=False), flush=True)
        assert report['query_counts']['evaluation_images'] == 20
        report['status'] = 'complete'
        save_report()
    except BaseException as error:
        report['status'] = 'failed'
        report['error'] = repr(error)
        save_report()
        raise
    for (label, field) in [('Clean', 'clean_target_similarity'), ('MF-ii', 'mfii_target_similarity'), ('MF-tt', 'mftt_target_similarity')]:
        passed = sum((r[field] >= 0.8 for r in report['records']))
        print(f'Fixed decoding, target similarity >=0.80: {label} {passed}/5', flush=True)
    print('Query counts:', report['query_counts'], flush=True)
    print('Results:', report_path, flush=True)
    print('MFTT_GIT_LARGE_COCO_PILOT_PASS', flush=True)

def main():
    (project, pilot) = map(Path, sys.argv[1:3])
    if len(sys.argv) > 3:
        resume_saved_evaluation(project, pilot, Path(sys.argv[3]))
        return
    previous = json.loads((pilot / 'results.json').read_text(encoding='utf-8'))
    rows = sorted(previous['records'], key=lambda r: r['image_key'])
    assert previous['status'] == 'complete' and len(rows) == 5, 'Need a completed pilot5'
    assert torch.cuda.is_available(), 'GPU unavailable'
    manifest_path = project / 'formal_tests/paper_coco1000_v1/manifests/coco1000_attack_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    samples = {s['sample_id']: s for s in manifest['samples']}
    assert len(samples) == len(manifest['samples']), 'Duplicate manifest IDs'
    keys = [r['image_key'] for r in rows]
    assert len(set(keys)) == 5
    for r in rows:
        key = r['image_key']
        s = samples[Path(key).name]
        assert Path(s['source_experiment_path']).resolve() == Path(r['source']).resolve()
        assert hashlib.sha256(Path(r['source']).read_bytes()).hexdigest() == r['source_sha256']
        for (field, folder) in [('mfii_image', 'images'), ('clean_reference', 'clean_reference')]:
            assert Path(r[field]).resolve() == (pilot / folder / (key + '.png')).resolve()
            with Image.open(r[field]) as im:
                assert im.size == (224, 224), 'Unexpected pilot image resolution'
    for folder in ['images', 'clean_reference']:
        found = sorted((str(p.relative_to(pilot / folder).with_suffix('')) for p in (pilot / folder).rglob('*') if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tif', '.tiff'}))
        assert found == keys, 'Image-folder order or contents do not match pilot records'
    legacy_path = project / 'work_code/mf_tt_query.py'
    legacy = legacy_path.read_text(encoding='utf-8')
    adapted = adapt_legacy(legacy)
    (sampled, lavis_source, lavis_hash) = sampling_preset()
    checkpoint = previous['checkpoint']
    assert checkpoint == 'microsoft/git-large-coco'
    revision = previous.get('revision')
    model = GitForCausalLM.from_pretrained(checkpoint, revision=revision).eval().to('cuda')
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    processor = AutoProcessor.from_pretrained(checkpoint, revision=revision)
    tokenizer = processor.tokenizer
    mean = torch.tensor(processor.image_processor.image_mean, device='cuda').view(1, 3, 1, 1)
    std = torch.tensor(processor.image_processor.image_std, device='cuda').view(1, 3, 1, 1)
    assert processor.image_processor.do_normalize and bool((std > 0).all())
    assert processor.image_processor.image_mean == previous['image_mean'] and processor.image_processor.image_std == previous['image_std']
    deterministic = previous['caption_diagnostic_preset']
    global _GIT_CACHE_ENABLED
    assert hasattr(model, 'git') and hasattr(model.git, 'image_encoder')
    with Image.open(rows[0]['clean_reference']) as im:
        array = np.array(im.convert('RGB'), copy=True)
    assert array.shape == (224, 224, 3)
    pixels = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to('cuda').float()
    pixels = (pixels / 255.0 - mean) / std
    cpu_rng = torch.random.get_rng_state()
    gpu_rng = torch.cuda.get_rng_state_all()
    try:
        with torch.inference_mode():
            native_ids = model.generate(pixel_values=pixels, **deterministic)
            cached_ids = _generate_git_cached(model, pixel_values=pixels, **deterministic)
        _GIT_CACHE_ENABLED = bool(torch.equal(native_ids, cached_ids))
    finally:
        torch.random.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state_all(gpu_rng)
    if _GIT_CACHE_ENABLED:
        print('GIT_VISION_CACHE_PREFLIGHT_PASS: cached/native token IDs identical', flush=True)
    else:
        print('GIT cache disabled: using native generation', flush=True)
    del pixels, native_ids, cached_ids
    batch_images = []
    for row in rows[:2]:
        with Image.open(row['clean_reference']) as im:
            array = np.array(im.convert('RGB'), copy=True)
        assert array.shape == (224, 224, 3)
        batch_images.append(torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0))
    batch_pixels = torch.cat(batch_images, dim=0).to('cuda').float()
    batch_pixels = (batch_pixels / 255.0 - mean) / std
    cpu_rng = torch.random.get_rng_state()
    gpu_rng = torch.cuda.get_rng_state_all()
    try:
        with torch.inference_mode():
            native_kwargs = _git_batch_inputs(model, dict(pixel_values=batch_pixels, **sampled))
            batch_native_ids = model.generate(**native_kwargs)
        torch.random.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state_all(gpu_rng)
        with torch.inference_mode():
            batch_cached_ids = _generate_git_cached(model, pixel_values=batch_pixels, **sampled)
        assert batch_native_ids.shape[0] == 2 and batch_cached_ids.shape[0] == 2
        if _GIT_CACHE_ENABLED and (not torch.equal(batch_native_ids, batch_cached_ids)):
            _GIT_CACHE_ENABLED = False
            print('GIT batch cache comparison differed: using native generation', flush=True)
        print('GIT_BATCH_PREFLIGHT_PASS: 2 images -> 2 captions', flush=True)
    finally:
        torch.random.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state_all(gpu_rng)
    del batch_images, batch_pixels, batch_native_ids, batch_cached_ids, native_kwargs
    counts = {'attack_images': 0, 'evaluation_images': 0}
    phase = 'attack_images'

    def caption(pixels, sampled_mode):
        assert pixels.shape[1:] == (3, 224, 224)
        counts[phase] += pixels.shape[0]
        with torch.inference_mode():
            ids = _generate_git_cached(model, pixel_values=(pixels / 255.0 - mean) / std, **sampled if sampled_mode else deterministic)
        return [t.strip() for t in tokenizer.batch_decode(ids, skip_special_tokens=True)]

    def git_i2t(args, txt_processors, victim, image):
        return caption(image, True)

    def load_adapter(**kwargs):
        return (model, {}, {})
    out = pilot.parent / ('mftt5_' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '_' + str(os.getpid()))
    out.mkdir()
    mfii_text = out / 'mfii_captions.txt'
    target_text = out / 'target_captions.txt'
    targets = [samples[Path(k).name]['target_caption'].strip() for k in keys]
    assert all((t and '\n' not in t and ('\r' not in t) for t in targets))
    mfii_text.write_text('\n'.join((r['mfii_caption'] for r in rows)) + '\n', encoding='utf-8')
    target_text.write_text('\n'.join(targets) + '\n', encoding='utf-8')
    report = dict(status='running', checkpoint=checkpoint, revision=revision, input_pilot=str(pilot), manifest=str(manifest_path), legacy_script=str(legacy_path), legacy_sha256=hashlib.sha256(legacy.encode()).hexdigest(), lavis_defaults_source=lavis_source, lavis_defaults_sha256=lavis_hash, torch=torch.__version__, transformers=transformers.__version__, attack_caption_preset=sampled, evaluation_caption_preset=deterministic, seed=2023, epsilon=8, alpha=1, steps=8, num_query=100, num_sub_query=50, sigma=8, probe_policy='Original RGF: clamp probes to RGB range; output projected to clean +/-8', query_counts=counts, records=[], git_vision_cache_enabled=_GIT_CACHE_ENABLED, implementation_preflight_images=6, git_batch_input_ids_fix=True)

    def save_report():
        temp = out / 'results.tmp'
        temp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
        temp.replace(out / 'results.json')
    save_report()
    print('Output:', out, flush=True)
    print('Attack sampling preset:', sampled, flush=True)
    print('Final evaluation preset:', deterministic, flush=True)
    print('Expected attack image queries: 816 per image, 4080 total', flush=True)
    namespace = {'__name__': '__main__', '__file__': str(legacy_path), '_git_i2t': git_i2t, 'load_model_and_preprocess': load_adapter}
    sys.argv = [str(legacy_path), '--model_name', 'git_large_coco', '--model_type', checkpoint, '--batch_size', '1', '--num_samples', '5', '--input_res', '224', '--alpha', '1', '--epsilon', '8', '--steps', '8', '--num_query', '100', '--num_sub_query', '50', '--sigma', '8', '--data_path', str(pilot / 'images'), '--clean_dir', str(pilot / 'clean_reference'), '--text_path', str(mfii_text), '--target_text_path', str(target_text), '--image_output', str(out / 'images'), '--output', str(out / 'best_captions')]
    try:
        exec(compile(adapted, str(legacy_path), 'exec'), namespace)
        assert counts['attack_images'] == 4080, 'Unexpected attack query count'
        phase = 'evaluation_images'
        clip = namespace['clip']
        clip_model = namespace['clip_img_model_vitb32']
        best_text = (out / 'best_captions.txt').read_text(encoding='utf-8').splitlines()
        assert len(best_text) == 5

        def score(text, target):
            with torch.inference_mode():
                feats = clip_model.encode_text(clip.tokenize([text, target]).to('cuda'))
                feats = feats / feats.norm(dim=1, keepdim=True)
                return float((feats[0] * feats[1]).sum())

        def read_image(path):
            with Image.open(path) as im:
                rgb = np.array(im.convert('RGB'), copy=True)
            assert rgb.shape == (224, 224, 3)
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to('cuda').float()
            return (rgb, tensor)
        print('\nSaved-image evaluation: CLIP ViT-B/32 text cosine', flush=True)
        for (i, (r, target)) in enumerate(zip(rows, targets)):
            key = r['image_key']
            (clean_rgb, clean) = read_image(r['clean_reference'])
            (_, mfii) = read_image(r['mfii_image'])
            final_path = out / 'images' / (key + '.png')
            (final_rgb, final) = read_image(final_path)
            diff = np.abs(final_rgb.astype(np.int16) - clean_rgb.astype(np.int16))
            assert int(diff.max()) <= 8, 'Saved MF-tt exceeds total clean-image budget'
            caps = [caption(img, False)[0] for img in [clean, mfii, final]]
            sims = [score(c, target) for c in caps]
            fresh_sampled = caption(final, True)[0]
            record = dict(image_key=key, source_coco_image_id=samples[Path(key).name]['source_coco_image_id'], target_caption=target, mfii_image=r['mfii_image'], mftt_image=str(final_path), max_pixel_difference=int(diff.max()), mean_pixel_difference=float(diff.mean()), clean_caption=caps[0], mfii_caption=caps[1], mftt_caption=caps[2], clean_target_similarity=sims[0], mfii_target_similarity=sims[1], mftt_target_similarity=sims[2], historical_best_sampled_caption=best_text[i], historical_best_sampled_similarity=float(namespace['query_attack_results'][i]), fresh_saved_sampled_caption=fresh_sampled, fresh_saved_sampled_similarity=score(fresh_sampled, target), prior_clean_caption_matches=caps[0] == r['clean_caption'], prior_mfii_caption_matches=caps[1] == r['mfii_caption'])
            report['records'].append(record)
            save_report()
            print(f'{key}: clean={sims[0]:.6f} MF-ii={sims[1]:.6f} MF-tt={sims[2]:.6f}; maxdiff={diff.max()}', flush=True)
            print('Target:', target, '\nMF-tt:', caps[2], flush=True)
        report['status'] = 'complete'
        save_report()
    except BaseException as error:
        report['status'] = 'failed'
        report['error'] = repr(error)
        save_report()
        raise
    print('Query counts:', counts, flush=True)
    print('Results:', out / 'results.json', flush=True)
    print('MFTT_GIT_LARGE_COCO_PILOT_PASS', flush=True)
if __name__ == '__main__':
    main()
