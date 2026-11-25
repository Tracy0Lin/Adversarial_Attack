"""Adapt the supplied BLIP MF-ii objective to the ViT-GPT2 encoder.

Uses normalized CLS cosine similarity, zero initialization, sign ascent,
and one total pixel-space budget. Like the supplied script, intermediate
updates constrain delta; the RGB image is clipped when saved. Captions
use a separate deterministic diagnostic preset, not the formal MF-tt preset.
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
import numpy as np
from PIL import Image

def image_index(root):
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f'Missing image directory: {root}')
    result = {}
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
            key = path.relative_to(root).with_suffix('').as_posix()
            if key in result:
                raise ValueError(f'Duplicate image key: {key}')
            result[key] = path
    return result

def paired_paths(source, target, count):
    (left, right) = (image_index(source), image_index(target))
    if set(left) != set(right):
        raise ValueError(f'Pair mismatch: missing targets={sorted(set(left) - set(right))[:5]}, missing sources={sorted(set(right) - set(left))[:5]}')
    if not 1 <= count <= len(left):
        raise ValueError(f'Requested {count} pairs; available={len(left)}')
    return [(key, left[key], right[key]) for key in sorted(left)[:count]]

def save_checked_image(clean_uint8, adv_float, path, epsilon):
    adv_uint8 = np.rint(np.clip(adv_float, 0, 1) * 255).astype(np.uint8)
    difference = np.abs(adv_uint8.astype(np.int16) - clean_uint8.astype(np.int16))
    max_diff = int(difference.max())
    if max_diff > epsilon:
        raise ValueError(f'Saved image exceeds budget: {max_diff} > {epsilon}')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(adv_uint8.transpose(1, 2, 0)).save(path)
    with Image.open(path) as opened:
        loaded = np.asarray(opened.convert('RGB')).transpose(2, 0, 1)
    if not np.array_equal(adv_uint8, loaded):
        raise ValueError('PNG round-trip changed pixels')
    return (adv_uint8, max_diff, float(difference.mean()))

def _verify_resume(saved, current, pairs, out):
    fields = ['implementation_sha256', 'checkpoint', 'revision', 'args', 'torch', 'transformers', 'image_mean', 'image_std', 'processor', 'caption_diagnostic_preset', 'objective']
    for field in fields:
        assert saved.get(field) == current.get(field), 'Resume setting changed: ' + field
    records = saved['records']
    assert isinstance(records, list) and len(records) <= len(pairs)
    for (record, (key, source, target)) in zip(records, pairs):
        assert record['image_key'] == key, 'Resume IDs are not an ordered prefix'
        assert Path(record['source']).resolve() == source.resolve()
        assert Path(record['target']).resolve() == target.resolve()
        for (field, path) in [('source_sha256', source), ('target_sha256', target)]:
            assert hashlib.sha256(path.read_bytes()).hexdigest() == record[field], 'Resume input changed: ' + str(path)
        for (field, folder) in [('clean_reference', 'clean_reference'), ('mfii_image', 'images')]:
            path = out / folder / (key + '.png')
            assert Path(record[field]).resolve() == path.resolve()
            assert hashlib.sha256(path.read_bytes()).hexdigest() == record[field + '_sha256'], 'Resume PNG changed: ' + str(path)
        assert record['max_pixel_difference'] <= current['args']['epsilon']
    print('RESUME_VERIFIED:', len(records), '/', len(pairs), flush=True)
    return records

def main():
    import torch
    import torch.nn.functional as F
    import transformers
    from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
    project = Path('/workspace/projects/couser/AttackVLM_Official_Reproduction')
    data = project / 'formal_tests/paper_coco1000_v1/data'
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', default=str(data / 'source_images'))
    parser.add_argument('--target_dir', default=str(data / 'target_images_sd14_1000_paired_v1'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--epsilon', type=int, default=8)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=2023)
    args = parser.parse_args()
    if args.steps < 1 or not 1 <= args.epsilon <= 255 or args.alpha <= 0:
        parser.error('Require steps >= 1, epsilon in [1,255], alpha > 0')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    pairs = paired_paths(args.source_dir, args.target_dir, args.num_samples)
    out = Path(args.output)
    saved = None
    if out.exists():
        assert (out / 'results.json').is_file(), 'Existing output has no checkpoint report'
        saved = json.loads((out / 'results.json').read_text(encoding='utf-8'))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    checkpoint = 'nlpconnect/vit-gpt2-image-captioning'
    print('Loading:', checkpoint, flush=True)
    model = VisionEncoderDecoderModel.from_pretrained(checkpoint, revision='dc68f91c06a1ba6f15268e5b9c13ae7a7c514084').eval().to('cuda')
    revision = getattr(model.config, '_commit_hash', None)
    processor = ViTImageProcessor.from_pretrained(checkpoint, revision=revision)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, revision=revision)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if not processor.do_normalize:
        raise ValueError('Expected a normalizing ViT processor')
    mean = torch.tensor(processor.image_mean, device='cuda').view(1, 3, 1, 1)
    std = torch.tensor(processor.image_std, device='cuda').view(1, 3, 1, 1)
    if not bool((std > 0).all()):
        raise ValueError('Invalid normalization std')

    def normalize(raw):
        return (raw - mean) / std

    def canonical(path):
        with Image.open(path) as opened:
            image = opened.convert('RGB')
        uint8 = processor(images=image, do_rescale=False, do_normalize=False, return_tensors='pt').pixel_values
        if uint8.dtype != torch.uint8:
            raise ValueError(f'Expected unscaled uint8 processor output, got {uint8.dtype}')
        raw = uint8.to(device='cuda', dtype=torch.float32) / 255.0
        expected = processor(images=image, return_tensors='pt').pixel_values.to('cuda')
        if not torch.allclose(normalize(raw), expected, atol=1e-05, rtol=1e-05):
            raise ValueError('Differentiable normalization does not match processor')
        return (raw, uint8[0].cpu().numpy())

    def cls_features(raw):
        tokens = model.encoder(pixel_values=normalize(raw), return_dict=True).last_hidden_state
        return F.normalize(tokens[:, 0, :], dim=1)
    generation = dict(do_sample=False, num_beams=4, max_new_tokens=32)

    def caption(raw):
        with torch.no_grad():
            ids = model.generate(pixel_values=normalize(raw), **generation)
        text = tokenizer.batch_decode(ids, skip_special_tokens=True)[0].strip()
        if not text:
            raise ValueError('Empty caption')
        return text
    out.mkdir(parents=True, exist_ok=True)
    results = dict(status='running', checkpoint=checkpoint, revision=revision, python=sys.version, executable=sys.executable, torch=torch.__version__, transformers=transformers.__version__, args=vars(args), image_mean=processor.image_mean, image_std=processor.image_std, processor=processor.to_dict(), caption_diagnostic_preset=generation, objective='normalized CLS cosine; sign ascent', records=[])
    results['implementation_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if saved is not None:
        results['records'] = _verify_resume(saved, results, pairs, out)
    completed_on_start = len(results['records'])

    def checkpoint_results():
        if results['records']:
            record = results['records'][-1]
            for field in ['clean_reference', 'mfii_image']:
                record[field + '_sha256'] = hashlib.sha256(Path(record[field]).read_bytes()).hexdigest()
        temporary = out / 'results.tmp'
        temporary.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
        temporary.replace(out / 'results.json')
    checkpoint_results()
    print('Output:', out, flush=True)
    for (number, (key, source_path, target_path)) in enumerate(pairs, 1):
        if number <= completed_on_start:
            continue
        (clean, clean_uint8) = canonical(source_path)
        (target, _) = canonical(target_path)
        with torch.no_grad():
            target_feature = cls_features(target)
            before = float((cls_features(clean) * target_feature).sum(dim=1).mean())
        clean_caption = caption(clean)
        if number == 1:
            if clean_caption != caption(clean):
                raise ValueError('Repeated deterministic caption mismatch')
            print('Input shape:', tuple(clean.shape), 'mean/std:', processor.image_mean, processor.image_std, 'MODEL_PREFLIGHT_PASS', flush=True)
        delta = torch.zeros_like(clean, requires_grad=True)
        first_grad_max = None
        for step in range(args.steps):
            score = (cls_features(clean + delta) * target_feature).sum(dim=1).mean()
            grad = torch.autograd.grad(score, delta)[0]
            if not bool(torch.isfinite(score)) or not bool(torch.isfinite(grad).all()):
                raise ValueError('Non-finite attack score/gradient')
            if step == 0:
                first_grad_max = float(grad.abs().max())
                if first_grad_max == 0:
                    raise ValueError('Zero input gradient')
            delta = (delta + args.alpha / 255.0 * grad.sign()).clamp(-args.epsilon / 255.0, args.epsilon / 255.0).detach().requires_grad_(True)
            if step == 0 or (step + 1) % 25 == 0 or step + 1 == args.steps:
                print(f'[{number}/{len(pairs)}] {key} step={step + 1}/{args.steps} CLS similarity={float(score):.6f}', flush=True)
        clean_path = out / 'clean_reference' / (key + '.png')
        adv_path = out / 'images' / (key + '.png')
        save_checked_image(clean_uint8, clean[0].cpu().numpy(), clean_path, 0)
        (adv_uint8, maximum, average) = save_checked_image(clean_uint8, (clean + delta).detach()[0].cpu().numpy(), adv_path, args.epsilon)
        saved_adv = torch.from_numpy(adv_uint8.copy()).unsqueeze(0).to('cuda').float() / 255.0
        with torch.no_grad():
            after = float((cls_features(saved_adv) * target_feature).sum(dim=1).mean())
        adv_caption = caption(saved_adv)
        results['records'].append(dict(image_key=key, source=str(source_path), target=str(target_path), source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(), target_sha256=hashlib.sha256(target_path.read_bytes()).hexdigest(), clean_reference=str(clean_path), mfii_image=str(adv_path), input_shape=list(clean.shape), first_gradient_max=first_grad_max, cls_similarity_before=before, cls_similarity_after_saved=after, max_pixel_difference=maximum, mean_pixel_difference=average, clean_caption=clean_caption, mfii_caption=adv_caption))
        checkpoint_results()
        print(f'{key}: CLS {before:.6f} -> {after:.6f}; max pixel diff={maximum}', flush=True)
        print('Clean:', clean_caption, flush=True)
        print('MF-ii:', adv_caption, flush=True)
    results['status'] = 'complete'
    checkpoint_results()
    print('Results:', out / 'results.json', flush=True)
    print('MFII_VITGPT2_FORMAL1000_PASS', flush=True)
if __name__ == '__main__':
    main()
