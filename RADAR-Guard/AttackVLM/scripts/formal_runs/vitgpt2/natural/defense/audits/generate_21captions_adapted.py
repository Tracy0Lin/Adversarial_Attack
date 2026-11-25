import json
import os
import time
from pathlib import Path
import torch
import torchvision.transforms.functional as TF
from PIL import Image
SEED = 20260729
ROTATION_DEGREES = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]
RESCALE_FACTORS = [0.9, 0.92, 0.94, 0.96, 0.98, 1.02, 1.04, 1.06, 1.08, 1.1]
ROTATION_NAMES = [f'rotate_m{abs(degree)}' if degree < 0 else f'rotate_p{degree}' for degree in ROTATION_DEGREES]
RESCALE_NAMES = [f'rescale_{int(round(scale * 100)):03d}' for scale in RESCALE_FACTORS]
TRANSFORM_NAMES = ROTATION_NAMES + RESCALE_NAMES
CAPTION_ORDER = ['base'] + TRANSFORM_NAMES
VALID_EXTS = {'.jpg', '.jpeg', '.png'}

def atomic_json_dump(data, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = Path(str(output_path) + '.tmp')
    with open(temporary_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
    os.replace(temporary_path, output_path)

def build_index(folder):
    return {path.stem: path for path in sorted(folder.iterdir()) if path.is_file() and path.suffix.lower() in VALID_EXTS}

def fixed_rescale(image, scale):
    (width, height) = image.size
    new_width = max(10, int(width * scale))
    new_height = max(10, int(height * scale))
    resized = image.resize((new_width, new_height))
    return resized.resize(OUTPUT_SIZE)

def make_variants(image_path):
    with Image.open(image_path) as opened:
        source_size = list(opened.size)
        source_mode = opened.mode
        source_format = opened.format
        base_image = opened.convert('RGB').resize(OUTPUT_SIZE)
    variants = [('base', base_image)]
    for degree in ROTATION_DEGREES:
        name = f'rotate_m{abs(degree)}' if degree < 0 else f'rotate_p{degree}'
        variants.append((name, TF.rotate(base_image, degree)))
    for scale in RESCALE_FACTORS:
        name = f'rescale_{int(round(scale * 100)):03d}'
        variants.append((name, fixed_rescale(base_image, scale)))
    actual_order = [name for (name, _) in variants]
    if actual_order != CAPTION_ORDER:
        raise RuntimeError(f'轉換順序錯誤：{actual_order}')
    return {'source_size': source_size, 'source_mode': source_mode, 'source_format': source_format, 'variants': variants}

def get_captions(record):
    captions = [record['base_caption']]
    captions.extend((record['transform_captions'][name] for name in TRANSFORM_NAMES))
    return captions

def validate_pair(payload, expected_id):
    try:
        if payload['image_id'] != expected_id:
            return (False, 'image_id 不一致')
        records = payload['records']
        if set(records) != {'clean', 'mf_tt'}:
            return (False, '缺少 clean 或 mf_tt')
        for label in ['clean', 'mf_tt']:
            record = records[label]
            if record['caption_order'] != CAPTION_ORDER:
                return (False, f'{label} caption 順序錯誤')
            if list(record['transform_captions']) != TRANSFORM_NAMES:
                return (False, f'{label} transform 名稱錯誤')
            captions = get_captions(record)
            if len(captions) != 21:
                return (False, f'{label} caption 數量不是 21')
            if any((not str(caption).strip() for caption in captions)):
                return (False, f'{label} 存在空 caption')
            if record['num_captions'] != 21:
                return (False, f'{label} num_captions 錯誤')
            if record['num_transform_captions'] != 20:
                return (False, f'{label} transformed 數量錯誤')
            if record['empty_caption_count'] != 0:
                return (False, f'{label} empty count 錯誤')
        return (True, 'OK')
    except (KeyError, TypeError, ValueError) as error:
        return (False, str(error))

def load_valid_pair(path, expected_id):
    if not path.exists():
        return None
    try:
        with open(path, 'r', encoding='utf-8') as file:
            payload = json.load(file)
    except (OSError, json.JSONDecodeError):
        return None
    (valid, _) = validate_pair(payload, expected_id)
    if not valid:
        return None
    return payload

def generate_caption_with_retry(model, vis_processor, image, device, max_attempts=3):
    caption = ''
    for _ in range(max_attempts):
        tensor = vis_processor(image).unsqueeze(0).to(device)
        generated = model.generate({'image': tensor})
        if generated:
            caption = str(generated[0]).strip()
        if caption:
            return caption
    return caption
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
PER_ID_DIR.mkdir(parents=True, exist_ok=True)
clean_index = build_index(CLEAN_DIR)
mf_tt_index = build_index(MF_TT_DIR)
common_ids = sorted(set(clean_index) & set(mf_tt_index))
clean_only = sorted(set(clean_index) - set(mf_tt_index))
mf_tt_only = sorted(set(mf_tt_index) - set(clean_index))
print('===== 資料檢查 =====', flush=True)
print('Clean：', len(clean_index), flush=True)
print('MF-tt：', len(mf_tt_index), flush=True)
print('成功配對：', len(common_ids), flush=True)
print('只有 Clean：', clean_only, flush=True)
print('只有 MF-tt：', mf_tt_only, flush=True)
if len(common_ids) != 1000:
    raise SystemExit(f'錯誤：應有 1000 組配對，目前為 {len(common_ids)}。')
completed_ids = []
pending_ids = []
for image_id in common_ids:
    pair_file = PER_ID_DIR / f'vitgpt2_21captions_{image_id}.json'
    if load_valid_pair(pair_file, image_id) is not None:
        completed_ids.append(image_id)
    else:
        pending_ids.append(image_id)
print('\n===== 續跑檢查 =====', flush=True)
print('已完成，可跳過：', len(completed_ids), flush=True)
print('本次需要處理：', len(pending_ids), flush=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if pending_ids:
    print('\n===== 載入 ViT-GPT2 =====', flush=True)
    print('Device：', device, flush=True)
    if device == 'cuda':
        print('GPU：', torch.cuda.get_device_name(0), flush=True)
    (model, vis_processors, _) = load_model_and_preprocess(name='vitgpt2', model_type='nlpconnect/vit-gpt2-image-captioning', is_eval=True, device=device)
    model.eval()
    vis_processor = vis_processors['eval']
    started_at = time.time()
    with torch.inference_mode():
        for (pending_number, image_id) in enumerate(pending_ids, start=1):
            pair_started_at = time.time()
            source_paths = {'clean': clean_index[image_id], 'mf_tt': mf_tt_index[image_id]}
            records = {}
            inference_items = []
            for (label, source_path) in source_paths.items():
                image_data = make_variants(source_path)
                records[label] = {'image_id': image_id, 'label': label, 'source_path': str(source_path), 'source_size': image_data['source_size'], 'source_mode': image_data['source_mode'], 'source_format': image_data['source_format'], 'standardized_size': list(OUTPUT_SIZE), 'base_caption': '', 'transform_captions': {}, 'caption_order': CAPTION_ORDER.copy()}
                for (transform_name, image) in image_data['variants']:
                    inference_items.append({'label': label, 'transform_name': transform_name, 'image': image})
            for start in range(0, len(inference_items), BATCH_SIZE):
                batch_items = inference_items[start:start + BATCH_SIZE]
                batch_tensor = torch.stack([vis_processor(item['image']) for item in batch_items]).to(device)
                generated = model.generate({'image': batch_tensor})
                if len(generated) != len(batch_items):
                    raise RuntimeError(f'{image_id}：ViT-GPT2 輸出數量不一致。')
                for (item, generated_caption) in zip(batch_items, generated):
                    caption = str(generated_caption).strip()
                    if not caption:
                        caption = generate_caption_with_retry(model=model, vis_processor=vis_processor, image=item['image'], device=device)
                    label = item['label']
                    transform_name = item['transform_name']
                    if transform_name == 'base':
                        records[label]['base_caption'] = caption
                    else:
                        records[label]['transform_captions'][transform_name] = caption
                del batch_tensor
            for label in ['clean', 'mf_tt']:
                record = records[label]
                captions = get_captions(record)
                record['num_base_captions'] = 1
                record['num_transform_captions'] = len(record['transform_captions'])
                record['num_captions'] = len(captions)
                record['empty_caption_count'] = sum((not str(caption).strip() for caption in captions))
            pair_payload = {'schema_version': '1.0', 'experiment': 'TC-VAD Experiment 1 ViT-GPT2 full caption generation', 'caption_model': {'name': 'vitgpt2', 'model_type': 'nlpconnect/vit-gpt2-image-captioning'}, 'image_id': image_id, 'output_size': list(OUTPUT_SIZE), 'rotation_degrees': ROTATION_DEGREES, 'rescale_factors': RESCALE_FACTORS, 'records': records}
            (valid, reason) = validate_pair(pair_payload, image_id)
            if not valid:
                raise RuntimeError(f'{image_id} 驗證失敗：{reason}')
            pair_file = PER_ID_DIR / f'vitgpt2_21captions_{image_id}.json'
            atomic_json_dump(pair_payload, pair_file)
            elapsed = time.time() - pair_started_at
            total_done = len(completed_ids) + pending_number
            print(f'[{total_done:04d}/{len(common_ids):04d}] {image_id} 完成｜42 captions｜{elapsed:.1f} 秒', flush=True)
    print(f'本次 ViT-GPT2 推論時間：{time.time() - started_at:.1f} 秒', flush=True)
print('\n===== 合併與完整性驗證 =====', flush=True)
combined_records = []
validated_ids = []
for image_id in common_ids:
    pair_file = PER_ID_DIR / f'vitgpt2_21captions_{image_id}.json'
    payload = load_valid_pair(pair_file, image_id)
    if payload is None:
        raise RuntimeError(f'{image_id} 的結果不存在或未通過驗證。')
    validated_ids.append(image_id)
    for label in ['clean', 'mf_tt']:
        combined_records.append(payload['records'][label])
total_captions = 0
total_empty = 0
for record in combined_records:
    captions = get_captions(record)
    total_captions += len(captions)
    total_empty += sum((not str(caption).strip() for caption in captions))
summary = {'num_image_ids': len(validated_ids), 'num_records': len(combined_records), 'num_captions_per_record': 21, 'num_transform_captions_per_record': 20, 'total_captions': total_captions, 'empty_caption_count': total_empty}
if summary != {'num_image_ids': 1000, 'num_records': 2000, 'num_captions_per_record': 21, 'num_transform_captions_per_record': 20, 'total_captions': 42000, 'empty_caption_count': 0}:
    raise RuntimeError(f'整體完整性檢查失敗：{summary}')
combined_payload = {'schema_version': '1.0', 'experiment': 'TC-VAD Experiment 1 ViT-GPT2 all captions', 'caption_model': {'name': 'vitgpt2', 'model_type': 'nlpconnect/vit-gpt2-image-captioning'}, 'output_size': list(OUTPUT_SIZE), 'rotation_degrees': ROTATION_DEGREES, 'rescale_factors': RESCALE_FACTORS, 'caption_order': CAPTION_ORDER, 'paired_image_ids': validated_ids, 'summary': summary, 'records': combined_records}
atomic_json_dump(combined_payload, COMBINED_FILE)
print('Image IDs：', summary['num_image_ids'])
print('影像紀錄：', summary['num_records'])
print('每張 captions：', summary['num_captions_per_record'])
print('Captions 總數：', summary['total_captions'])
print('空 captions：', summary['empty_caption_count'])
print('合併結果：', COMBINED_FILE)
print('\nBATCH_CAPTION_OK')
