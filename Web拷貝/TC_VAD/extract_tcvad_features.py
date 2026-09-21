# 每一張圖片進行壓力測試，並將結果量化

import os
import json
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F

from lavis.models import load_model_and_preprocess
from tcvad_transforms import sample_transform

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ===== 路徑 =====
clean_dir = "data/coco/val2017_100"
adv_dir = "data/coco/val2017_100_adv"
out_file = "data/coco/tcvad_features.json"

N = 8  # 每張圖抽 8 次轉換

# ===== 載入 BLIP =====
model, vis_processors, txt_processors = load_model_and_preprocess(
    name="blip_caption",
    model_type="base_coco",
    is_eval=True,
    device=device
)
model.eval()

import clip # 載入 CLIP 模型：用來計算兩句話的意思像不像
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

import lpips # 載入 LPIPS 模型：用來計算兩張圖人類肉眼看起來差多少
lpips_model = lpips.LPIPS(net='alex').to(device)

def pil_to_np(img):
    return np.array(img).astype(np.float32) / 255.0

def compute_ssim(img1, img2):
    x1 = pil_to_np(img1)
    x2 = pil_to_np(img2)
    return ssim(x1, x2, channel_axis=2, data_range=1.0)

def compute_lpips(img1, img2):
    x1 = to_tensor(img1).unsqueeze(0).to(device) * 2 - 1
    x2 = to_tensor(img2).unsqueeze(0).to(device) * 2 - 1
    with torch.no_grad():
        d = lpips_model(x1, x2)
    return float(d.item())

def compute_clip_image_sim(img1, img2): # 計算 Cosine Similarity，測量兩句話在語義空間的距離
    with torch.no_grad():
        x1 = clip_preprocess(img1).unsqueeze(0).to(device)
        x2 = clip_preprocess(img2).unsqueeze(0).to(device)
        f1 = clip_model.encode_image(x1)
        f2 = clip_model.encode_image(x2)
        f1 = F.normalize(f1, dim=-1)
        f2 = F.normalize(f2, dim=-1)
        sim = (f1 * f2).sum().item()
    return float(sim)

def get_caption(img):
    image = vis_processors["eval"](img).unsqueeze(0).to(device)
    with torch.no_grad():
        cap = model.generate({"image": image})[0]
    return cap

def get_text_embedding(text):
    tokens = clip.tokenize([text]).to(device)
    with torch.no_grad():
        feat = clip_model.encode_text(tokens)
        feat = F.normalize(feat, dim=-1)
    return feat

def text_cosine(t1, t2):
    e1 = get_text_embedding(t1)
    e2 = get_text_embedding(t2)
    return float((e1 * e2).sum().item())

def jaccard_similarity(s1, s2): # 計算 Jaccard 相似度，看兩句話裡面有多少單字是一模一樣的
    a = set(s1.lower().split())
    b = set(s2.lower().split())
    if len(a | b) == 0:
        return 1.0
    return len(a & b) / len(a | b)

def summarize(vals):
    arr = np.array(vals, dtype=np.float32)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }

results = {}

files = sorted(os.listdir(adv_dir))

for idx, fname in enumerate(files, 1):
    print(f"[{idx}/{len(files)}] {fname}")

    clean_img = Image.open(os.path.join(clean_dir, fname)).convert("RGB").resize((384, 384))
    adv_img = Image.open(os.path.join(adv_dir, fname)).convert("RGB").resize((384, 384))

    y0_clean = get_caption(clean_img)
    y0_adv = get_caption(adv_img)

    ssim_list = []
    lpips_list = []
    clipsim_list = []
    jaccard_list = []
    textcos_list = []

    trans_caps = []

    for _ in range(N):
        t_img = sample_transform(adv_img)
        y_i = get_caption(t_img)

        trans_caps.append(y_i)

        ssim_list.append(compute_ssim(adv_img, t_img))
        lpips_list.append(compute_lpips(adv_img, t_img))
        clipsim_list.append(compute_clip_image_sim(adv_img, t_img))
        jaccard_list.append(jaccard_similarity(y0_adv, y_i))
        textcos_list.append(text_cosine(y0_adv, y_i))

    results[fname] = {
        "clean_caption": y0_clean,
        "adv_caption": y0_adv,
        "transformed_captions": trans_caps,
        "Q_ssim": summarize(ssim_list),
        "Q_lpips": summarize(lpips_list),
        "Q_clipsim": summarize(clipsim_list),
        "S_jaccard": summarize(jaccard_list),
        "S_textcos": summarize(textcos_list),
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Saved features to {out_file}")