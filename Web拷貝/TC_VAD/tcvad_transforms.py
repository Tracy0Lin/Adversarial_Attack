import random
import torch
import torchvision.transforms.functional as TF
from PIL import Image

# 增加高斯雜訊
def add_gaussian_noise(img, sigma=0.02):
    x = TF.to_tensor(img)
    noise = torch.randn_like(x) * sigma # 產生與原圖形狀相同、標準差為 sigma 的正態分佈雜訊
    x = (x + noise).clamp(0, 1)
    return TF.to_pil_image(x)

# 隨機旋轉角度
def random_rotate(img, max_degree=5):
    angle = random.uniform(-max_degree, max_degree) # 在 -5 到 +5
    return TF.rotate(img, angle)

# 隨機裁剪與縮放
def random_crop_resize(img, crop_scale=(0.9, 0.98), out_size=(384, 384)):
    w, h = img.size
    scale = random.uniform(crop_scale[0], crop_scale[1]) # 決定裁剪掉多少邊緣
    new_w = int(w * scale)
    new_h = int(h * scale)
    if new_w < 10 or new_h < 10:
        return img.resize(out_size)
    i = random.randint(0, h - new_h) if h > new_h else 0
    j = random.randint(0, w - new_w) if w > new_w else 0
    cropped = TF.crop(img, i, j, new_h, new_w)
    return cropped.resize(out_size) # -> 想拉回原本圖片的大小 不要固定

# 隨機重定比例
def random_rescale(img, scale_range=(0.95, 1.05), out_size=(384, 384)):
    scale = random.uniform(scale_range[0], scale_range[1])
    w, h = img.size
    new_w = max(10, int(w * scale))
    new_h = max(10, int(h * scale))
    resized = img.resize((new_w, new_h)) # 縮放至目標大小
    return resized.resize(out_size) # 再強制拉回指定大小

# 隨機抽樣轉換 -> 用以上四種方法中隨機選一個執行
def sample_transform(img, target_size=(384, 384)):
    # 複製一份，避免改到原始物件
    out = img.copy()
    
    # 1. 隨機旋轉 (70% 機率)
    if random.random() > 0.3:
        angle = random.uniform(-5, 5)
        # resample=Image.BICUBIC 是防禦核心，能平滑化對抗擾動
        out = TF.rotate(out, angle, resample=Image.BICUBIC)
    
    # 2. 隨機縮放 (70% 機率)
    if random.random() > 0.3:
        w, h = out.size
        scale = random.uniform(0.9, 1.1) # 縮放比例
        new_w, new_h = int(w * scale), int(h * scale)
        # 第一次縮放改變解析度
        out = out.resize((new_w, new_h), Image.BICUBIC)
    
    # 3. 隨機裁剪 (70% 機率)
    if random.random() > 0.3:
        w, h = out.size
        # 裁剪掉 2%~10% 的邊緣
        crop_scale = random.uniform(0.9, 0.98)
        cw, ch = int(w * crop_scale), int(h * crop_scale)
        i = random.randint(0, h - ch) if h > ch else 0
        j = random.randint(0, w - cw) if w > cw else 0
        out = TF.crop(out, i, j, ch, cw)
    
    # 4. 高斯雜訊 (70% 機率)
    if random.random() > 0.3:
        x = TF.to_tensor(out)
        noise = torch.randn_like(x) * 0.02
        out = TF.to_pil_image((x + noise).clamp(0, 1))

    # [關鍵步驟] 最後強制回歸到模型需要的大小 (384, 384)
    # 這樣不管前面怎麼轉、怎麼裁，最後輸出的 Shape 一定正確且經過高品質插值
    out = out.resize(target_size, Image.BICUBIC)
    
    return out