import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import sys
import os
import cv2
import numpy as np

from transformers import BertForMaskedLM, BertTokenizer
from transformers.models.bert.modeling_bert import BertConfig
config_atk = BertConfig.from_pretrained('bert-base-uncased')
import copy
import torch
import torch.nn
import torch.nn.functional as F
import random
import json
import utils
import torch.optim
from filter_words import filter_words
import nltk
from nltk.corpus import stopwords
from torchvision.utils import save_image
from skimage.metrics import structural_similarity as compare_ssim
import cleverhans.torch.attacks.BLIP.projected_gradient_descent as pgd
import torchvision.transforms.functional as TF
import lpips
import gc
import clip
import hashlib

from attack.VQA.attack_rate_summary import save_attack_rate_summary

tf.config.set_visible_devices([], 'GPU')
nltk.download('stopwords')
filter_words = filter_words + stopwords.words('english')+['?','.']

class Feature(object):
    def __init__(self, seq_a):
        self.seq = seq_a
        self.final_adverse = seq_a
        self.query = 0
        self.change = 0
        self.success = 0
        self.sim = 0.0
        self.changes = []

def calculate_batch_ssim(img_tensor_1, img_tensor_2):
    device = img_tensor_1.device
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    # 還原去正規化 -> 變回 0~1 的數值
    img1 = img_tensor_1 * std + mean
    img2 = img_tensor_2 * std + mean

    # 轉成 Numpy 格式 [H, W, C]
    img1 = img1.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    img2 = img2.squeeze().permute(1, 2, 0).detach().cpu().numpy()

    # 確保數值在 0-1 之間
    img1 = np.clip(img1, 0, 1)
    img2 = np.clip(img2, 0, 1)

    # win_size 設為 3 對像素級的微小變動較敏感，channel_axis=-1 指定最後一維是顏色
    ssim_value = compare_ssim(img1, img2, data_range=1.0, win_size=3, channel_axis=-1)
    return ssim_value

def save_adv_image(img_tensor, path):
    mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device).view(1, 3, 1, 1)
    img = img_tensor * std + mean
    save_image(img, path)

def input_defense(images, min_scale=0.9, max_scale=1.0): # 隨機縮放 (Random Resize)
    if isinstance(images, list):
        images = images[0]
    
    # 獲取原始圖片的高度與寬度
    orig_h = images.shape[-2]
    orig_w = images.shape[-1]
    
    # 產生隨機比例
    rnd = random.uniform(min_scale, max_scale) 
    new_h = int(orig_h * rnd)
    new_w = int(orig_w * rnd)
    
    # 先縮小 (破壞攻擊結構)
    images_resized = F.interpolate(images, size=(new_h, new_w), mode='bilinear', align_corners=False)
    
    # 再放大回原尺寸 (讓模型能讀取)
    images_restored = F.interpolate(images_resized, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
    
    return images_restored

def resize_gaussian_defense(images, min_scale=0.9, max_scale=1.0, kernel=3):
    if isinstance(images, list):
        images = images[0]

    orig_h = images.shape[-2]
    orig_w = images.shape[-1]

    rnd = random.uniform(min_scale, max_scale)

    new_h = int(orig_h * rnd)
    new_w = int(orig_w * rnd)

    # Step1 resize
    images_resized = F.interpolate(
        images,
        size=(new_h, new_w),
        mode='bilinear',
        align_corners=False
    )

    # Step2 resize back
    images_restored = F.interpolate(
        images_resized,
        size=(orig_h, orig_w),
        mode='bilinear',
        align_corners=False
    )

    # Step3 gaussian blur
    images_blur = TF.gaussian_blur(images_restored, kernel_size=[kernel, kernel])

    return images_blur

def apply_balanced_sharpning(img_tensor, amount=1.2, threshold=0.05):
    # 1. 取得細節層 (High-pass)
    # 使用一個稍微大一點的 kernel 來定義什麼是「大尺度邊緣」
    base = TF.gaussian_blur(img_tensor, kernel_size=[5, 5], sigma=[1.0, 1.0])
    details = img_tensor - base
    
    # 2. 邊緣遮罩 (Edge-only Mask)
    # 只有明顯的邊緣才執行銳化，避開平滑區（平滑區最容易藏對抗噪點）
    mask = torch.abs(details) > threshold
    
    # 3. 執行補償
    # 我們只把細節加回「原本防禦後的圖」
    sharpened = img_tensor + (details * amount * mask.float())
    
    return torch.clamp(sharpened, -1.0, 1.0)

class Adv_attack:
    def __init__(self, vqa_model, pretrain_model, tokenizer, device, correct_idx_list, correct_pred_list, USE_model, output_dir="output"):
        self.device = device
        self.attack_dict = {}
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        self.processed_image_hashes = set()

        # 雙重計分板：分別記錄兩種防禦強度的結果
        self.acc_list_mild = []   # 輕度防禦 (0.9~1.0)
        self.acc_list_strong = [] # 重度防禦 (0.5~0.7)
        self.acc_list_nodefense = [] # 無防禦統計
        self.ssim_list = []
        self.ssim_resize = []
        self.ssim_gauss = []
        self.ssim_rg = []

        # === 1. 初始化 LPIPS (使用 VGG 作為後端) ===
        # net='vgg' 較準確，'alex' 較快
        print("Loading LPIPS VGG weights...")
        self.lpips_fn = lpips.LPIPS(net='vgg').to(self.device)
        self.lpips_list = [] # 儲存 Attack vs Original 的 LPIPS
        self.lpips_rg_list = [] # 儲存 Defense vs Attack 的 LPIPS
        
        # 紀錄清單
        self.acc_list_nodefense = []
        self.ssim_list = []

        self.tokenizer = tokenizer
        self.single_stg_step = 20
        self.total_stg_step = 40
        self.alpha = 0.01
        self.tokenizer_mlm = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
        self.text_bank = {}
        self.adv_txt_dict = {}
        self.cos_sim = 0.95
        self.k = 10
        self.text_budget = 100000
        self.correct_list = correct_idx_list
        self.blip_ans_table = correct_pred_list
        self.white_model = pretrain_model
        self.black_model = vqa_model
        self.USE_model = USE_model
        self.device = device
        self.batch = None
        self.captions = None
        self.vqa_score = 0
        self.mlm_model = BertForMaskedLM.from_pretrained('bert-base-uncased', config=config_atk).to(self.device)
        
        self.output_dir = output_dir
        self.max_samples = 200
        os.makedirs(self.output_dir, exist_ok=True)
        self.log_path = os.path.join(output_dir, 'attack_dual_defense_log.json')
        if os.path.exists(self.log_path):
            with open(self.log_path, 'r', encoding='utf-8') as f:
                try:
                    self.logs = json.load(f)
                    print(f"Loaded existing log with {len(self.logs)} entries.")
                except:
                    self.logs = []
                    print("Existing log empty or invalid; starting fresh.")
        else:
            self.logs = []
        
        # 建立一個「已完成 ID」的清單，用來快速查詢
        self.finished_ids = [item['question_id'] for item in self.logs]

        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)

    def calculate_clip_similarity(self, img_tensor_1, img_tensor_2):
        """計算兩張圖片在 CLIP 空間中的餘弦相似度 (Semantic Level)"""
        # 1. 還原正規化並轉為 CLIP 預期輸入
        # (這裡需確保 Tensor 轉為 0~1 並 resize 到 224x224)
        with torch.no_grad():
            feat1 = self.clip_model.encode_image(F.interpolate(img_tensor_1, (224,224)))
            feat2 = self.clip_model.encode_image(F.interpolate(img_tensor_2, (224,224)))
            
            feat1 /= feat1.norm(dim=-1, keepdim=True)
            feat2 /= feat2.norm(dim=-1, keepdim=True)
            
            similarity = (feat1 * feat2).sum(dim=-1)
        return similarity.item()
    
    # 2. 將此函數移入 class 內，並補全數值轉換邏輯
    def calculate_lpips(self, img_tensor_1, img_tensor_2):
        """
        img_tensor: [1, 3, H, W], 原始範圍通常是 BLIP normalized
        LPIPS 預期範圍為 [-1, 1]
        """
        # 定義 Mean 和 Std (BLIP/ImageNet 標準)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        # 先還原到 [0, 1]
        img1 = img_tensor_1 * std + mean
        img2 = img_tensor_2 * std + mean

        # 再轉換到 [-1, 1] -> 公式：(x - 0.5) * 2
        img1 = (img1 - 0.5) * 2
        img2 = (img2 - 0.5) * 2

        # 確保數值不溢出
        img1 = torch.clamp(img1, -1, 1)
        img2 = torch.clamp(img2, -1, 1)

        with torch.no_grad():
            dist = self.lpips_fn(img1, img2)
        return dist.item()

    def Gen_ori_feats(self, batch):
        image = batch['image'].to(self.device, non_blocking=True)
        img_feats_list, txt_feats_list = self.white_model.Gen_feats(image, batch['question'][0])
        img_feats = torch.cat(img_feats_list, axis=0)
        txt_feats = torch.cat(txt_feats_list, axis=0)
        return img_feats, txt_feats

    def pgd_attack(self, x):
        if random.random() > 0.5: # 50% 機率進行變換
            orig_size = x.shape[-2:]
            rnd = random.randint(int(orig_size[0] * 0.9), orig_size[0])
            x = F.interpolate(x, size=(rnd, rnd), mode='bilinear')
            pad_top = random.randint(0, orig_size[0] - rnd)
            pad_left = random.randint(0, orig_size[1] - rnd)
            x = F.pad(x, (pad_left, orig_size[1] - rnd - pad_left, pad_top, orig_size[0] - rnd - pad_top))

        img_feats_list, txt_feats_list = self.white_model.Gen_feats(x, self.batch['question'])
        img_feats = torch.cat(img_feats_list, axis=0)
        txt_feats = torch.cat(txt_feats_list, axis=0)
        return [txt_feats, img_feats]

    def _tokenize(self, seq, tokenizer):
        seq = seq.replace('\n', '').lower()
        words = seq.split(' ')
        sub_words = []
        keys = []
        index = 0
        for word in words:
            sub = tokenizer.tokenize(word)
            sub_words += sub
            keys.append([index, index + len(sub)])
            index += len(sub)
        return words, sub_words, keys

    def get_bpe_substitues(self, substitutes, tokenizer, mlm_model):
        substitutes = substitutes[0:12, 0:4]
        all_substitutes = []
        for i in range(substitutes.size(0)):
            if len(all_substitutes) == 0:
                lev_i = substitutes[i]
                all_substitutes = [[int(c)] for c in lev_i]
            else:
                lev_i = []
                for all_sub in all_substitutes:
                    for j in substitutes[i]:
                        lev_i.append(all_sub + [int(j)])
                all_substitutes = lev_i
        c_loss = torch.nn.CrossEntropyLoss(reduction='none')
        all_substitutes = torch.tensor(all_substitutes)
        all_substitutes = all_substitutes[:24].to(self.device)
        N, L = all_substitutes.size()
        word_predictions = mlm_model(all_substitutes)[0]
        ppl = c_loss(word_predictions.view(N * L, -1), all_substitutes.view(-1))
        ppl = torch.exp(torch.mean(ppl.view(N, L), dim=-1))
        _, word_list = torch.sort(ppl)
        word_list = [all_substitutes[i] for i in word_list]
        final_words = []
        for word in word_list:
            tokens = [tokenizer._convert_id_to_token(int(i)) for i in word]
            text = tokenizer.convert_tokens_to_string(tokens)
            final_words.append(text)
        return final_words

    def get_substitues(self, substitutes, tokenizer, mlm_model, substitutes_score=None, use_bpe=True, threshold=0.3):
        words = []
        sub_len, k = substitutes.size()
        if sub_len == 0:
            return words
        elif sub_len == 1:
            for (i, j) in zip(substitutes[0], substitutes_score[0]):
                if threshold != 0 and j < threshold:
                    break
                words.append(tokenizer._convert_id_to_token(int(i)))
        else:
            if use_bpe == 1:
                words = self.get_bpe_substitues(substitutes, tokenizer, mlm_model)
            else:
                return words
        return words

    def get_important_scores(self, words, batch, tgt_pos, score, image):
        masked_words = self._get_masked(words)
        texts = [' '.join(words) for words in masked_words]
        important_scores = []
        for mlm in texts:
            _, topk_ids, topk_probs = self.black_model(image, mlm, self.answer_candidates, train=False, inference='rank', k_test=128)
            if tgt_pos not in list(topk_ids[0].cpu().numpy()):
                important_scores.append((torch.tensor(-10000).to(self.device)).data.cpu().numpy())
            else:
                im_value = topk_probs[0][torch.where(topk_ids[0] == tgt_pos)][0]
                important_scores.append((im_value - score).data.cpu().numpy())
        return np.array(important_scores)

    def bert_attack(self, batch, tgt_pos, score, gth):
        ori_text = batch['question'][0]
        image = batch['image']
        image = image.to(self.device, non_blocking=True)
        text = ori_text.lower()
        feature = Feature(text)
        tokenizer = self.tokenizer_mlm
        words, sub_words, keys = self._tokenize(feature.seq, tokenizer)
        max_length = 512
        inputs = tokenizer.encode_plus(feature.seq, None, add_special_tokens=True, max_length=max_length, truncation=True)
        sub_words = ['[CLS]'] + sub_words[:2] + sub_words[2:max_length - 2] + ['[SEP]']
        input_ids_ = torch.tensor([tokenizer.convert_tokens_to_ids(sub_words)])
        word_predictions = self.mlm_model(input_ids_.to(self.device))[0].squeeze()
        word_pred_scores_all, word_predictions = torch.topk(word_predictions, self.k, -1)
        word_predictions = word_predictions[1:len(sub_words) + 1, :]
        word_pred_scores_all = word_pred_scores_all[1:len(sub_words) + 1, :]
        important_scores = self.get_important_scores(words, batch, tgt_pos, score, image)
        feature.query += int(len(words))
        list_of_index = sorted(enumerate(important_scores), key=lambda x: x[1], reverse=False)
        final_words = copy.deepcopy(words)
        success = 0
        text_bank = []
        sim_list = []
        for ii, top_index in enumerate(list_of_index):
            if feature.change >= self.text_budget:
                feature.success = 1
                break
            tgt_word = words[top_index[0]]
            if tgt_word in filter_words:
                continue
            if keys[top_index[0]][0] > max_length - 2:
                continue
            substitutes = word_predictions[keys[top_index[0]][0]:keys[top_index[0]][1]]
            word_pred_scores = word_pred_scores_all[keys[top_index[0]][0]:keys[top_index[0]][1]]
            substitutes = self.get_substitues(substitutes, tokenizer, self.mlm_model, substitutes_score=word_pred_scores)
            for substitute in substitutes:
                if substitute == tgt_word or '##' in substitute or substitute in filter_words:
                    continue
                temp_replace = copy.deepcopy(final_words)
                temp_replace[top_index[0]] = substitute
                temp_text = tokenizer.convert_tokens_to_string(temp_replace)
                embs = self.USE_model([ori_text, temp_text]).numpy()
                norm = np.linalg.norm(embs, axis=1)
                embs = embs / norm[:, None]
                sim = (embs[:1] * embs[1:]).sum(axis=1)[0]
                if sim > self.cos_sim:
                    sim_list.append(sim)
                    text_bank.append(temp_text)
                    answer_ids, topk_ids, topk_probs, = self.black_model(image, temp_text, self.answer_candidates, train=False, inference='rank', k_test=128)
                    result = []
                    for ques_id, answer_id in zip(batch['question_id'], answer_ids):
                        result.append({"question_id": int(ques_id.item()), "answer": self.answer_list[answer_id]})
                    ans_after_attack = result[0]['answer']
                    if ans_after_attack != gth:
                        success = 1
                        return text_bank, success, sim_list
        text_cand = []
        if len(text_bank) != 0:
            sim_list_sort = copy.deepcopy(sim_list)
            for i in range(len(sim_list_sort)):
                si = sim_list_sort.index(max(sim_list_sort))
                text_cand.append(text_bank[si])
                sim_list_sort[si] = -1e8
        return text_cand, success, sim_list

    def _get_masked(self, words):
        len_text = max(len(words), 2)
        masked_words = []
        for i in range(len_text):
            masked_words.append(words[0:i] + ['[MASK]'] + words[i + 1:])
        return masked_words

    def black_box_predict(self, image, text):
        answer_ids, topk_ids, topk_probs, = self.black_model(image, text, self.answer_candidates, train=False, inference='rank', k_test=128)
        out_v = []
        for answer_id in answer_ids:
            out_v.append({"answer": self.answer_list[answer_id]})
        return out_v[0]['answer']

    @torch.no_grad()
    def evaluate(self, data_loader, tokenizer):
        answer_list = data_loader.dataset.answer_list
        self.answer_list = answer_list
        answer_candidates = self.black_model.tokenizer(answer_list, padding='longest', return_tensors='pt').to(self.device)
        self.answer_candidates = answer_candidates
        answer_candidates.input_ids[:, 0] = self.black_model.tokenizer.bos_token_id
        self.tokeizer = tokenizer
        self.white_model.eval()
        self.black_model.eval()
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = "Test:"
        print_freq = 50
        search_scales = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5]
        
        self.processed_image_hashes = set()

        for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            if int(batch['question_id'][0]) in self.finished_ids:
                continue
            
            if len(self.logs) >= self.max_samples:
                break

            attack_type = "none"
            if int(batch['question_id'][0]) in self.finished_ids:
                continue
            # if len(self.acc_list_mild) >= 5000:
            #    break
            if int(batch['question_id'][0]) not in self.correct_list:
                continue
                
            self.batch = copy.deepcopy(batch)
            ori_img = batch['image'].to(self.device, non_blocking=True)
            correct_ans = self.blip_ans_table[str(int(batch['question_id'][0]))]
            
            # 原始預測
            pred_ans = self.black_box_predict(ori_img, batch['question'][0])
            if pred_ans != correct_ans:
                continue

            ori_img_feats, ori_txt_feats = self.Gen_ori_feats(batch)
            adv_img = copy.deepcopy(ori_img)
            
            # --- 🚀 步驟 1：生成攻擊 (Image Attack) ---
            torch.set_grad_enabled(True)
            adv_img, loss = pgd.projected_gradient_descent(self.pgd_attack, adv_img, 0.125, self.alpha, self.total_stg_step,
                                                         np.inf, clip_min=-1.0, clip_max=1.0,
                                                         y=[ori_txt_feats, ori_img_feats, None, None, None],
                                                         time=0, ori_x=ori_img, method='BSA')
            torch.set_grad_enabled(False)
            
            # 暫存最終攻擊樣本 (會隨著文字攻擊更新)
            final_adv_img = adv_img
            final_adv_text = batch['question'][0]
            
            # 檢查圖片攻擊是否成功
            temp_pred = self.black_box_predict(final_adv_img, final_adv_text)
            if temp_pred != correct_ans:
                attack_type = "image"
            else:
                attack_type = "failed"
            
            """if temp_pred == correct_ans:
                # === 🚀 步驟 2：文字攻擊 (Text Attack) ===
                # (圖片沒騙過，嘗試改文字)
                answer_ids, topk_ids, topk_probs = self.black_model(ori_img, batch['question'][0], answer_candidates, train=False, inference='rank', k_test=128)
                result = []
                for ques_id, answer_id in zip(batch['question_id'], answer_ids):
                    result.append({"question_id": int(ques_id.item()), "answer": answer_list[answer_id]})
                gth = result[0]['answer']
                score, pred = topk_probs[0].max(dim=0)
                tgt_pos = topk_ids[0][pred]
                
                adv_text, success, sim = self.bert_attack(batch, tgt_pos, score, gth)
                
                if success == 1:
                    attack_type = "text"
                    # 文字攻擊成功
                    final_adv_text = adv_text[0] # 取第一個成功樣本
                    # 圖片恢復原狀 (純文字攻擊通常不改圖)
                    final_adv_img = ori_img 
                else:
                    # === 🚀 步驟 3：混合攻擊 (Hybrid Loop) ===
                    text_bank = adv_text[:-1]
                    if len(text_bank) > self.single_stg_step:
                        text_bank = text_bank[:self.single_stg_step]
                    if len(text_bank) == 0:
                        text_bank = text_bank + [batch['question'][0]]
                    
                    iters = int((self.total_stg_step - self.single_stg_step) / len(text_bank))
                    iters_list = [iters for _ in range(len(text_bank))]
                    iters_list[-1] += self.single_stg_step - sum(iters_list)

                    idx = 0
                    max_len = len(text_bank) - 1
                    
                    while True:
                        if idx > max_len:
                            break
                        else:
                            idx += 1
                        
                        self.batch['question'] = text_bank[idx - 1]
                        
                        # 檢查當前組合是否成功 (不開防禦，這是為了生成攻擊)
                        out_v_ans = self.black_box_predict(final_adv_img, text_bank[idx - 1])
                        
                        if out_v_ans != correct_ans:
                            attack_type = "hybrid"
                            final_adv_text = text_bank[idx - 1]
                            break
                        
                        # 繼續 PGD 圖片微調
                        torch.set_grad_enabled(True)
                        final_adv_img, _ = pgd.projected_gradient_descent(self.pgd_attack, final_adv_img, 0.125, 0.01, iters,
                                                                 np.inf, clip_min=-1.0, clip_max=1.0,
                                                                 y=[ori_txt_feats, ori_img_feats, None, None, None],
                                                                 time=1, ori_x=ori_img, method='BSA')
                        torch.set_grad_enabled(False)

                        out_v_ans = self.black_box_predict(final_adv_img, text_bank[idx-1])
                        final_adv_text = text_bank[idx-1] # 更新當前使用的文字
                        
                        if out_v_ans != correct_ans:
                            break"""
            clip_sim = self.calculate_clip_similarity(ori_img, final_adv_img)
            # if i < 20: # 限制只存前 20 張，避免硬碟爆炸 (想全存就把這行拿掉)
            img_filename = f"adv_{int(batch['question_id'][0])}.png"
            save_path = os.path.join(self.output_dir, 'result', img_filename)
            
            # 確保 result 資料夾存在
            os.makedirs(os.path.join(self.output_dir, 'result'), exist_ok=True)
            
            # 呼叫存檔函數 (存的是 final_adv_img，這是最強的攻擊樣本)
            save_adv_image(final_adv_img, save_path)

            
            # === 🔥🔥🔥 最終雙重防禦驗證 (Dual Defense Validation) 🔥🔥🔥 ===
            # 現在我們有了最強的攻擊組合 (final_adv_img, final_adv_text)
            # 我們來測試兩種防禦是否能擋住它
            # 紀錄【無防禦】結果
            no_defense_pred = self.black_box_predict(final_adv_img, final_adv_text)
            if no_defense_pred != correct_ans:
                self.acc_list_nodefense.append(1) # 攻擊成功
                nodefense_success = True
            else:
                self.acc_list_nodefense.append(0)
                nodefense_success = False

            # ==========================================
            # 🛡️ 實驗 A：高斯去雜訊防禦 (Gaussian Blur)
            # ==========================================

            # ==========================================
            # 🛡️ Gaussian defense
            # ==========================================

            # Mild Gaussian
            # 適合抹除微小雜訊，保留大部分圖片細節
            blur_img_mild = TF.gaussian_blur(final_adv_img, kernel_size=[3,3])
            gauss_mild_pred = self.black_box_predict(blur_img_mild, final_adv_text)
            # Mild Gaussian + Sharpen (加銳化)
            blur_img_mild_s =apply_balanced_sharpning(blur_img_mild)
            gauss_mild_s_pred = self.black_box_predict(blur_img_mild_s, final_adv_text)

            if gauss_mild_pred != correct_ans:
                gauss_mild_success = True
            else:
                gauss_mild_success = False


            # Strong Gaussian
            # 防禦力更強，但圖片會變得比較柔焦
            blur_img_strong = TF.gaussian_blur(final_adv_img, kernel_size=[5,5])
            gauss_strong_pred = self.black_box_predict(blur_img_strong, final_adv_text)
            # Strong Gaussian + Sharpen (加銳化)
            blur_img_strong_s = apply_balanced_sharpning(blur_img_strong)
            gauss_strong_s_pred = self.black_box_predict(blur_img_strong_s, final_adv_text)

            if gauss_strong_pred != correct_ans:
                gauss_strong_success = True
            else:
                gauss_strong_success = False

            # 🛡️ Resize defense (mild)
            adv_img_mild = input_defense(final_adv_img, 0.9, 1.0)
            pred_mild = self.black_box_predict(adv_img_mild, final_adv_text)
            # Resize Mild + Sharpen (加銳化)
            adv_img_mild_s = apply_balanced_sharpning(adv_img_mild)
            pred_mild_s = self.black_box_predict(adv_img_mild_s, final_adv_text)

            if pred_mild != correct_ans:
                self.acc_list_mild.append(1)
                mild_success = True
            else:
                self.acc_list_mild.append(0)
                mild_success = False


            # 🛡️ Resize defense (strong)
            adv_img_strong = input_defense(final_adv_img, 0.5, 0.7)
            pred_strong = self.black_box_predict(adv_img_strong, final_adv_text)
            # Resize Strong + Sharpen (加銳化)
            adv_img_strong_s = apply_balanced_sharpning(adv_img_strong)
            pred_strong_s = self.black_box_predict(adv_img_strong_s, final_adv_text)

            if pred_strong != correct_ans:
                self.acc_list_strong.append(1)
                strong_success = True
            else:
                self.acc_list_strong.append(0)
                strong_success = False

            # ==========================================
            # 🛡️ Resize + Gaussian defense
            # ==========================================

            adv_img_rg = resize_gaussian_defense(
                final_adv_img,
                min_scale=0.9,
                max_scale=1.0,
                kernel=3
            )

            pred_rg = self.black_box_predict(adv_img_rg, final_adv_text)

            # RG Combined + Sharpen (加銳化)
            adv_img_rg_s = apply_balanced_sharpning(adv_img_rg)
            pred_rg_s = self.black_box_predict(adv_img_rg_s, final_adv_text)

            if pred_rg != correct_ans:
                rg_success = True
            else:
                rg_success = False

            # === 🎯 甜蜜點消融實驗搜尋區 (Sweet Spot Search) ===
            sweet_spot_data = {}
            
            for scale in search_scales:
                # 1. 執行不同強度的防禦轉換 T(theta)
                # 使用你現有的 input_defense 或 resize_gaussian_defense
                # 這裡建議用 resize_gaussian 作為綜合指標測試
                temp_defended_img = resize_gaussian_defense(final_adv_img, scale, scale, kernel=3)
                
                # 2. 獲取轉換後的預測結果 (用於一致性分析)
                temp_pred = self.black_box_predict(temp_defended_img, final_adv_text)
                
                # 3. 計算該強度下的各項指標
                temp_ssim = calculate_batch_ssim(ori_img, temp_defended_img)
                temp_lpips = self.calculate_lpips(ori_img, temp_defended_img)
                
                # 若你想加入語義一致性，可用 CLIP (此處示範省略，若需加入請告知)
                # temp_clip = self.calculate_clip_similarity(ori_img, temp_defended_img)

                # 4. 判定是否防禦成功 (ASR 狀態)
                defense_successful = (temp_pred == correct_ans)
                
                # 5. 儲存該刻度的結果
                sweet_spot_data[f"scale_{scale}"] = {
                    "ssim": float(temp_ssim),
                    "lpips": float(temp_lpips),
                    "defense_success": defense_successful,
                    "prediction": temp_pred
                }

           # ===== SSIM 計算 =====

            # attack vs original
            ssim_attack = calculate_batch_ssim(ori_img, final_adv_img)
            lpips_attack = self.calculate_lpips(ori_img, final_adv_img)
            clip_sim = self.calculate_clip_similarity(ori_img, final_adv_img)

            # Resize defense
            ssim_resize_mild = calculate_batch_ssim(final_adv_img, adv_img_mild)
            lpips_resize_mild = self.calculate_lpips(final_adv_img, adv_img_mild)
            ssim_resize_strong = calculate_batch_ssim(final_adv_img, adv_img_strong)
            lpips_resize_strong = self.calculate_lpips(final_adv_img, adv_img_strong)


            ssim_ori_resize_mild = calculate_batch_ssim(ori_img, adv_img_mild)
            lpips_ori_resize_mild = self.calculate_lpips(ori_img, adv_img_mild)
            ssim_ori_resize_strong = calculate_batch_ssim(ori_img, adv_img_strong)
            lpips_ori_resize_strong = self.calculate_lpips(ori_img, adv_img_strong)

            # Gaussian defense
            ssim_gauss_mild = calculate_batch_ssim(final_adv_img, blur_img_mild)
            lpips_gauss_mild = self.calculate_lpips(final_adv_img, blur_img_mild)
            ssim_gauss_strong = calculate_batch_ssim(final_adv_img, blur_img_strong)
            lpips_gauss_strong = self.calculate_lpips(final_adv_img, blur_img_strong)

            ssim_ori_gauss_mild = calculate_batch_ssim(ori_img, blur_img_mild)
            lpips_ori_gauss_mild = self.calculate_lpips(ori_img, blur_img_mild)
            ssim_ori_gauss_strong = calculate_batch_ssim(ori_img, blur_img_strong)
            lpips_ori_gauss_strong = self.calculate_lpips(ori_img, blur_img_strong)

            # Resize + Gaussian
            ssim_rg = calculate_batch_ssim(final_adv_img, adv_img_rg)
            lpips_rg = self.calculate_lpips(final_adv_img, adv_img_rg)
            ssim_ori_rg = calculate_batch_ssim(ori_img, adv_img_rg)
            lpips_ori_rg = self.calculate_lpips(ori_img, adv_img_rg)

            self.ssim_list.append(float(ssim_attack))
            self.lpips_list.append(float(lpips_attack))
            self.ssim_resize.append(float(ssim_resize_mild))
            self.ssim_gauss.append(float(ssim_gauss_mild))
            self.ssim_rg.append(float(ssim_rg))
            self.lpips_rg_list.append(float(lpips_rg))

            # 計算 SSIM (僅以 Mild 系列為例，其餘依此類推)
            ssim_rs_m_pure = calculate_batch_ssim(final_adv_img, adv_img_mild)
            lpips_rs_m_pure = self.calculate_lpips(final_adv_img, adv_img_mild)
            ssim_rs_m_sharp = calculate_batch_ssim(final_adv_img, adv_img_mild_s)
            lpips_rs_m_sharp = self.calculate_lpips(final_adv_img, adv_img_mild_s)

            ssim_rs_s_pure = calculate_batch_ssim(final_adv_img, adv_img_strong)
            lpips_rs_s_pure = self.calculate_lpips(final_adv_img, adv_img_strong)
            ssim_rs_s_sharp = calculate_batch_ssim(final_adv_img, adv_img_strong_s)
            lpips_rs_s_sharp = self.calculate_lpips(final_adv_img, adv_img_strong_s)

            ssim_gs_m_pure = calculate_batch_ssim(final_adv_img, blur_img_mild)
            lpips_gs_m_pure = self.calculate_lpips(final_adv_img, blur_img_mild)
            ssim_gs_m_sharp = calculate_batch_ssim(final_adv_img, blur_img_mild_s)
            lpips_gs_m_sharp = self.calculate_lpips(final_adv_img, blur_img_mild_s)

            ssim_gs_s_pure = calculate_batch_ssim(final_adv_img, blur_img_strong)
            lpips_gs_s_pure = self.calculate_lpips(final_adv_img, blur_img_strong)
            ssim_gs_s_sharp = calculate_batch_ssim(final_adv_img, blur_img_strong_s)
            lpips_gs_s_sharp = self.calculate_lpips(final_adv_img, blur_img_strong_s)

            ssim_rg_pure = calculate_batch_ssim(final_adv_img, adv_img_rg)
            lpips_rg_pure = self.calculate_lpips(final_adv_img, adv_img_rg)

            ssim_rg_sharp = calculate_batch_ssim(final_adv_img, adv_img_rg_s)
            lpips_rg_sharp = self.calculate_lpips(final_adv_img, adv_img_rg_s)



            # if i < 20:  # 只有前 20 張才執行存檔邏輯
                # 1. 先定義資料夾與檔名
            qid = int(batch['question_id'][0])
            v_dir = os.path.join(self.output_dir, 'visual_results')
            os.makedirs(v_dir, exist_ok=True)
            
            # 2. 定義 save_path (這是你剛才報錯的地方)
            save_adv_image(ori_img, os.path.join(v_dir, f"{qid}_1_ori.png"))
            save_adv_image(final_adv_img, os.path.join(v_dir, f"{qid}_2_atk.png"))
            
            # Resize 對比
            save_adv_image(adv_img_mild, os.path.join(v_dir, f"{qid}_3a_rs_m.png"))
            save_adv_image(adv_img_mild_s, os.path.join(v_dir, f"{qid}_3b_rs_m_sharp.png"))
            save_adv_image(adv_img_strong, os.path.join(v_dir, f"{qid}_3a_rs_s.png"))
            save_adv_image(adv_img_strong_s, os.path.join(v_dir, f"{qid}_3b_rs_s_sharp.png"))
            
            # Gaussian 對比
            save_adv_image(blur_img_mild, os.path.join(v_dir, f"{qid}_4a_gs_m.png"))
            save_adv_image(blur_img_mild_s, os.path.join(v_dir, f"{qid}_4b_gs_m_sharp.png"))
            save_adv_image(blur_img_strong, os.path.join(v_dir, f"{qid}_4a_gs_s.png"))
            save_adv_image(blur_img_strong_s, os.path.join(v_dir, f"{qid}_4b_gs_s_sharp.png"))
            
            # RG 對比
            save_adv_image(adv_img_rg, os.path.join(v_dir, f"{qid}_5a_rg.png"))
            save_adv_image(adv_img_rg_s, os.path.join(v_dir, f"{qid}_5b_rg_sharp.png"))
            # save_adv_image(adv_img_sharp, os.path.join(visual_dir, f"{qid}_4_sharp_def.png"))

            # 💡 注意：不要在 if i < 20 之外呼叫 save_path！

            # === 顯示即時數據 (每一筆都顯示) ===
            asr_nodef = sum(self.acc_list_nodefense) / len(self.acc_list_nodefense)
            asr_mild = sum(self.acc_list_mild) / len(self.acc_list_mild)
            asr_strong = sum(self.acc_list_strong) / len(self.acc_list_strong)
            avg_ssim = sum(self.ssim_list) / len(self.ssim_list)
            print(f'[Count: {len(self.acc_list_mild)}] SSIM: {avg_ssim:.4f} | NoDef: {asr_nodef:.2%} | Mild(0.9): {asr_mild:.2%} | Strong(0.6): {asr_strong:.2%}')
            
            # === 詳細 Log 紀錄 ===
            log_entry = {
                "question_id": int(batch['question_id'][0]),
                "question_original": batch['question'][0],
                "question_attack": final_adv_text,
                "ssim": {
                    "original_vs_attack": float(ssim_attack.item()),

                    "resize_mild": float(ssim_resize_mild.item()),
                    "resize_strong": float(ssim_resize_strong.item()),

                    "ori_resize_mild": float(ssim_ori_resize_mild.item()),
                    "ori_resize_strong": float(ssim_ori_resize_strong.item()),

                    "gaussian_mild": float(ssim_gauss_mild.item()),
                    "gaussian_strong": float(ssim_gauss_strong.item()),

                    "ori_gaussian_mild": float(ssim_ori_gauss_mild.item()),
                    "ori_gaussian_strong": float(ssim_ori_gauss_strong.item()),

                    "resize_gaussian": float(ssim_rg.item()),
                    "ori_resize_gaussian": float(ssim_ori_rg.item())
                },
                "lpips": {
                    "original_vs_attack": float(lpips_attack),

                    "resize_mild": float(lpips_resize_mild),
                    "resize_strong": float(lpips_resize_strong),

                    "ori_resize_mild": float(lpips_ori_resize_mild),
                    "ori_resize_strong": float(lpips_ori_resize_strong),

                    "gaussian_mild": float(lpips_gauss_mild),
                    "gaussian_strong": float(lpips_gauss_strong),

                    "ori_gaussian_mild": float(lpips_ori_gauss_mild),
                    "ori_gaussian_strong": float(lpips_ori_gauss_strong),

                    "resize_gaussian": float(lpips_rg),
                    "ori_resize_gaussian": float(lpips_ori_rg)
                },
                "clip": {
                    "original_vs_attack": float(clip_sim)
                },
                "sweet_spot_sweep": sweet_spot_data,
                "defense_comparison": {
                    "resize_mild": {
                        "pure_ssim": float(ssim_rs_m_pure.item()),
                        "sharp_ssim": float(ssim_rs_m_sharp.item()),
                        "pure_lpips": float(lpips_rs_m_pure),
                        "sharp_lpips": float(lpips_rs_m_sharp),
                        "pure_success": pred_mild != correct_ans,
                        "sharp_success": pred_mild_s != correct_ans
                    },
                    "resize_strong": {
                        "pure_ssim": float(ssim_rs_s_pure.item()),
                        "sharp_ssim": float(ssim_rs_s_sharp.item()),
                        "pure_lpips": float(lpips_rs_s_pure),
                        "sharp_lpips": float(lpips_rs_s_sharp),
                        "pure_success": pred_strong != correct_ans,
                        "sharp_success": pred_strong_s != correct_ans
                    },
                    "gaussian_mild": {
                        "pure_ssim": float(ssim_gs_m_pure.item()),
                        "sharp_ssim": float(ssim_gs_m_sharp.item()),
                        "pure_lpips": float(lpips_gs_m_pure),
                        "sharp_lpips": float(lpips_gs_m_sharp),
                        "pure_success": gauss_mild_pred != correct_ans,
                        "sharp_success": gauss_mild_s_pred != correct_ans
                    },
                    "gaussian_strong": {
                        "pure_ssim": float(ssim_gs_s_pure.item()),
                        "sharp_ssim": float(ssim_gs_s_sharp.item()),
                        "pure_lpips": float(lpips_gs_s_pure),
                        "sharp_lpips": float(lpips_gs_s_sharp),
                        "pure_success": gauss_strong_pred != correct_ans,
                        "sharp_success": gauss_strong_s_pred != correct_ans
                    },
                    "rg_combined": {
                        "pure_ssim": float(ssim_rg_pure.item()),
                        "sharp_ssim": float(ssim_rg_sharp.item()),
                        "pure_lpips": float(lpips_rg_pure),
                        "sharp_lpips": float(lpips_rg_sharp),
                        "pure_success": pred_rg != correct_ans,
                        "sharp_success": pred_rg_s != correct_ans
                    }
                },
                "ground_truth": correct_ans,
                "original_prediction": pred_ans,

                "adversarial_prediction": no_defense_pred, # 攻擊後(無防禦)的預測
                "attack_success": no_defense_pred != correct_ans,

                "attack_type":attack_type,

                "mild_defense": {
                    "scale": "0.9-1.0",
                    "prediction": pred_mild,
                    "attack_success": mild_success
                },
                "strong_defense": {
                    "scale": "0.5-0.7",
                    "prediction": pred_strong,
                    "attack_success": strong_success
                },
                "gaussian_mild": { # Kernel 3x3
                    "prediction": gauss_mild_pred,
                    "attack_success": gauss_mild_success
                },

                "gaussian_strong": { # Kernel 5x5
                    "prediction": gauss_strong_pred,
                    "attack_success": gauss_strong_success
                },
                "resize_gaussian": {
                    "prediction": pred_rg,
                    "attack_success": rg_success
                },
            }
            self.logs.append(log_entry)
            
            # 定期存檔
            if len(self.acc_list_mild) % 1 == 0:
                with open(self.log_path, 'w', encoding='utf-8') as f:
                    json.dump(self.logs, f, ensure_ascii=False, indent=4)

            gc.collect()             # 強制清理系統記憶體 (RAM)
            if torch.cuda.is_available():
                torch.cuda.empty_cache() # 如果你有用 GPU，清理顯存

        # 最後存檔
        with open(self.log_path, 'w', encoding='utf-8') as f:
            json.dump(self.logs, f, ensure_ascii=False, indent=4)
            
        # === 最終總結輸出 ===
        if self.ssim_list:
            print('=' * 40)
            print(f'Final Average SSIM: {sum(self.ssim_list) / len(self.ssim_list):.4f}')
        if self.acc_list_mild:
            print(f'Final ASR (Mild Defense 0.9): {sum(self.acc_list_mild) / len(self.acc_list_mild):.4f}')
        if self.acc_list_strong:
            print(f'Final ASR (Strong Defense 0.6): {sum(self.acc_list_strong) / len(self.acc_list_strong):.4f}')
        print('=' * 40)
        save_attack_rate_summary(self.output_dir, self.logs)
        print('Complete VLAttack & Dual Defense Experiment Done.')
        if self.ssim_resize:
            print("Avg SSIM Resize:", sum(self.ssim_resize)/len(self.ssim_resize))
        if self.ssim_gauss:
            print("Avg SSIM Gaussian:", sum(self.ssim_gauss)/len(self.ssim_gauss))
        if self.ssim_rg:
            print("Avg SSIM Resize+Gaussian:", sum(self.ssim_rg)/len(self.ssim_rg))
        print(f'Detailed logs saved to: {self.log_path}')