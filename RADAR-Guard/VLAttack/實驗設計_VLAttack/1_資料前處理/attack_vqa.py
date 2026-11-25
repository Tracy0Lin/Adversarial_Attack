'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
from pathlib import Path
import importlib
import json
import tensorflow as tf
import tensorflow_hub as hub
from models.blip_pretrain import blip_pretrain
import sys
import gc
sys.path.append('../cleverhans')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.backends.cudnn as cudnn
from models.blip_vqa import blip_vqa
import utils
from data import create_dataset, create_sampler, create_loader
from data.vqa_dataset import vqa_collate_fn
from transformers import BertTokenizer


def init_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token': '[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens': ['[ENC]']})
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
    return tokenizer


def main(args, config, config_pretrain):
    # utils.init_distributed_mode(args)
    device = torch.device(args.device)
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    #### Dataset ####
    print("Creating vqa datasets")
    datasets = create_dataset('vqa', config)
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler(datasets, [True, False], num_tasks, global_rank)
    else:
        samplers = [None, None]
    train_loader, test_loader = create_loader(datasets, samplers,
                                              batch_size=[config['batch_size_train'], config['batch_size_test']],
                                              num_workers=[0, 0], is_trains=[True, False],
                                              collate_fns=[vqa_collate_fn, None])
    #### Model ####
    print("Creating model")
    model = blip_vqa(pretrained=config['pretrained'], image_size=config['image_size'],
                     vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'])

    model = model.to(device)
    pretrain_model = blip_pretrain(pretrained=config_pretrain['pretrained'], image_size=config_pretrain['image_size'],
                                   vit=config_pretrain['vit'], vit_grad_ckpt=config['vit_grad_ckpt'],
                                   vit_ckpt_layer=config_pretrain['vit_ckpt_layer'],
                                   queue_size=config_pretrain['queue_size'])
    pretrain_model = pretrain_model.to(device)
    tokenizer = init_tokenizer()
    with tf.device('cpu'):
        USE_model = hub.load(args.USE_model_path)
    f = open(args.correct_idx_store, 'r')
    a = list(f)
    f.close()
    correct_list = [int(l.strip('\n')) for l in a]
    with open(args.correct_pred_store, 'r') as f:
        answer_list = json.load(f)

    # ==========================================
    # 🚀 這裡開始 Grid Search
    # ==========================================
    step_options = args.steps if args.steps else [100, 120, 150]
    alpha_options = args.alphas if args.alphas else [0.002, 0.0015, 0.001]
    run_suffix = args.run_suffix or ''

    # 取得目前程式所在的絕對路徑
    base_abs_path = os.path.dirname(os.path.abspath(__file__))
    vqa_output_root = os.path.join(base_abs_path, args.output_dir)

    for s in step_options:
        for a in alpha_options:
            print(f"\n[Grid] Steps={s}, Alpha={a}")
            
            # 💡 強制使用絕對路徑建立資料夾（run_suffix 可開新目錄，不覆蓋舊實驗）
            folder_name = f"steps{s}_alpha{a}{run_suffix}"
            current_output_dir = os.path.join(vqa_output_root, folder_name)
            log_path = os.path.join(current_output_dir, "attack_dual_defense_log.json") # 確保檔名跟類別內一致
            
            # 2. ⚡ 快速檢查：如果這組已經跑完目標筆數，直接跳過不進 evaluate
            num_existing = 0
            if os.path.exists(log_path):
                try:
                    with open(log_path, 'r', encoding='utf-8') as f:
                        existing_logs = json.load(f)
                        num_existing = len(existing_logs)
                        if num_existing >= args.num_samples:
                            print(f"[Skip] Steps={s}, Alpha={a} already has {len(existing_logs)} logs.")
                            continue
                        else:
                            print(f"[Resume] Steps={s}, Alpha={a} progress {num_existing}/{args.num_samples}")
                except Exception as e:
                    print(f"[Warn] Could not read log, restarting combo. Error: {e}")

            # 3. 如果沒跑完，才開始執行
            if num_existing == 0:
                print(f"\n[New run] Steps={s}, Alpha={a}")
            os.makedirs(current_output_dir, exist_ok=True)
            print(f"Output dir: {current_output_dir}")

            # 初始化攻擊
            pgd_attack = Adv_attack(model, pretrain_model, tokenizer, device, 
                                    correct_list, answer_list, USE_model, current_output_dir)
            
            # 覆蓋參數
            pgd_attack.total_stg_step = s
            pgd_attack.alpha = a
            pgd_attack.max_samples = args.num_samples
            
            # --- 💡 關鍵：先確認資料夾裡有東西 ---
            print("Starting evaluate...")
            pgd_attack.evaluate(test_loader, tokenizer)
            
            from attack.VQA.attack_rate_summary import save_attack_rate_summary
            save_attack_rate_summary(current_output_dir, pgd_attack.logs)
            
            print(f"Done: Steps={s}, Alpha={a}")
            
            # 釋放記憶體防止連續跑崩潰
            torch.cuda.empty_cache()
            gc.collect()
    # pgd_attack = Adv_attack(model, pretrain_model, tokenizer, device, correct_list, answer_list, USE_model, args.output_dir)
    # pgd_attack.evaluate(test_loader, tokenizer)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_pre', default='./configs/pretrain.yaml')
    parser.add_argument('--config', default='./configs/vqa.yaml')
    parser.add_argument('--output_dir', default='output/VQA')
    parser.add_argument('--correct_idx_store', default='right_vqa_list.txt')
    parser.add_argument('--USE_model_path', default='https://tfhub.dev/google/universal-sentence-encoder-large/5')
    parser.add_argument('--correct_pred_store', default='right_vqa_ans_table.txt')
    parser.add_argument('--method', default='',type=str)
    parser.add_argument('--run_suffix', default='', type=str,
                        help='Append to output folder name, e.g. _run20250912 (keeps original results)')
    parser.add_argument('--steps', nargs='+', type=int, default=None,
                        help='PGD step counts to run (default: 100 120 150)')
    parser.add_argument('--alphas', nargs='+', type=float, default=None,
                        help='PGD alpha values to run (default: 0.002 0.0015 0.001)')
    parser.add_argument('--num_samples', type=int, default=200,
                        help='Number of attacked samples to log per (steps, alpha) combo')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=False, type=bool)
    args = parser.parse_args()
    if args.method=='BERTAttack':
        module=importlib.import_module('attack.VQA.adv_attack_BERTattack')
        Adv_attack=getattr(module,'Adv_attack')
    elif args.method=='BSA':
        module = importlib.import_module('attack.VQA.adv_attack_blip')
        Adv_attack =getattr(module,'Adv_attack')
    elif args.method=='Co-Attack':
        module = importlib.import_module('attack.VQA.adv_attack_coattack')
        Adv_attack =getattr(module,'Adv_attack')
    elif args.method == 'VLAttack':
        module = importlib.import_module('attack.VQA.adv_attack_blip_vla')
        Adv_attack =getattr(module,'Adv_attack')
    else:
        print('attack method is not supported!')
        raise ValueError


    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    config_pretrain = yaml.load(open(args.config_pre, 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))
    main(args, config, config_pretrain)