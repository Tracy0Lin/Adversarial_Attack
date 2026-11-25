import argparse
import os
import random
import clip
import numpy as np
import torch
import torchvision
from PIL import Image
# This pilot does not enable wandb.

# LAVIS caption interface replaced by ViT-GPT2.
# LAVIS caption interface replaced by ViT-GPT2.

# seed for everything
# credit: https://www.kaggle.com/code/rhythmcam/random-seed-everything
DEFAULT_RANDOM_SEED = 2023
device = "cuda" if torch.cuda.is_available() else "cpu"

# basic random seed
def seedBasic(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

# torch random seed
def seedTorch(seed=DEFAULT_RANDOM_SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# combine
def seedEverything(seed=DEFAULT_RANDOM_SEED):
    seedBasic(seed)
    seedTorch(seed)
# ------------------------------------------------------------------ #  

def to_tensor(pic):
    mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
    img = torch.from_numpy(np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True))
    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
    img = img.permute((2, 0, 1)).contiguous()
    return img.to(dtype=torch.get_default_dtype())


# to replace vis_processor
transform_a = torchvision.transforms.Compose(
    [   
        torchvision.transforms.Lambda(lambda img: img.convert("RGB")),
        torchvision.transforms.Resize(size=(384, 384), interpolation=torchvision.transforms.InterpolationMode.BICUBIC, max_size=None, antialias='warn'),
        torchvision.transforms.Lambda(lambda img: to_tensor(img)),
        # torchvision.transforms.ToTensor(),
    ]
)
transform_b = torchvision.transforms.Compose(
    [   
        torchvision.transforms.Lambda(lambda img: img.convert("RGB")),
        torchvision.transforms.Resize(size=(224, 224), interpolation=torchvision.transforms.InterpolationMode.BICUBIC, max_size=None, antialias='warn'),
        torchvision.transforms.Lambda(lambda img: to_tensor(img)),
        # torchvision.transforms.ToTensor(),
    ]
)
normalize = torchvision.transforms.Compose(
    [   
        # torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ]
)


class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    def __getitem__(self, index: int):
        original_tuple = super().__getitem__(index)
        path, _ = self.samples[index]
        return original_tuple + (path,)


_i2t = _vit_i2t


if __name__ == "__main__":
    seedEverything()
    parser = argparse.ArgumentParser()
    # load models for i2t
    parser.add_argument("--model_name", default="blip_caption", type=str)
    parser.add_argument("--model_type", default="base_coco", type=str)
    
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_samples", default=5, type=int)
    parser.add_argument("--input_res", default='224', type=int)
    parser.add_argument("--alpha", default=1.0, type=float)
    parser.add_argument("--epsilon", default=8, type=int)
    parser.add_argument("--steps", default=1, type=int)
    parser.add_argument("--output", default="temp", type=str)
    parser.add_argument("--image_output", required=True, type=str)
    parser.add_argument("--data_path", default="temp", type=str)
    parser.add_argument("--text_path", default="temp.txt", type=str)
    parser.add_argument("--clean_dir", required=True, type=str)
    parser.add_argument("--target_text_path", required=True, type=str)
    parser.add_argument("--num_workers", default=0, type=int)
    
    parser.add_argument("--delta", default="normal", type=str)
    parser.add_argument("--num_query", default=20, type=int)
    parser.add_argument("--num_sub_query", default=5, type=int)
    parser.add_argument("--sigma", default=16, type=float)
    
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project_name", type=str, default='temp_proj')
    parser.add_argument("--wandb_run_name", type=str, default='temp_run')
    
    args = parser.parse_args()

    # ---------------------- #
    print(f"Loading ViT-GPT2 adapter: {args.model_name}, model_type: {args.model_type}...")
    # load models for i2t
    model, vis_processors, txt_processors = load_model_and_preprocess(name=args.model_name, model_type=args.model_type, is_eval=True, device=device)
    
    # use clip text coder for attack
    clip_img_model_vitb32, _ = clip.load("ViT-B/32", device=device, jit=False)
    print("Done")
    # ---------------------- #

    # load clip_model params
    num_sub_query, num_query, sigma = args.num_sub_query, args.num_query, args.sigma
    batch_size    = args.batch_size
    alpha         = args.alpha
    epsilon       = args.epsilon
    
    args.input_res = 224

    if args.input_res == 384:
        vit_adv_data  = ImageFolderWithPaths(args.data_path, transform=transform_a)
        clean_data    = ImageFolderWithPaths(args.clean_dir, transform=transform_a)
    
    else:
        vit_adv_data  = ImageFolderWithPaths(args.data_path, transform=transform_b)
        clean_data    = ImageFolderWithPaths(args.clean_dir, transform=transform_b)
        
    data_loader       = torch.utils.data.DataLoader(vit_adv_data, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
    clean_data_loader = torch.utils.data.DataLoader(clean_data, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
        
    # org text/features
    adv_vit_text_path = args.text_path
    with open(os.path.join(adv_vit_text_path), 'r') as f:
        lavis_text_of_adv_vit  = f.readlines()[:args.num_samples] 
        f.close()
    
    with torch.no_grad():
        adv_vit_text_token    = clip.tokenize(lavis_text_of_adv_vit).to(device)
        adv_vit_text_features = clip_img_model_vitb32.encode_text(adv_vit_text_token)
        adv_vit_text_features = adv_vit_text_features / adv_vit_text_features.norm(dim=1, keepdim=True)
        adv_vit_text_features = adv_vit_text_features.detach()
    
    # tgt text/features
    tgt_text_path = args.target_text_path
    with open(os.path.join(tgt_text_path), 'r') as f:
        tgt_text  = f.readlines()[:args.num_samples] 
        f.close()
    
    # clip text features of the target
    with torch.no_grad():
        target_text_token    = clip.tokenize(tgt_text).to(device)
        target_text_features = clip_img_model_vitb32.encode_text(target_text_token)
        target_text_features = target_text_features / target_text_features.norm(dim=1, keepdim=True)
        target_text_features = target_text_features.detach()

    # baseline results
    vit_attack_results   = torch.sum(adv_vit_text_features * target_text_features, dim=1).reshape(-1).detach().cpu().numpy()
    query_attack_results = torch.sum(adv_vit_text_features * target_text_features, dim=1).reshape(-1).detach().cpu().numpy()
    assert (vit_attack_results == query_attack_results).all()
    
    # Other CLIP evaluators omitted for 4GB GPU; RGF uses ViT-B/32.
    
    if args.wandb:
        run = wandb.init(project=args.wandb_project_name, name=args.wandb_run_name, reinit=True)
    
    for i, ((image, _, path), (image_clean, _, _)) in enumerate(zip(data_loader, clean_data_loader)):
        if batch_size * (i+1) > args.num_samples:
            break
        image = image.to(device)  # size=(10, 3, 224, 224)
        image_clean = image_clean.to(device)  # size=(10, 3, 224, 224)
        
        # obtain all text features (via CLIP text encoder)
        adv_text_features = adv_vit_text_features[batch_size * (i): batch_size * (i+1)]        
        tgt_text_features = target_text_features[batch_size * (i): batch_size * (i+1)]
        
        # ------------------- random gradient-free method
        print("init delta with projected diff(adv-clean)")

        delta = torch.clamp(
            (image - image_clean).clone().detach(),
            min=-epsilon,
            max=epsilon,
        )

        # Reconstruct a valid baseline after resizing to victim resolution.
        image = torch.clamp(image_clean + delta, 0.0, 255.0)

        with torch.no_grad():
            projected_baseline_caption = _i2t(
                args,
                txt_processors,
                model,
                image=image,
            )

            projected_baseline_token = clip.tokenize(
                projected_baseline_caption
            ).to(device)

            projected_baseline_features = (
                clip_img_model_vitb32.encode_text(
                    projected_baseline_token
                )
            )

            projected_baseline_features = (
                projected_baseline_features
                / projected_baseline_features.norm(
                    dim=1,
                    keepdim=True,
                )
            )

            projected_baseline_features = (
                projected_baseline_features.detach()
            )

            projected_baseline_score = torch.mean(
                torch.sum(
                    projected_baseline_features
                    * tgt_text_features,
                    dim=1,
                )
            ).item()

        adv_text_features = projected_baseline_features
        query_attack_results[i] = projected_baseline_score

        best_caption = projected_baseline_caption[0].strip()
        best_image = image.clone().detach()
        best_score = projected_baseline_score
        better_flag = 0

        print("projected baseline caption:", best_caption)
        print(
            "projected baseline target similarity:",
            f"{best_score:.6f}",
        )
        print(
            "projected baseline max delta:",
            torch.max(torch.abs(delta)).item(),
        )

        torch.cuda.empty_cache()
        
        for step_idx in range(args.steps):
            print(f"{i}-th image - {step_idx}-th step")
            # step 1. obtain purturbed images
            if step_idx == 0:
                image_repeat      = image.repeat(num_query, 1, 1, 1)  # size = (num_query x batch_size, 3, args.input_res, args.input_res)
            else:
                image_repeat      = adv_image_in_current_step.repeat(num_query, 1, 1, 1)             

                lavis_text_of_adv_image_in_current_step = _i2t(args, txt_processors, model, image=adv_image_in_current_step)
                adv_vit_text_token_in_current_step      = clip.tokenize(lavis_text_of_adv_image_in_current_step).to(device)
                adv_vit_text_features_in_current_step   = clip_img_model_vitb32.encode_text(adv_vit_text_token_in_current_step)
                adv_vit_text_features_in_current_step   = adv_vit_text_features_in_current_step / adv_vit_text_features_in_current_step.norm(dim=1, keepdim=True)
                adv_vit_text_features_in_current_step   = adv_vit_text_features_in_current_step.detach()                
                adv_text_features     = adv_vit_text_features_in_current_step
                torch.cuda.empty_cache()
            query_noise            = torch.randn_like(image_repeat).sign() # Rademacher noise
            perturbed_image_repeat = torch.clamp(image_repeat + (sigma * query_noise), 0.0, 255.0)  # size = (num_query x batch_size, 3, args.input_res, args.input_res)
            
            # num_query is obtained via serveral iterations
            text_of_perturbed_imgs = []
            for query_idx in range(num_query//num_sub_query):
                sub_perturbed_image_repeat = perturbed_image_repeat[num_sub_query * (query_idx) : num_sub_query * (query_idx+1)]
                if args.model_name == 'img2prompt_vqa':
                    text_of_sub_perturbed_imgs = _i2t(args, txt_processors, model, image=sub_perturbed_image_repeat)
                else:
                    with torch.no_grad():
                        text_of_sub_perturbed_imgs = _i2t(args, txt_processors, model, image=sub_perturbed_image_repeat)
                text_of_perturbed_imgs.extend(text_of_sub_perturbed_imgs)
            
            # step 2. estimate grad
            with torch.no_grad():
                perturb_text_token    = clip.tokenize(text_of_perturbed_imgs).to(device)
                perturb_text_features = clip_img_model_vitb32.encode_text(perturb_text_token)
                perturb_text_features = perturb_text_features / perturb_text_features.norm(dim=1, keepdim=True)
                perturb_text_features = perturb_text_features.detach()
            
            coefficient = torch.sum((perturb_text_features - adv_text_features) * tgt_text_features, dim=-1)  # size = (num_query * batch_size)
            coefficient = coefficient.reshape(num_query, batch_size, 1, 1, 1)
            query_noise = query_noise.reshape(num_query, batch_size, 3, args.input_res, args.input_res)
            pseudo_gradient = coefficient * query_noise / sigma # size = (num_query, batch_size, 3, args.input_res, args.input_res)
            pseudo_gradient = pseudo_gradient.mean(0) # size = (bs, 3, args.input_res, args.input_res)
            
            # step 3. log metrics
            delta_data = torch.clamp(delta + alpha * torch.sign(pseudo_gradient), min=-epsilon, max=epsilon)
            delta.data = delta_data
            print(f"img: {i:3d}-step {step_idx} max  delta", torch.max(torch.abs(delta)).item())
            print(f"img: {i:3d}-step {step_idx} mean delta", torch.mean(torch.abs(delta)).item())
            
            adv_image_in_current_step = torch.clamp(image_clean+delta, 0.0, 255.0)
            
            # get adv text
            if args.model_name == 'img2prompt_vqa':
                lavis_text_of_adv_image_in_current_step = _i2t(args, txt_processors, model, image=adv_image_in_current_step)
            else:
                with torch.no_grad():
                    lavis_text_of_adv_image_in_current_step = _i2t(args, txt_processors, model, image=adv_image_in_current_step)
            
            # log sim
            with torch.no_grad():
                lavis_text_token = clip.tokenize(lavis_text_of_adv_image_in_current_step).to(device)
                lavis_text_features_of_adv_image_in_current_step = clip_img_model_vitb32.encode_text(lavis_text_token)
                lavis_text_features_of_adv_image_in_current_step = lavis_text_features_of_adv_image_in_current_step / lavis_text_features_of_adv_image_in_current_step.norm(dim=1, keepdim=True)
                lavis_text_features_of_adv_image_in_current_step = lavis_text_features_of_adv_image_in_current_step.detach()

                adv_txt_tgt_txt_score_in_current_step = torch.mean(torch.sum(lavis_text_features_of_adv_image_in_current_step * tgt_text_features, dim=1)).item()
                
                # update results
                if adv_txt_tgt_txt_score_in_current_step > query_attack_results[i]:
                    query_attack_results[i] = adv_txt_tgt_txt_score_in_current_step
                    best_caption = lavis_text_of_adv_image_in_current_step[0]
                    best_image = adv_image_in_current_step.clone().detach()
                    best_score = adv_txt_tgt_txt_score_in_current_step
                    better_flag = 1
                    
                # Other CLIP evaluators omitted; ViT-B/32 score retained.
                

        if args.wandb:
            wandb.log(
                {   
                    "moving-avg-adv-rn50"    : np.mean(vit_attack_results_rn50[:(i+1)]),
                    "moving-avg-query-rn50"  : np.mean(query_attack_results_rn50[:(i+1)]),
                    
                    "moving-avg-adv-rn101"   : np.mean(vit_attack_results_rn101[:(i+1)]),
                    "moving-avg-query-rn101" : np.mean(query_attack_results_rn101[:(i+1)]),
                    
                    "moving-avg-adv-vitb16"  : np.mean(vit_attack_results_vitb16[:(i+1)]),
                    "moving-avg-query-vitb16": np.mean(query_attack_results_vitb16[:(i+1)]),
                    
                    "moving-avg-adv-vitb32"  : np.mean(vit_attack_results[:(i+1)]),
                    "moving-avg-query-vitb32": np.mean(query_attack_results[:(i+1)]),
                    
                    "moving-avg-adv-vitl14"  : np.mean(vit_attack_results_vitl14[:(i+1)]),
                    "moving-avg-query-vitl14": np.mean(query_attack_results_vitl14[:(i+1)]),
                }
            )

        # log text
        print("best caption of current image:", best_caption)
        with open(os.path.join(args.output + '.txt'), 'a') as f:
            f.write(' '.join(best_caption.split()) + '\n')

        folder = os.path.basename(os.path.dirname(path[0]))
        image_name = os.path.splitext(os.path.basename(path[0]))[0] + ".png"
        image_folder = os.path.join(args.image_output, folder)
        os.makedirs(image_folder, exist_ok=True)
        torchvision.utils.save_image(
            best_image[0] / 255.0,
            os.path.join(image_folder, image_name)
        )
        print(f"best target similarity: {best_score:.6f}")
        print("best image saved to:", os.path.join(image_folder, image_name))
        f.close()