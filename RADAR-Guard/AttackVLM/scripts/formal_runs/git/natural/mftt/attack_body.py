image = image.to(device)
image_clean = image_clean.to(device)
adv_text_features = adv_vit_text_features[batch_size * i:batch_size * (i + 1)]
tgt_text_features = target_text_features[batch_size * i:batch_size * (i + 1)]
print('init delta with projected diff(adv-clean)')
delta = torch.clamp((image - image_clean).clone().detach(), min=-epsilon, max=epsilon)
image = torch.clamp(image_clean + delta, 0.0, 255.0)
with torch.no_grad():
    projected_baseline_caption = _i2t(args, txt_processors, model, image=image)
    projected_baseline_token = clip.tokenize(projected_baseline_caption).to(device)
    projected_baseline_features = clip_img_model_vitb32.encode_text(projected_baseline_token)
    projected_baseline_features = projected_baseline_features / projected_baseline_features.norm(dim=1, keepdim=True)
    projected_baseline_features = projected_baseline_features.detach()
    projected_baseline_score = torch.mean(torch.sum(projected_baseline_features * tgt_text_features, dim=1)).item()
adv_text_features = projected_baseline_features
query_attack_results[i] = projected_baseline_score
best_caption = projected_baseline_caption[0].strip()
best_image = image.clone().detach()
best_score = projected_baseline_score
better_flag = 0
print('projected baseline caption:', best_caption)
print('projected baseline target similarity:', f'{best_score:.6f}')
print('projected baseline max delta:', torch.max(torch.abs(delta)).item())
torch.cuda.empty_cache()
for step_idx in range(args.steps):
    print(f'{i}-th image - {step_idx}-th step')
    if step_idx == 0:
        image_repeat = image.repeat(num_query, 1, 1, 1)
    else:
        image_repeat = adv_image_in_current_step.repeat(num_query, 1, 1, 1)
        lavis_text_of_adv_image_in_current_step = _i2t(args, txt_processors, model, image=adv_image_in_current_step)
        adv_vit_text_token_in_current_step = clip.tokenize(lavis_text_of_adv_image_in_current_step).to(device)
        adv_vit_text_features_in_current_step = clip_img_model_vitb32.encode_text(adv_vit_text_token_in_current_step)
        adv_vit_text_features_in_current_step = adv_vit_text_features_in_current_step / adv_vit_text_features_in_current_step.norm(dim=1, keepdim=True)
        adv_vit_text_features_in_current_step = adv_vit_text_features_in_current_step.detach()
        adv_text_features = adv_vit_text_features_in_current_step
        torch.cuda.empty_cache()
    query_noise = torch.randn_like(image_repeat).sign()
    perturbed_image_repeat = torch.clamp(image_repeat + sigma * query_noise, 0.0, 255.0)
    text_of_perturbed_imgs = []
    for query_idx in range(num_query // num_sub_query):
        sub_perturbed_image_repeat = perturbed_image_repeat[num_sub_query * query_idx:num_sub_query * (query_idx + 1)]
        if args.model_name == 'img2prompt_vqa':
            text_of_sub_perturbed_imgs = _i2t(args, txt_processors, model, image=sub_perturbed_image_repeat)
        else:
            with torch.no_grad():
                text_of_sub_perturbed_imgs = _i2t(args, txt_processors, model, image=sub_perturbed_image_repeat)
        text_of_perturbed_imgs.extend(text_of_sub_perturbed_imgs)
    with torch.no_grad():
        perturb_text_token = clip.tokenize(text_of_perturbed_imgs).to(device)
        perturb_text_features = clip_img_model_vitb32.encode_text(perturb_text_token)
        perturb_text_features = perturb_text_features / perturb_text_features.norm(dim=1, keepdim=True)
        perturb_text_features = perturb_text_features.detach()
    coefficient = torch.sum((perturb_text_features - adv_text_features) * tgt_text_features, dim=-1)
    coefficient = coefficient.reshape(num_query, batch_size, 1, 1, 1)
    query_noise = query_noise.reshape(num_query, batch_size, 3, args.input_res, args.input_res)
    pseudo_gradient = coefficient * query_noise / sigma
    pseudo_gradient = pseudo_gradient.mean(0)
    delta_data = torch.clamp(delta + alpha * torch.sign(pseudo_gradient), min=-epsilon, max=epsilon)
    delta.data = delta_data
    print(f'img: {i:3d}-step {step_idx} max  delta', torch.max(torch.abs(delta)).item())
    print(f'img: {i:3d}-step {step_idx} mean delta', torch.mean(torch.abs(delta)).item())
    adv_image_in_current_step = torch.clamp(image_clean + delta, 0.0, 255.0)
    if args.model_name == 'img2prompt_vqa':
        lavis_text_of_adv_image_in_current_step = _i2t(args, txt_processors, model, image=adv_image_in_current_step)
    else:
        with torch.no_grad():
            lavis_text_of_adv_image_in_current_step = _i2t(args, txt_processors, model, image=adv_image_in_current_step)
    with torch.no_grad():
        lavis_text_token = clip.tokenize(lavis_text_of_adv_image_in_current_step).to(device)
        lavis_text_features_of_adv_image_in_current_step = clip_img_model_vitb32.encode_text(lavis_text_token)
        lavis_text_features_of_adv_image_in_current_step = lavis_text_features_of_adv_image_in_current_step / lavis_text_features_of_adv_image_in_current_step.norm(dim=1, keepdim=True)
        lavis_text_features_of_adv_image_in_current_step = lavis_text_features_of_adv_image_in_current_step.detach()
        adv_txt_tgt_txt_score_in_current_step = torch.mean(torch.sum(lavis_text_features_of_adv_image_in_current_step * tgt_text_features, dim=1)).item()
        if adv_txt_tgt_txt_score_in_current_step > query_attack_results[i]:
            query_attack_results[i] = adv_txt_tgt_txt_score_in_current_step
            best_caption = lavis_text_of_adv_image_in_current_step[0]
            best_image = adv_image_in_current_step.clone().detach()
            best_score = adv_txt_tgt_txt_score_in_current_step
            better_flag = 1
if args.wandb:
    wandb.log({'moving-avg-adv-rn50': np.mean(vit_attack_results_rn50[:i + 1]), 'moving-avg-query-rn50': np.mean(query_attack_results_rn50[:i + 1]), 'moving-avg-adv-rn101': np.mean(vit_attack_results_rn101[:i + 1]), 'moving-avg-query-rn101': np.mean(query_attack_results_rn101[:i + 1]), 'moving-avg-adv-vitb16': np.mean(vit_attack_results_vitb16[:i + 1]), 'moving-avg-query-vitb16': np.mean(query_attack_results_vitb16[:i + 1]), 'moving-avg-adv-vitb32': np.mean(vit_attack_results[:i + 1]), 'moving-avg-query-vitb32': np.mean(query_attack_results[:i + 1]), 'moving-avg-adv-vitl14': np.mean(vit_attack_results_vitl14[:i + 1]), 'moving-avg-query-vitl14': np.mean(query_attack_results_vitl14[:i + 1])})
print('best caption of current image:', best_caption)
with open(os.path.join(args.output + '.txt'), 'a') as f:
    f.write(' '.join(best_caption.split()) + '\n')
folder = os.path.basename(os.path.dirname(path[0]))
image_name = os.path.splitext(os.path.basename(path[0]))[0] + '.png'
image_folder = os.path.join(args.image_output, folder)
os.makedirs(image_folder, exist_ok=True)
torchvision.utils.save_image(best_image[0] / 255.0, os.path.join(image_folder, image_name))
print(f'best target similarity: {best_score:.6f}')
print('best image saved to:', os.path.join(image_folder, image_name))
f.close()
