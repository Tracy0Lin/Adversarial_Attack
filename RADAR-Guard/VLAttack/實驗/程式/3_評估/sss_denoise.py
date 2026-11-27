"""SSS (Sweet Spot Score) denoising module for TC-VAD stage-2 mid-risk routing.

    SSS = (delta_ASR * IQP * SP) ** (1/3)

    delta_ASR : attack success rate drop after gaussian denoise
    IQP       : geo-mean(SSIM, 1-LPIPS, LapVar ratio, 1-BRISQUE/100)
    SP        : geo-mean(CLIP similarity, MSS caption similarity)

IQP/SP are measured against the un-denoised adversarial input, which is what a
deployed detector actually has. The same metrics against the clean image are
also recorded (``*_oracle`` columns) as an upper bound for reporting only.

Theta search runs at two scopes: set-level picks one theta maximising the
aggregate SSS, per-sample picks theta per image as an adaptive-defense ablation.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from skimage.metrics import structural_similarity as compare_ssim

THIS_FILE = Path(__file__).resolve()
EXPERIMENT_ROOT = THIS_FILE.parent
sys.path.insert(0, str(THIS_FILE.parent))

from evaluate_best_stage1_new import (
    BLIP_ROOT,
    CaptionEmbedder,
    K_DEFAULT,
    load_caption_model,
    load_sentence_encoder,
    resolve_device,
    set_seed,
)

# The attack wrote its PNGs with save_adv_image(), which de-normalised using the
# ImageNet constants even though the VQA dataloader normalises with 0.5/0.5.
# Inverting the *save* transform is therefore what reconstructs the exact tensor
# the black-box model saw, so these constants must not be changed to 0.5.
SAVE_MEAN = (0.485, 0.456, 0.406)
SAVE_STD = (0.229, 0.224, 0.225)

VQA_IMAGE_SIZE = 480
CAPTION_IMAGE_SIZE = 384
EPS = 1e-6
DEFAULT_KERNELS = [5]
DEFAULT_SIGMAS = [0.3, 0.5, 0.8, 1.0, 1.5]


def vqa_config():
    import yaml

    with open(BLIP_ROOT / "configs" / "vqa.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def geo_mean(values):
    arr = np.clip(np.asarray(values, dtype=np.float64), EPS, 1.0)
    return float(np.exp(np.mean(np.log(arr))))


def gaussian_denoise(pil_img, kernel, sigma):
    tensor = TF.to_tensor(pil_img)
    blurred = TF.gaussian_blur(tensor, kernel_size=[kernel, kernel], sigma=[sigma, sigma])
    return TF.to_pil_image(blurred.clamp(0.0, 1.0))


def laplacian_var(pil_img):
    gray = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


class VQAScorer:
    """Re-runs the black-box BLIP VQA model on saved PNGs, matching the attack setup."""

    def __init__(self, vqa_ckpt, answer_list_path, device, k_test=128):
        sys.path.insert(0, str(BLIP_ROOT))
        os.chdir(BLIP_ROOT)  # blip_vqa() reads configs/med_config.json by relative path
        from models.blip_vqa import blip_vqa

        self.device = device
        self.k_test = k_test
        self.model = blip_vqa(
            pretrained=str(vqa_ckpt), image_size=VQA_IMAGE_SIZE, vit="base",
            vit_grad_ckpt=False, vit_ckpt_layer=0,
        ).to(device)
        self.model.eval()
        with open(answer_list_path, "r", encoding="utf-8") as f:
            self.answer_list = json.load(f)
        cand = self.model.tokenizer(self.answer_list, padding="longest", return_tensors="pt").to(device)
        cand.input_ids[:, 0] = self.model.tokenizer.bos_token_id
        self.answer_candidates = cand
        self.mean = torch.tensor(SAVE_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(SAVE_STD, device=device).view(1, 3, 1, 1)
        print(f"VQAScorer ready ({len(self.answer_list)} candidate answers)")

    def to_model_input(self, pil_img):
        img = pil_img.convert("RGB")
        if img.size != (VQA_IMAGE_SIZE, VQA_IMAGE_SIZE):
            img = img.resize((VQA_IMAGE_SIZE, VQA_IMAGE_SIZE), Image.BICUBIC)
        x = TF.to_tensor(img).unsqueeze(0).to(self.device)
        return (x - self.mean) / self.std

    @torch.no_grad()
    def predict(self, pil_img, question):
        x = self.to_model_input(pil_img)
        answer_ids, _, _ = self.model(
            x, [question], self.answer_candidates, train=False,
            inference="rank", k_test=self.k_test,
        )
        return self.answer_list[int(answer_ids[0])]


class QualityScorer:
    """SSIM / LPIPS / LapVar / BRISQUE / CLIP on [0,1] pixel space."""

    def __init__(self, device):
        import lpips
        import clip
        import piq

        self.device = device
        self.piq = piq
        self.lpips_fn = lpips.LPIPS(net="vgg").to(device)
        self.clip_model, _ = clip.load("ViT-B/32", device=device)
        print("QualityScorer ready (LPIPS-VGG, CLIP ViT-B/32, piq BRISQUE)")

    def _t(self, pil_img):
        return TF.to_tensor(pil_img.convert("RGB")).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def metrics(self, denoised, ref):
        a = self._t(denoised)
        b = self._t(ref)
        ssim = float(
            compare_ssim(
                np.asarray(denoised.convert("RGB"), dtype=np.float64) / 255.0,
                np.asarray(ref.convert("RGB"), dtype=np.float64) / 255.0,
                data_range=1.0, channel_axis=-1,
            )
        )
        lpips_val = float(self.lpips_fn((a - 0.5) * 2, (b - 0.5) * 2).item())
        try:
            brisque = float(self.piq.brisque(a, data_range=1.0).item())
        except Exception:
            brisque = float("nan")
        lap_ref = laplacian_var(ref)
        lap_den = laplacian_var(denoised)
        lap_ratio = 1.0 if lap_ref <= 0 else min(1.0, lap_den / lap_ref)
        fa = self.clip_model.encode_image(F.interpolate(a, (224, 224), mode="bilinear", align_corners=False))
        fb = self.clip_model.encode_image(F.interpolate(b, (224, 224), mode="bilinear", align_corners=False))
        fa = fa / fa.norm(dim=-1, keepdim=True)
        fb = fb / fb.norm(dim=-1, keepdim=True)
        clip_sim = float((fa * fb).sum(dim=-1).item())
        return {
            "ssim": ssim,
            "lpips": lpips_val,
            "brisque": brisque,
            "lap_var_ref": lap_ref,
            "lap_var_denoised": lap_den,
            "lap_ratio": lap_ratio,
            "clip_sim": clip_sim,
        }

    @staticmethod
    def iqp(m):
        factors = [m["ssim"], 1.0 - m["lpips"], m["lap_ratio"]]
        if not np.isnan(m["brisque"]):
            factors.append(1.0 - np.clip(m["brisque"], 0.0, 100.0) / 100.0)
        return geo_mean(factors)


class SemanticScorer:
    """MSS: caption-space cosine similarity between denoised and reference image."""

    def __init__(self, embedder):
        self.embedder = embedder
        self._cache = {}

    def embed(self, pil_img, cache_key=None):
        if cache_key is not None and cache_key in self._cache:
            return self._cache[cache_key]
        out = self.embedder.embed_image(pil_img)
        if cache_key is not None:
            self._cache[cache_key] = out
        return out

    def mss(self, denoised, ref, ref_key=None):
        e_den, cap_den = self.embed(denoised)
        e_ref, cap_ref = self.embed(ref, cache_key=ref_key)
        sim = float(torch.mm(e_den, e_ref.t()).item())
        return max(0.0, min(1.0, sim)), cap_den, cap_ref


def load_log_index(log_path):
    with open(log_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    index = {}
    for item in data:
        index[str(item["question_id"])] = {
            "question": item.get("question_original", ""),
            "ground_truth": item.get("ground_truth", ""),
            "correct_ans": item.get("original_prediction", item.get("ground_truth", "")),
            "attack_success": bool(item.get("attack_success", False)),
        }
    return index


def run_sss(
    experiment_root=None,
    visual_dir=None,
    stage1_csv=None,
    log_path=None,
    vqa_ckpt=None,
    answer_list=None,
    caption_ckpt=None,
    device="cuda",
    method="rotate_5",
    kernels=None,
    sigmas=None,
    seed=42,
    limit=None,
    out_suffix=None,
):
    tag = "" if out_suffix is None else f"_{out_suffix}"
    # load_caption_model() chdirs into BLIP_ROOT, so every path must be absolute first.
    root = Path(experiment_root or EXPERIMENT_ROOT).resolve()
    visual_dir = Path(visual_dir or (root / "visual_results")).resolve()
    stage1_csv = Path(stage1_csv or (root / "tcvad_stage1_4_academic_dataset.csv")).resolve()
    log_path = Path(log_path or (root / "attack_dual_defense_log.json")).resolve()
    caption_ckpt = Path(caption_ckpt or (BLIP_ROOT / "checkpoints" / "model_base_caption_capfilt_large.pth")).resolve()
    kernels = list(kernels or DEFAULT_KERNELS)
    sigmas = list(sigmas or DEFAULT_SIGMAS)
    device = resolve_device(device)
    set_seed(seed)

    stage1 = pd.read_csv(stage1_csv)
    stage1["image_id"] = stage1["image_id"].astype(str)
    mid = stage1[(stage1["method_type"] == method) & (stage1["risk_bin"] == "mid")].copy()
    if limit:
        mid = mid.head(limit)
    total_method = len(stage1[stage1["method_type"] == method])
    print(f"Mid-risk rows on {method}: {len(mid)} / {total_method}")
    if mid.empty:
        print("No mid-risk samples; SSS skipped.")
        return None

    log_index = load_log_index(log_path)
    cfg = vqa_config()
    vqa = VQAScorer(
        vqa_ckpt or cfg["pretrained"],
        answer_list or (Path(cfg["ann_root"]) / "answer_list.json"),
        device,
        k_test=cfg.get("k_test", 128),
    )
    quality = QualityScorer(device)
    caption_model, processor = load_caption_model(str(caption_ckpt), device)
    tokenizer, sent_model = load_sentence_encoder(device)
    semantic = SemanticScorer(CaptionEmbedder(caption_model, processor, tokenizer, sent_model, device))

    thetas = [(k, s) for k in kernels for s in sigmas]
    print(f"Theta grid: {len(thetas)} combos -> kernels={kernels} sigmas={sigmas}")

    rows = []
    for i, rec in enumerate(mid.itertuples(index=False), 1):
        qid = str(rec.image_id)
        meta = log_index.get(qid)
        if meta is None:
            print(f"[skip] qid={qid} not in attack log")
            continue
        is_attack = int(rec.label) == 1
        ref_path = visual_dir / (f"{qid}_2_atk.png" if is_attack else f"{qid}_1_ori.png")
        clean_path = visual_dir / f"{qid}_1_ori.png"
        if not ref_path.is_file():
            print(f"[skip] missing {ref_path.name}")
            continue
        ref_img = Image.open(ref_path).convert("RGB")
        clean_img = Image.open(clean_path).convert("RGB") if clean_path.is_file() else None

        correct = meta["correct_ans"]
        pred_ref = vqa.predict(ref_img, meta["question"])
        fooled_before = pred_ref != correct

        for kernel, sigma in thetas:
            den = gaussian_denoise(ref_img, kernel, sigma)
            pred_den = vqa.predict(den, meta["question"])
            fooled_after = pred_den != correct

            m = quality.metrics(den, ref_img)
            mss, cap_den, cap_ref = semantic.mss(den, ref_img, ref_key=f"ref|{qid}|{int(rec.label)}")
            iqp = QualityScorer.iqp(m)
            sp = geo_mean([max(0.0, min(1.0, m["clip_sim"])), mss])

            row = {
                "image_id": qid,
                "label": int(rec.label),
                "kind": "attack" if is_attack else "clean",
                "method_type": method,
                "kernel": kernel,
                "sigma": sigma,
                "question": meta["question"],
                "correct_ans": correct,
                "log_attack_success": meta["attack_success"],
                "pred_ref": pred_ref,
                "pred_denoised": pred_den,
                "fooled_before": bool(fooled_before),
                "fooled_after": bool(fooled_after),
                "rescued": bool(fooled_before and not fooled_after),
                "risk_stage1": float(rec.risk_score),
                "iqp": iqp,
                "sp": sp,
                "mss": mss,
                "caption_ref": cap_ref,
                "caption_denoised": cap_den,
                **m,
            }
            if clean_img is not None:
                mo = quality.metrics(den, clean_img)
                mss_o, _, _ = semantic.mss(den, clean_img, ref_key=f"clean|{qid}")
                row["iqp_oracle"] = QualityScorer.iqp(mo)
                row["sp_oracle"] = geo_mean([max(0.0, min(1.0, mo["clip_sim"])), mss_o])
                row["ssim_oracle"] = mo["ssim"]
                row["lpips_oracle"] = mo["lpips"]
            rows.append(row)

        print(f"sss [{i}/{len(mid)}] qid={qid} label={rec.label} fooled_before={fooled_before}", flush=True)

    per_theta = pd.DataFrame(rows)
    per_theta.to_csv(root / f"sss_per_theta_raw{tag}.csv", index=False, encoding="utf-8")

    # ---- set-level: one theta for the whole mid-risk pool ----
    summary = []
    for (kernel, sigma), grp in per_theta.groupby(["kernel", "sigma"]):
        atk = grp[(grp["kind"] == "attack") & (grp["fooled_before"])]
        clean = grp[grp["kind"] == "clean"]
        asr_before = 1.0 if len(atk) else np.nan
        asr_after = float(atk["fooled_after"].mean()) if len(atk) else np.nan
        delta_asr = float(asr_before - asr_after) if len(atk) else np.nan
        iqp = float(grp["iqp"].mean())
        sp = float(grp["sp"].mean())
        sss = geo_mean([max(delta_asr, 0.0), iqp, sp]) if not np.isnan(delta_asr) else np.nan
        clean_acc_before = float((~clean["fooled_before"]).mean()) if len(clean) else np.nan
        clean_acc_after = float((~clean["fooled_after"]).mean()) if len(clean) else np.nan
        summary.append(
            {
                "kernel": kernel,
                "sigma": sigma,
                "n_attack_fooled": int(len(atk)),
                "n_clean": int(len(clean)),
                "asr_before": asr_before,
                "asr_after": asr_after,
                "delta_asr": delta_asr,
                "delta_asr_rel": delta_asr / asr_before if len(atk) else np.nan,
                "iqp": iqp,
                "sp": sp,
                "sss": sss,
                "clean_acc_before": clean_acc_before,
                "clean_acc_after": clean_acc_after,
                "clean_acc_drop": clean_acc_before - clean_acc_after if len(clean) else np.nan,
                "iqp_oracle": float(grp["iqp_oracle"].mean()) if "iqp_oracle" in grp else np.nan,
                "sp_oracle": float(grp["sp_oracle"].mean()) if "sp_oracle" in grp else np.nan,
            }
        )
    grid = pd.DataFrame(summary).sort_values("sss", ascending=False)
    grid.to_csv(root / f"sss_grid_search{tag}.csv", index=False, encoding="utf-8")

    best = grid.iloc[0]
    theta_star = {
        "scope": "set_level",
        "method_type": method,
        "kernel": int(best["kernel"]),
        "sigma": float(best["sigma"]),
        "sss": float(best["sss"]),
        "delta_asr": float(best["delta_asr"]),
        "iqp": float(best["iqp"]),
        "sp": float(best["sp"]),
    }

    # ---- per-sample: adaptive theta per image (ablation) ----
    per_sample = []
    for qid_label, grp in per_theta.groupby(["image_id", "label"]):
        qid, label = qid_label
        if label == 1:
            grp = grp.assign(sss_sample=[
                geo_mean([1.0 if r.rescued else EPS, r.iqp, r.sp]) for r in grp.itertuples()
            ])
            rescued_any = bool(grp["rescued"].any())
            pick = grp.sort_values(["rescued", "sss_sample"], ascending=[False, False]).iloc[0]
        else:
            grp = grp.assign(sss_sample=[geo_mean([r.iqp, r.sp]) for r in grp.itertuples()])
            rescued_any = False
            pick = grp.sort_values("sss_sample", ascending=False).iloc[0]
        per_sample.append(
            {
                "image_id": qid,
                "label": int(label),
                "kind": pick["kind"],
                "theta_kernel": int(pick["kernel"]),
                "theta_sigma": float(pick["sigma"]),
                "sss_sample": float(pick["sss_sample"]),
                "rescued": rescued_any,
                "fooled_before": bool(pick["fooled_before"]),
                "fooled_after": bool(pick["fooled_after"]),
                "iqp": float(pick["iqp"]),
                "sp": float(pick["sp"]),
            }
        )
    per_sample_df = pd.DataFrame(per_sample)
    per_sample_df.to_csv(root / f"sss_per_sample{tag}.csv", index=False, encoding="utf-8")

    atk_ps = per_sample_df[per_sample_df["label"] == 1]
    atk_ps_fooled = atk_ps[atk_ps["fooled_before"]]
    theta_star["per_sample_delta_asr"] = (
        float(atk_ps_fooled["rescued"].mean()) if len(atk_ps_fooled) else float("nan")
    )
    theta_star["per_sample_sigma_hist"] = (
        per_sample_df["theta_sigma"].value_counts().sort_index().to_dict()
    )
    with open(root / f"sss_best_theta{tag}.json", "w", encoding="utf-8") as f:
        json.dump(theta_star, f, ensure_ascii=False, indent=2)

    print("\n=== SSS set-level grid (sorted by SSS) ===")
    cols = ["kernel", "sigma", "delta_asr", "iqp", "sp", "sss", "clean_acc_drop"]
    print(grid[cols].to_string(index=False))
    print(f"\ntheta* (set-level): kernel={theta_star['kernel']} sigma={theta_star['sigma']} SSS={theta_star['sss']:.4f}")
    print(f"per-sample delta_ASR (adaptive theta): {theta_star['per_sample_delta_asr']:.4f}")
    print(f"Saved {root / f'sss_grid_search{tag}.csv'}")
    print(f"Saved {root / f'sss_best_theta{tag}.json'}")
    return grid, per_sample_df, theta_star


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SSS sweet-spot denoise search for TC-VAD stage-2")
    parser.add_argument("--experiment_root", default=str(EXPERIMENT_ROOT))
    parser.add_argument("--visual_dir", default=None)
    parser.add_argument("--stage1_csv", default=None)
    parser.add_argument("--log_path", default=None)
    parser.add_argument("--vqa_ckpt", default=None)
    parser.add_argument("--answer_list", default=None)
    parser.add_argument("--caption_ckpt", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--method", default="rotate_5")
    parser.add_argument("--kernels", nargs="+", type=int, default=None)
    parser.add_argument("--sigmas", nargs="+", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="debug: only first N mid-risk rows")
    parser.add_argument("--out_suffix", default=None)
    args = parser.parse_args()
    run_sss(
        experiment_root=args.experiment_root,
        visual_dir=args.visual_dir,
        stage1_csv=args.stage1_csv,
        log_path=args.log_path,
        vqa_ckpt=args.vqa_ckpt,
        answer_list=args.answer_list,
        caption_ckpt=args.caption_ckpt,
        device=args.device,
        method=args.method,
        kernels=args.kernels,
        sigmas=args.sigmas,
        seed=args.seed,
        limit=args.limit,
        out_suffix=args.out_suffix,
    )
