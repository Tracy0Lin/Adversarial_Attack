"""Shared Florence-2-large-ft loader: VQA, caption, and differentiable loss for PGD."""
import os
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

FLORENCE_REPO = os.getenv("FLORENCE_MODEL_ID", "microsoft/Florence-2-large-ft")
CAPTION_TASK = "<CAPTION>"
SIDE_PKGS = Path(os.environ.get("LOCALAPPDATA", "")) / "vlattack_florence_pkgs"


def ensure_florence_deps():
    if SIDE_PKGS.is_dir() and str(SIDE_PKGS) not in sys.path:
        sys.path.insert(0, str(SIDE_PKGS))


def normalize_answer(text):
    if not text:
        return ""
    line = str(text).strip().split("\n")[0].strip()
    line = re.sub(r"\s+", " ", line)
    return line.lower().rstrip(".")


class FlorenceEngine:
    def __init__(self, device="cuda"):
        ensure_florence_deps()
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.device = device
        self.processor = AutoProcessor.from_pretrained(FLORENCE_REPO, trust_remote_code=True)
        dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            FLORENCE_REPO, trust_remote_code=True, torch_dtype=dtype,
        ).to(device).eval()
        self.dtype = dtype
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        if getattr(self.model, "config", None) is not None:
            self.model.config.use_cache = False
        lang = getattr(self.model, "language_model", None)
        if lang is not None and getattr(lang, "config", None) is not None:
            lang.config.use_cache = False
        print(f"FlorenceEngine: {FLORENCE_REPO} on {device} ({dtype})", flush=True)

    def _pixels_from_pil(self, pil_img):
        enc = self.processor(images=pil_img.convert("RGB"), return_tensors="pt")
        return enc["pixel_values"].to(self.device, dtype=self.dtype)

    def _decode_task(self, generated_ids, task_prompt, image_size):
        raw = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(
            raw, task=task_prompt, image_size=image_size,
        )
        if isinstance(parsed, dict):
            return parsed.get(task_prompt, "") or next(iter(parsed.values()), "")
        return str(parsed)

    @torch.no_grad()
    def caption(self, pil_img, max_new_tokens=30, num_beams=3):
        im = pil_img.convert("RGB")
        enc = self.processor(text=CAPTION_TASK, images=im, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        pixels = enc["pixel_values"].to(self.device, dtype=self.dtype)
        out = self.model.generate(
            input_ids=input_ids,
            pixel_values=pixels,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
        )
        text = self._decode_task(out, CAPTION_TASK, (im.width, im.height))
        return normalize_answer(text) or " ".join(str(text).split())

    @torch.no_grad()
    def vqa_on_pixels(self, input_ids, pixel_values, question, image_size, max_new_tokens=24, num_beams=3):
        q = question.strip()
        out = self.model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
        )
        text = self._decode_task(out, q, image_size)
        return normalize_answer(text) or normalize_answer(str(text))

    def pixels_to_pil(self, pixel_values):
        """Invert CLIP-style normalization from processor pixel_values -> PIL."""
        from torchvision.transforms.functional import to_pil_image

        ip = self.processor.image_processor
        x = pixel_values.detach().float().cpu()[0]
        mean = torch.tensor(ip.image_mean, dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor(ip.image_std, dtype=torch.float32).view(3, 1, 1)
        x = (x * std + mean).clamp(0.0, 1.0)
        x = (x * 255.0).to(torch.uint8)
        return to_pil_image(x)

    @torch.no_grad()
    def vqa(self, pil_img, question, max_new_tokens=24, num_beams=3):
        im = pil_img.convert("RGB")
        q = question.strip()
        enc = self.processor(text=q, images=im, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        pixels = enc["pixel_values"].to(self.device, dtype=self.dtype)
        out = self.model.generate(
            input_ids=input_ids,
            pixel_values=pixels,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
        )
        text = self._decode_task(out, q, (im.width, im.height))
        return normalize_answer(text) or normalize_answer(str(text))

    def encode_vqa(self, pil_img, question):
        """Fixed prompt tokens + baseline pixels for PGD (pixels may be replaced with adv)."""
        im = pil_img.convert("RGB")
        q = question.strip()
        enc = self.processor(text=q, images=im, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        pixels = enc["pixel_values"].to(self.device, dtype=self.dtype)
        return input_ids, pixels

    def vqa_teacher_loss(self, pixel_values, input_ids, target_answer):
        """Differentiable loss: negative log-likelihood of target_answer tokens (for PGD)."""
        ans = target_answer.strip()
        labels = self.processor.tokenizer(
            ans, return_tensors="pt", add_special_tokens=False,
        ).input_ids.to(self.device)
        if labels.numel() == 0:
            return pixel_values.sum() * 0.0
        # Teacher forcing: decoder targets = answer token ids
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            labels=labels,
            use_cache=False,
        )
        if hasattr(outputs, "loss") and outputs.loss is not None:
            loss = outputs.loss
            del outputs
            return loss
        # Fallback: manual CE on logits if loss not returned
        logits = outputs.logits
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )

    def pixel_values_for_attack(self, pil_img):
        px = self._pixels_from_pil(pil_img)
        return px.detach().clone().requires_grad_(True)
