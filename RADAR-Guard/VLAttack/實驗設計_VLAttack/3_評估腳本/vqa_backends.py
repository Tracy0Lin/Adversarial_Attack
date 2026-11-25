"""VQA answer backends.

- blip: official BLIP VQA (rank over answer list).
- git: *experimental* — GIT-large-COCO is a caption model; we prefix the VQA question
  in the text prompt and treat the continuation as the «answer» (not standard VQA v2).
- gemma3: vision LLM via LLM-API (optional).
"""
import base64
import hashlib
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

import requests
import torch

from common import DATA_ROOT, png_bytes

LLM_API_URL = os.getenv("AIAGENT_BASE_URL", "http://140.128.36.108:8025").rstrip("/")
GEMMA_MODEL = os.getenv("AIAGENT_VISION_MODEL", "gemma3:4b")
GEMMA_VQA_PROMPT = (
    "Look at the image and answer the question with a very short phrase only "
    "(one to five words). No full sentence.\nQuestion: {question}\nAnswer:"
)
GIT_VQA_PREFIX = "Question: {question} Short answer:"
MAX_GEN = 24
NUM_BEAMS = 3


def normalize_answer(text):
    if not text:
        return ""
    line = text.strip().split("\n")[0].strip()
    line = re.sub(r"\s+", " ", line)
    return line.lower().rstrip(".")


class VQACache:
    def __init__(self, name):
        root = DATA_ROOT / "vqa_cache"
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / f"{name}.jsonl"
        self.data = {}
        if self.path.is_file():
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    self.data[rec["k"]] = rec["a"]
        self._lock = threading.Lock()
        self._fh = open(self.path, "a", encoding="utf-8")

    def get(self, key):
        return self.data.get(key)

    def put(self, key, answer):
        with self._lock:
            self.data[key] = answer
            self._fh.write(json.dumps({"k": key, "a": answer}, ensure_ascii=False) + "\n")
            self._fh.flush()


class VQABackend:
    name = "base"

    def predict(self, pil_img, question):
        raise NotImplementedError


class BlipVQABackend(VQABackend):
    name = "blip"

    def __init__(self, device="cuda"):
        out = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(out))
        from sss_denoise import VQAScorer, vqa_config

        cfg = vqa_config()
        ann = Path(cfg["ann_root"]) / "answer_list.json"
        self._scorer = VQAScorer(cfg["pretrained"], ann, device, k_test=cfg.get("k_test", 128))

    def predict(self, pil_img, question):
        return self._scorer.predict(pil_img, question)


class GitPseudoVQABackend(VQABackend):
    """Caption model + question prefix (pilot only; not a trained VQA head)."""

    name = "git"

    def __init__(self, device="cuda"):
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.device = device
        repo = "microsoft/git-large-coco"
        self.processor = AutoProcessor.from_pretrained(repo)
        self.model = AutoModelForCausalLM.from_pretrained(repo).to(device).eval()
        self.cache = VQACache("git_vqa")
        print(f"GIT pseudo-VQA loaded ({repo})", flush=True)

    def _cache_key(self, pil_img, question):
        from common import image_hash

        q = question.strip().lower()
        return hashlib.md5(f"{image_hash(pil_img)}|{q}".encode()).hexdigest()

    @torch.no_grad()
    def predict(self, pil_img, question):
        key = self._cache_key(pil_img, question)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        im = pil_img.convert("RGB")
        prefix = GIT_VQA_PREFIX.format(question=question.strip())
        enc = self.processor(images=im, text=prefix, return_tensors="pt")
        px = enc.pixel_values.to(self.device)
        input_ids = enc.input_ids.to(self.device)
        prompt_len = input_ids.shape[1]
        out = self.model.generate(
            pixel_values=px,
            input_ids=input_ids,
            max_length=prompt_len + MAX_GEN,
            num_beams=NUM_BEAMS,
            early_stopping=True,
        )
        new_tokens = out[0, prompt_len:]
        raw = self.processor.decode(new_tokens, skip_special_tokens=True)
        ans = normalize_answer(raw) or normalize_answer(prefix)
        self.cache.put(key, ans)
        return ans


class Florence2VQABackend(VQABackend):
    name = "florence2"

    def __init__(self, device="cuda"):
        fv = Path(__file__).resolve().parent.parent / "florence_victim"
        sys.path.insert(0, str(fv))
        from florence_engine import FlorenceEngine

        self.engine = FlorenceEngine(device)
        self.cache = VQACache("florence2_vqa")
        print("Florence-2 VQA backend (florence_victim)", flush=True)

    def _cache_key(self, pil_img, question):
        from common import image_hash

        q = question.strip().lower()
        return hashlib.md5(f"{image_hash(pil_img)}|{q}".encode()).hexdigest()

    def predict(self, pil_img, question):
        key = self._cache_key(pil_img, question)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        ans = self.engine.vqa(pil_img, question)
        self.cache.put(key, ans)
        return ans


class Gemma3VQABackend(VQABackend):
    name = "gemma3"

    def __init__(self, device=None):
        self.session = requests.Session()
        self.cache = VQACache("gemma3_vqa")
        print(f"Gemma3 VQA via {LLM_API_URL}/chat model={GEMMA_MODEL}", flush=True)

    def _cache_key(self, pil_img, question):
        from common import image_hash

        q = question.strip().lower()
        return hashlib.md5(f"{image_hash(pil_img)}|{q}".encode()).hexdigest()

    def predict(self, pil_img, question):
        key = self._cache_key(pil_img, question)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        payload = {
            "model": GEMMA_MODEL,
            "stream": False,
            "messages": [{
                "role": "user",
                "content": GEMMA_VQA_PROMPT.format(question=question.strip()),
                "images": [base64.b64encode(png_bytes(pil_img.convert("RGB"))).decode("ascii")],
            }],
            "options": {"temperature": 0, "seed": 42, "num_predict": 24},
        }
        delay = 5
        for attempt in range(8):
            try:
                r = self.session.post(f"{LLM_API_URL}/chat", json=payload, timeout=180)
                r.raise_for_status()
                raw = r.json()["message"]["content"].strip()
                ans = normalize_answer(raw) or raw.lower()
                self.cache.put(key, ans)
                return ans
            except (requests.RequestException, KeyError, ValueError) as err:
                print(f"[gemma3-vqa] attempt {attempt + 1} failed: {err}", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 120)
        raise RuntimeError("gemma3 VQA failed after retries")


BACKENDS = {
    "blip": BlipVQABackend,
    "git": GitPseudoVQABackend,
    "gemma3": Gemma3VQABackend,
    "florence2": Florence2VQABackend,
}


def build_vqa(name, device="cuda"):
    if name not in BACKENDS:
        raise ValueError(f"unknown vqa backend {name}; choose from {list(BACKENDS)}")
    return BACKENDS[name](device)
