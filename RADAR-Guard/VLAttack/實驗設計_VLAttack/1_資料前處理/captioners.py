"""Caption backends (BLIP / ViT-GPT2 / GIT-large-COCO / gemma3 via LLM-API) with a disk cache.

Every backend maps a list of PIL images to a list of strings. Captions are cached
by pixel hash, so the same image (e.g. the clean photo shared by several
questions, or a +5 degree view reused across rotation draws) is captioned once.
"""
import base64
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import torch
import torch.nn.functional as F

from common import BLIP_ROOT, CACHE_ROOT, image_hash, png_bytes

LLM_API_URL = os.getenv("AIAGENT_BASE_URL", "http://140.128.36.108:8025").rstrip("/")
GEMMA_MODEL = os.getenv("AIAGENT_VISION_MODEL", "gemma3:4b")
GEMMA_PROMPT = "Describe this image in one short sentence."
NUM_BEAMS = 3
MAX_LEN = 30


class CaptionCache:
    def __init__(self, name):
        CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        self.path = CACHE_ROOT / f"{name}.jsonl"
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
                    self.data[rec["k"]] = rec["c"]
        self._lock = threading.Lock()
        self._fh = open(self.path, "a", encoding="utf-8")

    def get(self, key):
        return self.data.get(key)

    def put(self, key, caption):
        with self._lock:
            self.data[key] = caption
            self._fh.write(json.dumps({"k": key, "c": caption}, ensure_ascii=False) + "\n")
            self._fh.flush()


class Captioner:
    name = "base"
    batch_size = 8

    def __init__(self, device):
        self.device = device
        self.cache = CaptionCache(self.name)

    def _generate(self, images):
        raise NotImplementedError

    def caption(self, images):
        keys = [image_hash(im) for im in images]
        todo = {}
        for k, im in zip(keys, images):
            if self.cache.get(k) is None and k not in todo:
                todo[k] = im
        items = list(todo.items())
        for i in range(0, len(items), self.batch_size):
            chunk = items[i:i + self.batch_size]
            caps = self._generate([im for _, im in chunk])
            for (k, _), c in zip(chunk, caps):
                self.cache.put(k, c)
        return [self.cache.get(k) for k in keys]


class BLIPCaptioner(Captioner):
    name = "blip"

    def __init__(self, device):
        super().__init__(device)
        from evaluate_best_stage1_new import load_caption_model

        ckpt = BLIP_ROOT / "checkpoints" / "model_base_caption_capfilt_large.pth"
        self.model, self.processor = load_caption_model(str(ckpt), device)

    @torch.no_grad()
    def _generate(self, images):
        x = torch.stack([self.processor(im.convert("RGB")) for im in images]).to(self.device)
        return [t.strip() for t in self.model.generate(x, sample=False, num_beams=NUM_BEAMS)]


class ViTGPT2Captioner(Captioner):
    name = "vitgpt2"

    def __init__(self, device):
        super().__init__(device)
        from transformers import AutoTokenizer, ViTImageProcessor, VisionEncoderDecoderModel

        repo = "nlpconnect/vit-gpt2-image-captioning"
        self.model = VisionEncoderDecoderModel.from_pretrained(repo).to(device).eval()
        self.processor = ViTImageProcessor.from_pretrained(repo)
        self.tokenizer = AutoTokenizer.from_pretrained(repo)
        print(f"Loaded {repo}")

    @torch.no_grad()
    def _generate(self, images):
        px = self.processor(images=[im.convert("RGB") for im in images], return_tensors="pt").pixel_values
        out = self.model.generate(px.to(self.device), max_length=MAX_LEN, num_beams=NUM_BEAMS)
        return [t.strip() for t in self.tokenizer.batch_decode(out, skip_special_tokens=True)]


class GITCaptioner(Captioner):
    name = "git"
    # In transformers 4.26 GIT tiles (not interleaves) visual features across beams,
    # so batched beam search pairs beams with the wrong image. One image per call.
    batch_size = 1

    def __init__(self, device):
        super().__init__(device)
        from transformers import AutoModelForCausalLM, AutoProcessor

        repo = "microsoft/git-large-coco"
        self.processor = AutoProcessor.from_pretrained(repo)
        self.model = AutoModelForCausalLM.from_pretrained(repo).to(device).eval()
        print(f"Loaded {repo}")

    @torch.no_grad()
    def _generate(self, images):
        px = self.processor(images=[im.convert("RGB") for im in images], return_tensors="pt").pixel_values
        # transformers 4.26 infers batch size 1 for input_ids when only pixel_values is
        # given, which breaks beam search for batches; pass the BOS column explicitly.
        bos = torch.full((px.shape[0], 1), self.model.config.bos_token_id, dtype=torch.long, device=self.device)
        out = self.model.generate(
            input_ids=bos, pixel_values=px.to(self.device), max_length=MAX_LEN, num_beams=NUM_BEAMS,
        )
        return [t.strip() for t in self.processor.batch_decode(out, skip_special_tokens=True)]


class Florence2Captioner(Captioner):
    """microsoft/Florence-2-large short caption (<CAPTION>), beam 3.

    Florence-2's custom generate path mis-pairs beams when several images are
    batched, so each call is one image. fp16 keeps the 0.77B model inside 8 GB.
    """

    name = "florence2"
    batch_size = 1
    prompt = "<CAPTION>"

    def __init__(self, device):
        super().__init__(device)
        # Florence-2's remote code needs transformers>=4.41 and timm.layers.
        # Those live beside the BLIP env (still on transformers 4.26) so the
        # attack code's transformers.file_utils import keeps working.
        side = Path(os.environ.get("LOCALAPPDATA", "")) / "vlattack_florence_pkgs"
        if side.is_dir() and str(side) not in sys.path:
            sys.path.insert(0, str(side))
        from transformers import AutoModelForCausalLM, AutoProcessor

        repo = os.getenv("FLORENCE_MODEL_ID", "microsoft/Florence-2-large-ft")
        self.processor = AutoProcessor.from_pretrained(repo, trust_remote_code=True)
        dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            repo, trust_remote_code=True, torch_dtype=dtype,
        ).to(device).eval()
        print(f"Loaded {repo} dtype={dtype}", flush=True)

    @torch.no_grad()
    def _generate(self, images):
        texts = []
        for im in images:
            im = im.convert("RGB")
            enc = self.processor(text=self.prompt, images=im, return_tensors="pt")
            input_ids = enc["input_ids"].to(self.device)
            pixels = enc["pixel_values"].to(self.device, dtype=self.model.dtype)
            out = self.model.generate(
                input_ids=input_ids,
                pixel_values=pixels,
                max_new_tokens=MAX_LEN,
                num_beams=NUM_BEAMS,
                do_sample=False,
            )
            raw = self.processor.batch_decode(out, skip_special_tokens=False)[0]
            parsed = self.processor.post_process_generation(
                raw, task=self.prompt, image_size=(im.width, im.height),
            )
            text = parsed.get(self.prompt, "") if isinstance(parsed, dict) else str(parsed)
            texts.append(" ".join(text.replace(self.prompt, " ").split()))
        return texts


class Gemma3Captioner(Captioner):
    """Vision LLM through the LLM-API proxy (Ollama /api/chat with an image)."""

    name = "gemma3"
    batch_size = 16
    workers = 4

    def __init__(self, device):
        super().__init__(device)
        self.session = requests.Session()
        print(f"Using {GEMMA_MODEL} via {LLM_API_URL}/chat")

    def _one(self, im):
        payload = {
            "model": GEMMA_MODEL,
            "stream": False,
            "messages": [{
                "role": "user",
                "content": GEMMA_PROMPT,
                "images": [base64.b64encode(png_bytes(im.convert("RGB"))).decode("ascii")],
            }],
            "options": {"temperature": 0, "seed": 42, "num_predict": 48},
        }
        delay = 5
        for attempt in range(8):
            try:
                r = self.session.post(f"{LLM_API_URL}/chat", json=payload, timeout=180)
                r.raise_for_status()
                text = r.json()["message"]["content"].strip()
                if text:
                    return " ".join(text.split())
            except (requests.RequestException, KeyError, ValueError) as err:
                print(f"[gemma3] attempt {attempt + 1} failed: {err}", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 120)
        raise RuntimeError("gemma3 captioning failed after retries")

    def _generate(self, images):
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            return list(pool.map(self._one, images))


BACKENDS = {
    "blip": BLIPCaptioner,
    "vitgpt2": ViTGPT2Captioner,
    "git": GITCaptioner,
    "gemma3": Gemma3Captioner,
    "florence2": Florence2Captioner,
}


def build_captioner(name, device):
    return BACKENDS[name](device)


class SentenceEncoder:
    """all-MiniLM-L6-v2 mean pooling, L2-normalised (same as the original stage 1)."""

    def __init__(self, device):
        from transformers import AutoModel, AutoTokenizer

        name = "sentence-transformers/all-MiniLM-L6-v2"
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(device).eval()
        self._cache = {}

    @torch.no_grad()
    def encode(self, texts):
        todo = [t for t in dict.fromkeys(texts) if t not in self._cache]
        for i in range(0, len(todo), 64):
            chunk = todo[i:i + 64]
            enc = self.tokenizer(chunk, padding=True, truncation=True, return_tensors="pt").to(self.device)
            tok = self.model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).expand(tok.size()).float()
            pooled = (tok * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            for t, v in zip(chunk, F.normalize(pooled, p=2, dim=-1).cpu()):
                self._cache[t] = v
        return torch.stack([self._cache[t] for t in texts])
