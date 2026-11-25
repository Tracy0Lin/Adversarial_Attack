"""One uploaded image through the experiment stack.

Attack matches ``attack_vlattack.py`` / ``adv_attack_blip_vla.evaluate``:
BLIP-pretrain BSA PGD, L∞ on the VQA dataloader tensor
(normalize 0.5, clip [-1, 1], ε=0.125, α=0.002, 100 steps). The question
string is not edited. Success is black-box BLIP-VQA rank (k_test=128):
the answer changed.

Stage 1 matches ``extract_four_dimensional_features`` (rotate ±5° × 10,
BLIP caption, 4-D angular distance) and the P1 RBF SVM in
``train_p1_svms``, fit on the experiment-1 rotate_5 rows. A new photo is
not in those folds, so this is the fitted classifier, not an OOF score.

Stage 2 matches the reported rule: 5×5 Gaussian, σ=1.5 asks whether the
VQA answer flips, σ=0.8 is the image that is kept.
"""
from __future__ import annotations

import os
import random
import re
import sys
import threading
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import structural_similarity as compare_ssim
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode, gaussian_blur, to_pil_image, to_tensor

def _first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def _cleverhans_parent(root: Path) -> Path | None:
    """Directory that must be on sys.path so `import cleverhans.torch` works."""
    if (root / "cleverhans" / "torch" / "__init__.py").is_file():
        return root
    nested = root / "cleverhans"
    if (nested / "cleverhans" / "torch" / "__init__.py").is_file():
        return nested
    if (root / "torch" / "__init__.py").is_file() and (root / "__init__.py").is_file():
        return root.parent
    return None


def _root_candidates() -> list[Path]:
    env = os.environ.get("VLATTACK_ROOT")
    beside_site = Path(__file__).resolve().parent
    candidates = []
    if env:
        candidates.append(Path(env))
    candidates.extend(
        [
            Path("/workspace/Web/VLAttack"),
            beside_site / "VLAttack",
            beside_site,
            Path("/workspace/Web"),
            Path("/workspace/projects/couser/VLAttack"),
            Path(r"C:\Users\tracy\github\VLAttack"),
            Path(r"C:\Users\tracy_aboikg6\OneDrive\Desktop\program\VLAttack_zip"),
        ]
    )
    unique = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def _vlattack_root() -> Path:
    """Experiment directory that contains BLIP_attack, including the checkpoints."""
    candidates = _root_candidates()
    for candidate in candidates:
        if (candidate / "BLIP_attack" / "checkpoints" / "model_base_14M.pth").is_file():
            return candidate
    for candidate in candidates:
        if (candidate / "BLIP_attack").is_dir():
            return candidate
    return candidates[0]


def _locate(relative: tuple[str, ...], extras: list[Path] | None = None) -> Path:
    paths = [root.joinpath(*relative) for root in _root_candidates()]
    paths.extend(extras or [])
    return _first_existing(paths)


_CKPT_DIR_ENV = os.environ.get("VLATTACK_CHECKPOINTS_DIR", "").strip()
_CKPT_EXTRAS = [Path(_CKPT_DIR_ENV)] if _CKPT_DIR_ENV else []
_CKPT_EXTRAS.append(Path(r"C:\Users\tracy\github\VLAttack\BLIP_attack\checkpoints"))

# The three BLIP checkpoints (resolved from VLATTACK_CHECKPOINTS_DIR, VLATTACK_ROOT, or C:\Users\tracy\github\VLAttack).
PRETRAIN_WEIGHTS = _locate(
    ("BLIP_attack", "checkpoints", "model_base_14M.pth"),
    [d / "model_base_14M.pth" for d in _CKPT_EXTRAS]
    + [Path(r"C:\Users\tracy_aboikg6\OneDrive\Desktop\program\VLAttack_Data\models\model_base.pth")],
)
VQA_WEIGHTS = _locate(
    ("BLIP_attack", "checkpoints", "model_vqa.pth"),
    [d / "model_vqa.pth" for d in _CKPT_EXTRAS]
    + [Path(r"C:\Users\tracy_aboikg6\OneDrive\Desktop\program\VLAttack_Data\models\model_vqa.pth")],
)
CAPTION_WEIGHTS = _locate(
    ("BLIP_attack", "checkpoints", "model_base_caption_capfilt_large.pth"),
    [d / "model_base_caption_capfilt_large.pth" for d in _CKPT_EXTRAS],
)
ANSWER_LIST = _locate(
    ("BLIP_attack", "annotation", "answer_list.json"),
    [Path(r"C:\Users\tracy_aboikg6\OneDrive\Desktop\program\VLAttack_Data\annotation\answer_list.json")],
)
STAGE1_CSV = _locate(
    ("BLIP_attack", "output", "VQA", "exp1_mixed1000", "tcvad_stage1_4_academic_dataset.csv"),
    [
        Path(r"C:\Users\tracy\github\VLAttack_final\BLIP_attack\output\VQA\exp1_mixed1000\tcvad_stage1_4_academic_dataset.csv"),
    ],
)
if PRETRAIN_WEIGHTS.is_file():
    BLIP_ROOT = PRETRAIN_WEIGHTS.parent.parent
    VLATTACK_ROOT = BLIP_ROOT.parent
else:
    VLATTACK_ROOT = _vlattack_root()
    BLIP_ROOT = VLATTACK_ROOT / "BLIP_attack"
CLEVERHANS_ROOT = VLATTACK_ROOT / "cleverhans"
def _code_blip_root() -> Path:
    """BLIP checkout that contains both models/ and configs/bert_config.json."""
    for root in _root_candidates():
        base = root / "BLIP_attack"
        if (base / "models" / "blip_vqa.py").is_file() and (base / "configs" / "bert_config.json").is_file():
            return base
    found = _locate(("BLIP_attack", "models", "blip_vqa.py"))
    if found.is_file():
        return found.parent.parent
    return BLIP_ROOT


_EVAL_SCRIPT = _locate(("BLIP_attack", "output", "VQA", "steps100_alpha0.002", "evaluate_best_stage1_new.py"))
EXPERIMENT_DIR = _EVAL_SCRIPT.parent if _EVAL_SCRIPT.is_file() else BLIP_ROOT / "output" / "VQA" / "steps100_alpha0.002"
CODE_BLIP_ROOT = _code_blip_root()
def _find_cleverhans_parent() -> Path | None:
    for root in (VLATTACK_ROOT, Path(__file__).resolve().parent, *_root_candidates()):
        parent = _cleverhans_parent(root)
        if parent is not None:
            return parent
    return None


def _use_local_cleverhans() -> None:
    parent = _find_cleverhans_parent()
    marker = None if parent is None else parent / "cleverhans" / "torch" / "__init__.py"
    if parent is None or not marker.is_file():
        raise FileNotFoundError(
            "找不到實驗用的 cleverhans.torch。"
            f"已找過 {VLATTACK_ROOT / 'cleverhans' / 'torch'} 和 "
            f"{VLATTACK_ROOT / 'cleverhans' / 'cleverhans' / 'torch'}。"
            "VLATTACK_ROOT 請指到 /workspace/Web/VLAttack"
        )
    root = str(parent)
    sys.path[:] = [item for item in sys.path if item not in {root, str(VLATTACK_ROOT / "cleverhans"), str(CLEVERHANS_ROOT)}]
    sys.path.insert(0, root)
    loaded = sys.modules.get("cleverhans")
    loaded_file = str(getattr(loaded, "__file__", "") or "")
    if loaded is not None and not loaded_file.startswith(root):
        for name in list(sys.modules):
            if name == "cleverhans" or name.startswith("cleverhans."):
                del sys.modules[name]


_CLEVERHANS_PARENT = _find_cleverhans_parent()
for _path in (_CLEVERHANS_PARENT, VLATTACK_ROOT, EXPERIMENT_DIR, CODE_BLIP_ROOT / "output", CODE_BLIP_ROOT, BLIP_ROOT):
    if _path is not None and _path.is_dir():
        sys.path.insert(0, str(_path))

IMAGE_SIZE = 480
NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD = (0.5, 0.5, 0.5)
DEFAULT_EPS = 0.125
DEFAULT_ALPHA = 0.002
DEFAULT_STEPS = 100
K_TRANSFORMS = 10
FEATURE_METHOD = "rotate_5"

_LOCK = threading.Lock()
_ENGINE = None
_PRE_QUESTION = None


def pixel_budget(eps: float) -> float:
    """L∞ in [0, 1] pixels. Tensor ε is divided by the dataloader std (0.5)."""
    return float(eps) * NORM_STD[0]


class ExperimentEngine:
    def __init__(self):
        self.device = None
        self.white = None
        self.vqa = None
        self.answer_list = None
        self.answer_candidates = None
        self.embedder = None
        self.svm = None
        self.to_tensor = None

    def load(self):
        if self.white is not None:
            return
        required = {
            "BLIP pretrain model_base_14M.pth": PRETRAIN_WEIGHTS,
            "BLIP VQA model_vqa.pth": VQA_WEIGHTS,
            "BLIP caption model_base_caption_capfilt_large.pth": CAPTION_WEIGHTS,
            "answer_list.json": ANSWER_LIST,
        }
        missing = [f"{name}：{path}" for name, path in required.items() if not path.is_file()]
        if missing:
            raise FileNotFoundError("實驗檔案不在：\n" + "\n".join(missing))
        cwd = os.getcwd()
        os.chdir(CODE_BLIP_ROOT)
        _use_local_cleverhans()
        for path in (EXPERIMENT_DIR, CODE_BLIP_ROOT / "output", CODE_BLIP_ROOT, BLIP_ROOT):
            if path.is_dir() and str(path) not in sys.path:
                sys.path.insert(0, str(path))
        try:
            from evaluate_best_stage1_new import (
                FEATURE_COLS,
                CaptionEmbedder,
                load_caption_model,
                load_sentence_encoder,
            )
            from evaluate_stage2_gaussian import score_with_svm
            from models.blip_pretrain import blip_pretrain
            from models.blip_vqa import blip_vqa
            import pandas as pd
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.svm import SVC

            self._feature_cols = FEATURE_COLS
            self._score_with_svm = score_with_svm
            self.device = _require_cuda()
            print(f"loading BLIP pretrain {PRETRAIN_WEIGHTS}", flush=True)
            self.white = blip_pretrain(
                pretrained=str(PRETRAIN_WEIGHTS),
                image_size=IMAGE_SIZE,
                vit="base",
                vit_grad_ckpt=True,
                vit_ckpt_layer=0,
                queue_size=1,
            ).to(self.device).eval()
            print(f"loading BLIP VQA {VQA_WEIGHTS}", flush=True)
            self.vqa = blip_vqa(
                pretrained=str(VQA_WEIGHTS),
                image_size=IMAGE_SIZE,
                vit="base",
                vit_grad_ckpt=False,
                vit_ckpt_layer=0,
            ).to(self.device).eval()
            import json
            with open(ANSWER_LIST, "r", encoding="utf-8") as f:
                self.answer_list = json.load(f)
            cand = self.vqa.tokenizer(self.answer_list, padding="longest", return_tensors="pt").to(self.device)
            cand.input_ids[:, 0] = self.vqa.tokenizer.bos_token_id
            self.answer_candidates = cand
            caption_model, processor = load_caption_model(str(CAPTION_WEIGHTS), self.device)
            tokenizer, sent_model = load_sentence_encoder(self.device)
            self.embedder = CaptionEmbedder(caption_model, processor, tokenizer, sent_model, self.device)
            svm_loaded = False
            site_dir = Path(__file__).resolve().parent
            svm_candidates = [
                Path(os.environ["VLATTACK_SVM_MODEL"]) if os.environ.get("VLATTACK_SVM_MODEL") else None,
                site_dir / "models_params" / "stage1_rbf_svm_rotate5_exp1.joblib",
                site_dir.parents[1] / "實驗設計_VLAttack" / "6_模型參數" / "stage1_rbf_svm_rotate5_exp1.joblib",
            ]
            for svm_path in svm_candidates:
                if svm_path is not None and svm_path.is_file():
                    try:
                        import joblib
                        import warnings as _w
                        with _w.catch_warnings():
                            _w.simplefilter("error")
                            self.svm = joblib.load(svm_path)
                        svm_loaded = True
                        print(f"Loaded pre-trained Stage 1 SVM from {svm_path}", flush=True)
                        break
                    except Exception as exc:
                        print(f"Fallback to CSV fit (could not load {svm_path}: {exc})", flush=True)
            if not svm_loaded:
                if not STAGE1_CSV.is_file():
                    raise FileNotFoundError(f"找不到已訓練 .joblib 且找不到 {STAGE1_CSV}")
                table = pd.read_csv(STAGE1_CSV)
                sub = table[table["method_type"] == FEATURE_METHOD]
                if sub.empty:
                    raise RuntimeError(f"{STAGE1_CSV} 沒有 {FEATURE_METHOD} 列，SVM 無法照實驗擬合")
                clf = Pipeline(
                    [("scaler", StandardScaler()), ("svm", SVC(kernel="rbf", C=1.0, class_weight="balanced"))]
                )
                clf.fit(sub[FEATURE_COLS].values, sub["label"].astype(int).values)
                self.svm = clf
                print(f"P1 SVM fit on {len(sub)} {FEATURE_METHOD} rows from exp1", flush=True)
        finally:
            os.chdir(cwd)
        self.to_tensor = transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(NORM_MEAN, NORM_STD),
            ]
        )

    def encode(self, pil: Image.Image) -> torch.Tensor:
        return self.to_tensor(pil.convert("RGB")).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def answer(self, tensor: torch.Tensor, question: str) -> str:
        question = _pre_question(question)
        answer_ids, _, _ = self.vqa(
            tensor, [question], self.answer_candidates, train=False, inference="rank", k_test=128,
        )
        return self.answer_list[int(answer_ids[0])]

    def attack(self, pil: Image.Image, question: str, eps: float, alpha: float, steps: int):
        _use_local_cleverhans()
        import cleverhans.torch.attacks.BLIP.projected_gradient_descent as pgd

        with _LOCK:
            return self._attack(pgd, pil, question, eps, alpha, steps)

    def _attack(self, pgd, pil: Image.Image, question: str, eps: float, alpha: float, steps: int):
        self.load()
        question = _pre_question(question)
        if not question:
            raise ValueError("問題是空的")
        eps = float(eps)
        alpha = float(alpha)
        steps = int(steps)
        if not (0 < eps <= 0.5) or not (0 < alpha <= eps):
            raise ValueError(f"ε={eps}、α={alpha} 不在實驗尺度。預設 ε=0.125、α=0.002，而且 α 不能大於 ε。")
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        ori = self.encode(pil)
        model_fn = _pgd_fn(self.white, question)
        with torch.no_grad():
            img_list, txt_list = self.white.Gen_feats(ori, question)
            ori_img_feats = torch.cat(img_list, axis=0)
            ori_txt_feats = torch.cat(txt_list, axis=0)
        torch.set_grad_enabled(True)
        try:
            adv, losses = pgd.projected_gradient_descent(
                model_fn, ori.clone(), eps, alpha, steps,
                np.inf, clip_min=-1.0, clip_max=1.0,
                y=[ori_txt_feats, ori_img_feats, None, None, None],
                time=0, ori_x=ori, method="BSA",
            )
        finally:
            torch.set_grad_enabled(False)
        clean_pil = tensor_to_pil(ori)
        adv_pil = tensor_to_pil(adv)
        clean_ans = self.answer(ori, question)
        adv_ans = self.answer(adv, question)
        history = [{"step": i + 1, "loss": float(v)} for i, v in enumerate(losses)]
        return {
            "question": question,
            "clean_pil": clean_pil,
            "adv_pil": adv_pil,
            "original_answer": clean_ans,
            "adversarial_answer": adv_ans,
            "attack_success": clean_ans != adv_ans,
            "eps": eps,
            "alpha": alpha,
            "steps": steps,
            "history": history,
            "ssim": ssim_pil(clean_pil, adv_pil),
            "device": str(self.device),
            "gpu_name": _gpu_name(),
        }

    def score(self, pil: Image.Image, image_id: str) -> dict:
        with _LOCK:
            return self._score(pil, image_id)

    def _score(self, pil: Image.Image, image_id: str) -> dict:
        from evaluate_best_stage1_new import extract_four_dimensional_features

        self.load()
        feats = extract_four_dimensional_features(
            self.embedder, pil.convert("RGB"), FEATURE_METHOD, K_TRANSFORMS, 42, image_id, "live",
        )
        risk, bin_name = self._score_with_svm(self.svm, feats)
        action = {"low": "放行", "high": "攔截", "mid": "送 Stage 2"}[bin_name]
        return {
            "risk_score": float(risk),
            "risk_bin": bin_name,
            "action": action,
            "d_center": feats["d_center"],
            "d_iqr": feats["d_iqr"],
            "d_tail": feats["d_tail"],
            "d_q95": feats["d_q95"],
            "k": K_TRANSFORMS,
            "method": "rotate_5_caption_svm",
        }

    def blur(self, pil: Image.Image, sigma: float) -> Image.Image:
        tensor = to_tensor(pil.convert("RGB"))
        blurred = gaussian_blur(tensor, kernel_size=[5, 5], sigma=[float(sigma), float(sigma)])
        return to_pil_image(blurred.clamp(0.0, 1.0))

    def answer_pil(self, pil: Image.Image, question: str) -> str:
        with _LOCK:
            self.load()
            return self.answer(self.encode(pil), question)


def _gpu_name() -> str | None:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return None


def _require_cuda() -> torch.device:
    requested = os.environ.get("VLATTACK_DEVICE", "cuda").strip() or "cuda"
    if requested == "cpu":
        print("VLAttack engine on CPU (VLATTACK_DEVICE=cpu)", flush=True)
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "這個 Python 沒有 CUDA。"
            "若無 GPU，請設定環境變數 VLATTACK_DEVICE=cpu 以 CPU 模式執行。"
        )
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    name = torch.cuda.get_device_name(0)
    print(f"VLAttack engine on CUDA: {name}", flush=True)
    return torch.device("cuda")


def engine_status() -> dict:
    engine = get_engine()
    return {
        "cuda": bool(torch.cuda.is_available()),
        "gpu_name": _gpu_name(),
        "device": str(engine.device) if engine.device is not None else None,
        "loaded": engine.white is not None,
    }


def get_engine() -> ExperimentEngine:
    global _ENGINE
    with _LOCK:
        if _ENGINE is None:
            _ENGINE = ExperimentEngine()
        return _ENGINE


def _pre_question(question: str, max_ques_words: int = 50) -> str:
    """Same cleaning as BLIP data.utils.pre_question."""
    question = re.sub(r"([.!\"()*#:;~])", "", question.lower())
    question = question.rstrip(" ")
    question_words = question.split(" ")
    if len(question_words) > max_ques_words:
        question = " ".join(question_words[:max_ques_words])
    return question


def _pgd_fn(model, question: str):
    """Same inner step as attack_vlattack.pgd_fn / Adv_attack.pgd_attack."""

    def _fn(x):
        if random.random() > 0.5:
            orig = x.shape[-2:]
            rnd = random.randint(int(orig[0] * 0.9), orig[0])
            resized = F.interpolate(x, size=(rnd, rnd), mode="bilinear")
            pad_top = random.randint(0, orig[0] - rnd)
            pad_left = random.randint(0, orig[1] - rnd)
            x = F.pad(resized, (pad_left, orig[1] - rnd - pad_left, pad_top, orig[0] - rnd - pad_top))
        img_feats_list, txt_feats_list = model.Gen_feats(x, question)
        return [torch.cat(txt_feats_list, axis=0), torch.cat(img_feats_list, axis=0)]

    return _fn


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    x = tensor.detach().float().cpu().squeeze(0)
    x = (x * NORM_STD[0] + NORM_MEAN[0]).clamp(0, 1)
    arr = (x.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def ssim_pil(a: Image.Image, b: Image.Image) -> float:
    x = np.asarray(a.convert("RGB")).astype(np.float32) / 255.0
    y = np.asarray(b.convert("RGB")).astype(np.float32) / 255.0
    if x.shape != y.shape:
        y_img = b.convert("RGB").resize((a.width, a.height), Image.BILINEAR)
        y = np.asarray(y_img).astype(np.float32) / 255.0
    return float(compare_ssim(x, y, data_range=1.0, win_size=3, channel_axis=-1))
