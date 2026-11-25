"""Live page wired to the experiment stack in vlattack_engine.py.

1) BLIP-pretrain BSA PGD. The question text is not edited.
2) Black-box BLIP-VQA. Success means the answer changed.
3) Stage 1: rotate ±5° × 10, BLIP caption, 4-D angular distance, P1 SVM.
4) Stage 2: 5×5 Gaussian. σ=1.5 checks an answer flip. σ=0.8 is the kept image.
"""
from __future__ import annotations

import io
import json
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "static" / "results" / "live"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

SSS_KERNEL = 5
SSS_SIGMA = 0.8
DET_SIGMA = 1.5

JOBS: dict[str, dict] = {}


def _job_file(job_id: str) -> Path:
    return RESULT_DIR / f"{job_id}.json"


def _store_job(job_id: str, job: dict) -> None:
    JOBS[job_id] = job
    _job_file(job_id).write_text(json.dumps(job), encoding="utf-8")


def _load_job(job_id: str) -> dict | None:
    cached = JOBS.get(job_id)
    if cached is not None:
        return cached
    path = _job_file(job_id)
    if not path.is_file():
        return None
    job = json.loads(path.read_text(encoding="utf-8"))
    JOBS[job_id] = job
    return job


def _pil_to_np(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB")
    return np.asarray(img).astype(np.float32) / 255.0


def _np_to_pil(arr: np.ndarray) -> Image.Image:
    x = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(x, mode="RGB")


def _save(pil: Image.Image, name: str) -> str:
    path = RESULT_DIR / name
    pil.save(path, format="PNG")
    return f"/static/results/live/{name}"


def _residual_preview(x0: np.ndarray, adv: np.ndarray, eps: float = 32 / 255) -> Image.Image:
    delta = np.abs(adv - x0).mean(axis=2)
    glow = np.clip(delta / max(float(eps), 1e-8), 0.0, 1.0)
    rgb = np.stack([glow, glow * 0.18, glow * 0.12], axis=2)
    return _np_to_pil(rgb)


def run_attack(image_bytes: bytes, question: str, eps_pixel: float, alpha_pixel: float, steps: int) -> dict:
    src = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    eps = float(eps_pixel)
    alpha = float(alpha_pixel)
    steps = int(max(1, min(int(steps), 100)))
    if eps > 1.0 or alpha > 1.0:
        raise ValueError("ε、α 要用實驗的張量尺度（預設 0.125 和 0.002），不是像素。請重新整理頁面。")
    from vlattack_engine import get_engine, pixel_budget

    out = get_engine().attack(src, question, eps, alpha, steps)
    job_id = uuid.uuid4().hex[:12]
    clean_url = _save(out["clean_pil"], f"{job_id}_ori.png")
    adv_url = _save(out["adv_pil"], f"{job_id}_atk.png")
    diff_url = _save(
        _residual_preview(_pil_to_np(out["clean_pil"]), _pil_to_np(out["adv_pil"]), pixel_budget(out["eps"])),
        f"{job_id}_diff.png",
    )
    _store_job(job_id, {
        "question": out["question"],
        "clean_path": str(RESULT_DIR / f"{job_id}_ori.png"),
        "adv_path": str(RESULT_DIR / f"{job_id}_atk.png"),
        "ssim": out["ssim"],
        "eps": out["eps"],
        "alpha": out["alpha"],
        "steps": out["steps"],
        "history": out["history"],
        "original_answer": out["original_answer"],
        "adversarial_answer": out["adversarial_answer"],
        "attack_success": out["attack_success"],
    })
    return {
        "job_id": job_id,
        "question": out["question"],
        "original_url": clean_url,
        "adv_url": adv_url,
        "diff_url": diff_url,
        "ssim": out["ssim"],
        "linf": out["eps"],
        "alpha": out["alpha"],
        "steps": out["steps"],
        "history": out["history"],
        "original_answer": out["original_answer"],
        "adversarial_answer": out["adversarial_answer"],
        "attack_success": out["attack_success"],
        "pipeline": "blip_pretrain_bsa_pgd + blip_vqa",
        "device": out.get("device"),
        "gpu_name": out.get("gpu_name"),
    }


def run_detect(job_id: str) -> dict:
    job = _load_job(job_id)
    if not job:
        raise KeyError(job_id)
    adv = Image.open(job["adv_path"]).convert("RGB")
    clean = Image.open(job["clean_path"]).convert("RGB")
    from vlattack_engine import get_engine

    engine = get_engine()
    atk = engine.score(adv, f"{job_id}:atk")
    cln = engine.score(clean, f"{job_id}:ori")
    job["detect"] = {"attack": atk, "clean": cln}
    _store_job(job_id, job)
    return {
        "job_id": job_id,
        "adv_url": f"/static/results/live/{job_id}_atk.png",
        "attack": atk,
        "clean": cln,
    }


def run_defend(job_id: str) -> dict:
    job = _load_job(job_id)
    if not job:
        raise KeyError(job_id)
    from vlattack_engine import get_engine, ssim_pil

    engine = get_engine()
    clean = Image.open(job["clean_path"]).convert("RGB")
    adv = Image.open(job["adv_path"]).convert("RGB")
    question = job["question"]
    purified = engine.blur(adv, SSS_SIGMA)
    probe = engine.blur(adv, DET_SIGMA)
    pur_url = _save(purified, f"{job_id}_sss.png")
    kept_answer = engine.answer_pil(purified, question)
    probe_answer = engine.answer_pil(probe, question)
    adv_answer = job["adversarial_answer"]
    flipped = probe_answer != adv_answer
    risk_adv = (job.get("detect") or {}).get("attack") or engine.score(adv, f"{job_id}:atk")
    bin_name = risk_adv["risk_bin"]
    if bin_name == "high":
        flagged = True
        decision = "Stage 1 高風險，直接攔截"
    elif bin_name == "low":
        flagged = False
        decision = "Stage 1 低風險，放行"
    else:
        flagged = flipped
        decision = (
            f"Stage 2：σ=1.5 後答案 {adv_answer} → {probe_answer}，"
            + ("判定為攻擊" if flipped else "未翻轉，放行")
        )
    job["defend"] = {"purified_url": pur_url, "flagged": flagged, "kept_answer": kept_answer}
    _store_job(job_id, job)
    return {
        "job_id": job_id,
        "purified_url": pur_url,
        "original_url": f"/static/results/live/{job_id}_ori.png",
        "adv_url": f"/static/results/live/{job_id}_atk.png",
        "risk_before": risk_adv,
        "ssim_to_clean": ssim_pil(clean, purified),
        "flip_probe": flipped,
        "original_answer": job["original_answer"],
        "adversarial_answer": adv_answer,
        "probe_answer": probe_answer,
        "kept_answer": kept_answer,
        "theta_star": {"kernel": SSS_KERNEL, "sigma": SSS_SIGMA},
        "theta_det": {"kernel": SSS_KERNEL, "sigma": DET_SIGMA},
        "end_to_end_flagged": flagged,
        "decision": decision,
    }

