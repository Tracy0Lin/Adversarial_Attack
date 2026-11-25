"""Run one VLAttack step in the CUDA Python, then exit.

The website process does not have torch. It sends a JSON payload on stdin:

    python vlattack_gpu_worker.py status
    python vlattack_gpu_worker.py attack
    python vlattack_gpu_worker.py detect
    python vlattack_gpu_worker.py defend
"""
from __future__ import annotations

import base64
import json
import sys


def _status() -> dict:
    import torch

    cuda = bool(torch.cuda.is_available())
    name = torch.cuda.get_device_name(0) if cuda else None
    return {
        "cuda": cuda,
        "gpu_name": name,
        "device": "cuda" if cuda else None,
        "loaded": False,
        "python": sys.executable,
    }


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item) and not isinstance(value, (str, bytes)):
        try:
            return item()
        except Exception:
            return value
    return value


def main() -> int:
    # Model loading prints progress. Keep that off the JSON stdout.
    out = sys.stdout
    sys.stdout = sys.stderr
    command = sys.argv[1] if len(sys.argv) > 1 else "status"
    payload = json.load(sys.stdin) if command != "status" else {}
    if command == "status":
        result = _status()
    else:
        from vlattack_live import run_attack, run_defend, run_detect

        if command == "attack":
            result = run_attack(
                image_bytes=base64.b64decode(payload["image_b64"]),
                question=payload.get("question") or "",
                eps_pixel=float(payload.get("eps") or 0.125),
                alpha_pixel=float(payload.get("alpha") or 0.002),
                steps=int(payload.get("steps") or 100),
            )
        elif command == "detect":
            result = run_detect(payload["job_id"])
        elif command == "defend":
            result = run_defend(payload["job_id"])
        else:
            raise SystemExit(f"未知指令 {command}")
    json.dump(_plain(result), out)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
