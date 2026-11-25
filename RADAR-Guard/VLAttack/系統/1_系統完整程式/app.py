from flask import Flask, jsonify, redirect, render_template, request, url_for
from werkzeug.middleware.proxy_fix import ProxyFix
import base64
import json
import os
import subprocess
from pathlib import Path
from vlattack_bridge import METHOD_EFFECTS, get_sample, list_samples, load_catalog

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024

# 處理 Proxy 路徑問題
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route('/')
@app.route('/index.html')
def index():
    return render_template('index.html')

@app.route('/vlattack')
@app.route('/vlattack.html')
def vlattack():
    return redirect(url_for('vlattack_live'))

@app.route('/vlattack-live')
@app.route('/vlattack-live.html')
def vlattack_live():
    return render_template('vlattack_live.html')

@app.route('/vlattack-cases')
@app.route('/vlattack-cases.html')
def vlattack_cases():
    return render_template('vlattack_cases.html')

@app.route('/results')
@app.route('/results.html')
def results():
    return render_template('results.html')

ROOT = Path(__file__).resolve().parent
GPU_PYTHON = os.environ.get("VLATTACK_GPU_PYTHON", "")


def _gpu_python() -> str | None:
    """Use the remote CUDA env when this website Python has no torch."""
    explicit = os.environ.get("VLATTACK_PYTHON", "").strip()
    if explicit.lower() in {"0", "local", "inprocess"}:
        return None
    if explicit:
        return explicit
    if os.name == "nt" or not GPU_PYTHON or not os.path.isfile(GPU_PYTHON):
        return None
    try:
        import torch
        if torch.cuda.is_available():
            return None
    except Exception:
        return GPU_PYTHON
    return None


def _gpu_call(command: str, payload: dict | None = None) -> dict:
    python = _gpu_python()
    if not python:
        raise RuntimeError("沒有可呼叫的 GPU Python")
    proc = subprocess.run(
        [python, str(ROOT / "vlattack_gpu_worker.py"), command],
        input="{}" if payload is None else json.dumps(payload),
        text=True,
        capture_output=True,
        cwd=str(ROOT),
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "GPU 程序失敗").strip()
        raise RuntimeError(detail[-2000:])
    return json.loads(proc.stdout)


def _public_url(value):
    """Turn /static/... into the URL this server actually serves, including a proxy prefix."""
    if isinstance(value, str) and value.startswith("/static/"):
        return url_for("static", filename=value[len("/static/"):])
    if isinstance(value, list):
        return [_public_url(item) for item in value]
    if isinstance(value, dict):
        return {key: _public_url(item) for key, item in value.items()}
    return value


def _live():
    """Import the GPU stack only when a live route runs."""
    from vlattack_engine import engine_status
    from vlattack_live import run_attack, run_defend, run_detect
    return engine_status, run_attack, run_detect, run_defend


@app.route('/api/vlattack/catalog')
def vlattack_catalog():
    catalog = load_catalog()
    return jsonify(_public_url({
        "attack": catalog["attack"],
        "pipeline": catalog["pipeline"],
        "metrics": catalog["metrics"],
        "samples": list_samples(),
        "method_effects": METHOD_EFFECTS,
    }))


@app.route('/api/vlattack/sample/<sample_id>')
def vlattack_sample(sample_id):
    sample = get_sample(sample_id)
    if sample is None:
        return jsonify({"error": f"找不到樣本 {sample_id}"}), 404
    return jsonify(_public_url(sample))


@app.route('/api/vlattack/live/status')
def vlattack_live_status():
    try:
        if _gpu_python():
            return jsonify(_gpu_call("status"))
        engine_status, _, _, _ = _live()
        return jsonify(engine_status())
    except Exception as exc:
        return jsonify({"error": str(exc), "cuda": False, "loaded": False}), 500


@app.route('/api/vlattack/live/attack', methods=['POST'])
def vlattack_live_attack():
    file = request.files.get('image')
    if file is None or file.filename == '':
        return jsonify({"error": "請上傳或選擇圖片"}), 400
    try:
        image_bytes = file.read()
        question = request.form.get('question') or ''
        eps_pixel = float(request.form.get('eps') or 0.125)
        alpha_pixel = float(request.form.get('alpha') or 0.002)
        steps = int(request.form.get('steps') or 100)
        if _gpu_python():
            result = _gpu_call("attack", {
                "image_b64": base64.b64encode(image_bytes).decode("ascii"),
                "question": question,
                "eps": eps_pixel,
                "alpha": alpha_pixel,
                "steps": steps,
            })
        else:
            _, run_attack, _, _ = _live()
            result = run_attack(
                image_bytes=image_bytes,
                question=question,
                eps_pixel=eps_pixel,
                alpha_pixel=alpha_pixel,
                steps=steps,
            )
        return jsonify(_public_url(result))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route('/api/vlattack/live/detect', methods=['POST'])
def vlattack_live_detect():
    job_id = (request.get_json(silent=True) or {}).get('job_id')
    if not job_id:
        return jsonify({"error": "缺少 job_id"}), 400
    try:
        if _gpu_python():
            result = _gpu_call("detect", {"job_id": job_id})
        else:
            _, _, run_detect, _ = _live()
            result = run_detect(job_id)
        return jsonify(_public_url(result))
    except KeyError:
        return jsonify({"error": "找不到這次攻擊，請先重新執行"}), 404
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route('/api/vlattack/live/defend', methods=['POST'])
def vlattack_live_defend():
    job_id = (request.get_json(silent=True) or {}).get('job_id')
    if not job_id:
        return jsonify({"error": "缺少 job_id"}), 400
    try:
        if _gpu_python():
            result = _gpu_call("defend", {"job_id": job_id})
        else:
            _, _, _, run_defend = _live()
            result = run_defend(job_id)
        return jsonify(_public_url(result))
    except KeyError:
        return jsonify({"error": "找不到這次攻擊，請先重新執行"}), 404
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005, debug=False, use_reloader=False)
