"""Quick pilot: how often does a non-BLIP victim answer differ on clean vs attacked PNGs?

Does not run the full defense pipeline; use run_probe + analyze --vqa-backend for that.
"""
import argparse
import json
from collections import Counter

from common import DATA_ROOT, experiment_rows
from vqa_backends import build_vqa


def probe_suffix(vqa_backend):
    return "" if vqa_backend == "blip" else f"_{vqa_backend}"


def run(exp, vqa_backend, device="cuda", limit=None):
    rows = experiment_rows(exp)
    if limit:
        rows = rows[:limit]
    vqa = build_vqa(vqa_backend, device)
    by_qid = {}
    for r in rows:
        by_qid.setdefault(r["image_id"], {})[r["label"]] = r

    stats = Counter()
    samples = []
    for qid, pair in by_qid.items():
        if 0 not in pair or 1 not in pair:
            continue
        from common import open_rgb

        clean, atk = pair[0], pair[1]
        q = clean["question"]
        ans_clean = vqa.predict(open_rgb(clean["path"]), q)
        ans_atk = vqa.predict(open_rgb(atk["path"]), q)
        fooled = ans_clean != ans_atk
        blip_fooled = bool(atk.get("log_attack_success"))
        stats["pairs"] += 1
        stats["victim_fooled"] += int(fooled)
        stats["blip_log_success"] += int(blip_fooled)
        stats["both"] += int(fooled and blip_fooled)
        stats["victim_only"] += int(fooled and not blip_fooled)
        stats["blip_only"] += int(not fooled and blip_fooled)
        if len(samples) < 8:
            samples.append({
                "question_id": qid,
                "question": q[:80],
                "victim_clean": ans_clean,
                "victim_atk": ans_atk,
                "blip_clean_pred": clean.get("original_prediction", ""),
                "victim_fooled": fooled,
                "blip_attack_success": blip_fooled,
            })

    n = stats["pairs"]
    report = {
        "experiment": exp,
        "vqa_backend": vqa_backend,
        "n_pairs": n,
        "victim_asr": stats["victim_fooled"] / n if n else None,
        "blip_log_asr": stats["blip_log_success"] / n if n else None,
        "agreement_both_fooled": stats["both"] / n if n else None,
        "victim_only_fooled": stats["victim_only"],
        "blip_only_fooled": stats["blip_only"],
        "samples": samples,
    }
    out = DATA_ROOT / "vqa_pilot" / f"{exp}_{vqa_backend}_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}, ensure_ascii=False, indent=2))
    print(f"Wrote {out}", flush=True)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--exp", default="exp2")
    p.add_argument("--vqa-backend", default="git", choices=("blip", "git", "gemma3"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=20, help="number of log rows (clean+atk pairs use half)")
    args = p.parse_args()
    run(args.exp, args.vqa_backend, device=args.device, limit=args.limit)
