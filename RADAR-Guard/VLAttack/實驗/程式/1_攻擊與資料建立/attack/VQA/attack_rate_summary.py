"""Compute and persist attack success rate (ASR) from attack_dual_defense_log.json entries."""
import json
import os


def compute_attack_rate_summary(logs):
    if not logs:
        return None

    n = len(logs)
    success = sum(1 for item in logs if item.get("attack_success"))
    orig_asr = success / n if n else 0.0
    denom = success if success > 0 else n

    defense_nodes = [
        "mild_defense",
        "strong_defense",
        "gaussian_mild",
        "gaussian_strong",
        "resize_gaussian",
    ]
    defense_rates = {}
    for name in defense_nodes:
        residual = sum(
            1
            for item in logs
            if item.get("attack_success") and item.get(name, {}).get("attack_success")
        )
        defense_rates[name] = {
            "residual_attack_count": residual,
            "residual_attack_rate_percent": round(residual / denom * 100, 2),
        }

    ssim_vals = [item["ssim"]["original_vs_attack"] for item in logs if "ssim" in item]
    avg_ssim = sum(ssim_vals) / len(ssim_vals) if ssim_vals else None

    return {
        "total_samples": n,
        "attack_success_count": success,
        "attack_success_rate": round(orig_asr, 6),
        "attack_success_rate_percent": round(orig_asr * 100, 2),
        "residual_asr_denominator": denom,
        "defense_residual_asr": defense_rates,
        "mean_ssim_original_vs_attack": round(avg_ssim, 4) if avg_ssim is not None else None,
    }


def save_attack_rate_summary(output_dir, logs):
    summary = compute_attack_rate_summary(logs)
    if summary is None:
        print("[ASR] No samples in log; skip summary.")
        return None

    json_path = os.path.join(output_dir, "attack_rate_summary.json")
    txt_path = os.path.join(output_dir, "attack_rate_summary.txt")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    lines = [
        "VLAttack VQA Attack Rate Summary",
        f"Total samples: {summary['total_samples']}",
        f"Attack success (no defense): {summary['attack_success_count']} / {summary['total_samples']}",
        f"Attack success rate (ASR): {summary['attack_success_rate_percent']:.2f}%",
    ]
    if summary["mean_ssim_original_vs_attack"] is not None:
        lines.append(f"Mean SSIM (original vs attack): {summary['mean_ssim_original_vs_attack']:.4f}")
    lines.append("")
    lines.append("Residual ASR under defense (among originally successful attacks):")
    for name, stats in summary["defense_residual_asr"].items():
        lines.append(
            f"  - {name}: {stats['residual_attack_count']}/{summary['residual_asr_denominator']} "
            f"({stats['residual_attack_rate_percent']:.2f}%)"
        )

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("=" * 48)
    print("\n".join(lines))
    print("=" * 48)
    print(f"ASR summary saved: {json_path}")
    print(f"ASR summary saved: {txt_path}")
    return summary
