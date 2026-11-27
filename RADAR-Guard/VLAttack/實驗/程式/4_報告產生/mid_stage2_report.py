"""Mid-risk stage-2 breakdown for BLIP caption + BLIP VQA (rows_scored.csv)."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent / "VQA" / "multi_captioner"


def victim_clean_pred(df):
    clean = df[df["label"] == 0][["image_id", "pred_ref"]].rename(columns={"pred_ref": "vc"})
    out = df.merge(clean, on="image_id", how="left")
    out["vc"] = out["vc"].where(out["label"] == 1, out["pred_ref"])
    return out


def mid_stage2_table(df, theta_det, theta_star):
    mid = victim_clean_pred(df[df["bin"] == "mid"].copy())
    flip = mid[f"pred_{theta_det}"] != mid["pred_ref"]
    mit = mid[f"pred_{theta_star}"]
    served = mit.where(~flip, mid["pred_ref"])
    atk = mid["label"] == 1
    cln = ~atk
    rows = {}
    for mask, name in [(atk, "attack"), (cln, "clean")]:
        sub = mid[mask]
        if sub.empty:
            continue
        vc = sub["vc"]
        ref = sub["pred_ref"]
        srv = served[mask]
        ok_before = ref == vc
        ok_after = srv == vc
        rows[name] = {
            "n": int(len(sub)),
            "ok_before": int(ok_before.sum()),
            "ok_after": int(ok_after.sum()),
            "wrong_before": int((~ok_before).sum()),
            "wrong_after": int((~ok_after).sum()),
        }
    return rows


def main():
    out = {}
    for exp in ("exp1", "exp2", "exp3"):
        summ = json.loads((ROOT / exp / "blip" / "summary.json").read_text(encoding="utf-8"))
        df = pd.read_csv(ROOT / exp / "blip" / "rows_scored.csv", dtype=str, keep_default_na=False)
        df["label"] = df["label"].astype(int)
        fp = summ["full_pipeline"]
        out[exp] = {
            "asr_before_pct": round(fp["asr_before"] * 100, 1),
            "effective_asr_pct": round(fp["effective_asr"] * 100, 1),
            "asr_drop_pct": round((fp["asr_before"] - fp["effective_asr"]) * 100, 1),
            "bins": summ["bins"],
            "full_tpr_pct": round(fp["TPR"] * 100, 1),
            "full_fpr_pct": round(fp["FPR"] * 100, 1) if fp.get("FPR") == fp.get("FPR") else None,
            "clean_service_pct": round(fp["clean_service_rate"] * 100, 1)
            if fp.get("clean_service_rate") == fp.get("clean_service_rate")
            else None,
            "mid_stage2": mid_stage2_table(df, summ["theta_det_mid"], summ["theta_star_mid"]),
            "theta_det": summ["theta_det_mid"],
            "theta_star": summ["theta_star_mid"],
        }
    (ROOT / "blip_three_exp_breakdown.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
