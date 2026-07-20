import json

feature_file = "data/coco/tcvad_features.json"
out_file = "data/coco/tcvad_detection_results.json"

tau = 0.25

with open(feature_file, "r", encoding="utf-8") as f:
    feats = json.load(f)

results = {}

for fname, item in feats.items():
    mean_j = item["S_jaccard"]["mean"]
    std_j = item["S_jaccard"]["std"]
    mean_t = item["S_textcos"]["mean"]
    std_t = item["S_textcos"]["std"]

    mean_lpips = item["Q_lpips"]["mean"]
    mean_clipsim = item["Q_clipsim"]["mean"]

    risk = (
        0.25 * (1 - mean_j) + # 內容重疊度很低
        0.25 * (1 - mean_t) + # 語義意思差很多
        0.10 * std_j + # 內容波動很大
        0.10 * std_t + # 語義波動很大
        0.15 * mean_lpips + # 畫質損失很大
        0.15 * (1 - mean_clipsim) # 圖片特徵偏移很大
    )

    pred = "adversarial" if risk >= tau else "benign"

    results[fname] = {
        "risk_score": round(risk, 6),
        "prediction": pred,
        "mean_jaccard": round(mean_j, 6),
        "std_jaccard": round(std_j, 6),
        "mean_textcos": round(mean_t, 6),
        "std_textcos": round(std_t, 6),
        "mean_lpips": round(mean_lpips, 6),
        "mean_clipsim": round(mean_clipsim, 6),
    }

    print(fname, "| risk =", round(risk, 6), "| pred =", pred)

with open(out_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Saved detection results to {out_file}")