"""Stage 2: after Gaussian blur, how many answers match ground truth (BLIP VQA)?

Definition (same as sss_denoise.py / fooled_after):
    correct_after_blur  <=>  str(pred_denoised) == str(correct_ans)
    attack_still_wrong  <=>  str(pred_denoised) != str(correct_ans)

Uses existing sss_per_theta_raw.csv @ theta* (no GPU).

Scopes:
  - mid_attack / mid_clean: only Mid-risk rows (where blur was actually run)
  - routed_1000_attack: full mitigation routing (Low=adv VQA, Mid=blur VQA, High=blocked→correct)

Run:
  python BLIP_attack/output/VQA/stage2_blip_accuracy_after_blur.py

Writes:
  data_supplement/checklist_prepared/stage2_blip_correct_after_blur.csv
  data_supplement/checklist_prepared/stage2_blip_correct_after_blur.md
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

VQA = Path(__file__).resolve().parent
OUT = VQA / "data_supplement" / "checklist_prepared"

EXPS = [
    (
        "exp1",
        "實驗一",
        VQA / "steps100_alpha0.002_run1000",
        VQA / "exp1_mixed1000" / "dataset_manifest.csv",
    ),
    (
        "exp2",
        "實驗二",
        VQA / "exp2_success1000",
        VQA / "exp2_success1000" / "dataset_manifest.csv",
    ),
]


def load_theta_sigma(root: Path) -> float:
    with open(root / "sss_best_theta.json", encoding="utf-8") as f:
        return float(json.load(f)["sigma"])


def accuracy(pred: pd.Series, gt: pd.Series) -> tuple[int, int, float]:
    ok = pred.astype(str) == gt.astype(str)
    n = len(ok)
    c = int(ok.sum())
    return c, n, c / n if n else float("nan")


def summarize_scope(
    experiment: str,
    title: str,
    scope: str,
    n: int,
    correct_n: int,
    correct_before_n: int | None,
    sigma: float,
    note: str,
) -> dict:
    rate = correct_n / n if n else float("nan")
    row = {
        "experiment": experiment,
        "experiment_title": title,
        "scope": scope,
        "sigma": sigma,
        "n": n,
        "correct_after_blur_n": correct_n,
        "correct_after_blur_rate": rate,
        "wrong_after_blur_n": n - correct_n,
        "wrong_after_blur_rate": 1.0 - rate if n else float("nan"),
        "note": note,
    }
    if correct_before_n is not None:
        row["correct_before_blur_n"] = correct_before_n
        row["correct_before_blur_rate"] = correct_before_n / n if n else float("nan")
    return row


def run_one(experiment: str, title: str, root: Path, manifest_path: Path) -> list[dict]:
    sigma = load_theta_sigma(root)
    raw = pd.read_csv(root / "sss_per_theta_raw.csv")
    raw["image_id"] = raw["image_id"].astype(str)
    sub = raw[raw["sigma"] == sigma].copy()

    stage1 = pd.read_csv(root / "tcvad_stage1_4_academic_dataset.csv")
    stage1 = stage1[stage1["method_type"] == "rotate_5"]
    stage1["image_id"] = stage1["image_id"].astype(str)

    manifest = pd.read_csv(manifest_path)
    manifest["image_id"] = manifest["question_id"].astype(str)
    gt_m = manifest.set_index("image_id")["ground_truth"].astype(str)
    adv_m = manifest.set_index("image_id")["adversarial_prediction"].astype(str)

    rows = []

    mid_atk = sub[(sub["label"] == 1) & (sub["kind"] == "attack")]
    c_bef, n, r_bef = accuracy(mid_atk["pred_ref"], mid_atk["correct_ans"])
    c_aft, _, r_aft = accuracy(mid_atk["pred_denoised"], mid_atk["correct_ans"])
    rows.append(
        summarize_scope(
            experiment,
            title,
            "mid_pool_attack",
            n,
            c_aft,
            c_bef,
            sigma,
            "Mid 對抗圖：煳前 pred_ref、煳後 pred_denoised，皆與 correct_ans 比",
        )
    )

    mid_cl = sub[(sub["label"] == 0) & (sub["kind"] == "clean")]
    c_bef_c, nc, _ = accuracy(mid_cl["pred_ref"], mid_cl["correct_ans"])
    c_aft_c, _, _ = accuracy(mid_cl["pred_denoised"], mid_cl["correct_ans"])
    rows.append(
        summarize_scope(
            experiment,
            title,
            "mid_pool_clean",
            nc,
            c_aft_c,
            c_bef_c,
            sigma,
            "Mid 乾淨圖：煳後仍答對 = 未因模糊答錯",
        )
    )

    # Full 1000 attack: mitigation routing
    s1_atk = stage1[stage1["label"] == 1].merge(manifest[["image_id", "ground_truth"]], on="image_id")
    den = mid_atk.set_index("image_id")["pred_denoised"]
    final_pred = []
    for r in s1_atk.itertuples():
        g = str(r.ground_truth)
        if r.risk_bin == "high":
            final_pred.append(g)
        elif r.risk_bin == "low":
            final_pred.append(str(adv_m[r.image_id]))
        else:
            q = r.image_id
            final_pred.append(str(den[q]) if q in den.index else str(adv_m[q]))
    s1_atk = s1_atk.copy()
    s1_atk["final_pred"] = final_pred
    s1_atk["gt"] = s1_atk["ground_truth"].astype(str)
    ok = s1_atk["final_pred"] == s1_atk["gt"]
    rows.append(
        summarize_scope(
            experiment,
            title,
            "routed_mitigation_1000_attack",
            len(s1_atk),
            int(ok.sum()),
            int((adv_m.reindex(s1_atk["image_id"]).astype(str).values == s1_atk["gt"].values).sum()),
            sigma,
            "Low=adv VQA；Mid=煳後 VQA；High=阻擋視為輸出 GT（攻擊失效）",
        )
    )

    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for exp, title, root, man in EXPS:
        all_rows.extend(run_one(exp, title, root, man))
    df = pd.DataFrame(all_rows)
    df.to_csv(OUT / "stage2_blip_correct_after_blur.csv", index=False, encoding="utf-8-sig")

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Stage 2 高斯後：BLIP 答對率（vs 標準答案）",
        "",
        f"更新：{ts}  ",
        "",
        "## 怎麼做（定義）",
        "",
        "1. 對影像做 Stage 2 高斯（本實驗 θ*：**5×5、σ 來自 `sss_best_theta.json`**，通常 0.8）。",
        "2. 用 **同一套 BLIP VQA** 對 **煳後圖** 再問同一題。",
        "3. 將輸出字串與 **`correct_ans`（題目標準答案）** 比對：",
        "   - **答對**：`pred_denoised == correct_ans`",
        "   - **仍攻擊成功／答錯**：`pred_denoised != correct_ans`（即 `fooled_after=True`）",
        "",
        "資料已在 `sss_per_theta_raw.csv` 的 **`pred_denoised`** 欄（當初跑 `sss_denoise.py` 時就算好）。",
        "",
        "## 注意範圍",
        "",
        "| 範圍 | 意義 |",
        "|------|------|",
        "| **mid_pool_*** | 只有 **Mid 風險** 列有跑煳圖+VQA（實驗一約 391 對抗 + 395 乾淨） |",
        "| **routed_mitigation_1000_attack** | **1000 張對抗** 依管線：Low 不煳、Mid 用煳後答案、High 視為阻擋（輸出視為 GT） |",
        "",
        "若要对 **Low/High 也各自跑高斯+BLIP**，需擴充 `sss_denoise.py` 重跑 GPU（目前 log 未存）。",
        "",
        "## 結果表",
        "",
        "| 實驗 | 範圍 | n | 煳前答對 | 煳後答對 | 煳後答對率 |",
        "|------|------|---|----------|----------|------------|",
    ]
    for _, r in df.iterrows():
        if pd.notna(r.get("correct_before_blur_n")):
            bef_s = f"{int(r['correct_before_blur_n'])}（{100*r['correct_before_blur_rate']:.2f}%）"
        else:
            bef_s = "—"
        lines.append(
            f"| {r['experiment_title']} | {r['scope']} | {int(r['n'])} | {bef_s} | "
            f"{int(r['correct_after_blur_n'])}（{100*r['correct_after_blur_rate']:.2f}%） | "
            f"{100*r['correct_after_blur_rate']:.2f}% |"
        )
    lines += [
        "",
        "CSV：`stage2_blip_correct_after_blur.csv`",
        "",
        "重算：`python BLIP_attack/output/VQA/stage2_blip_accuracy_after_blur.py`",
        "",
    ]
    (OUT / "stage2_blip_correct_after_blur.md").write_text("\n".join(lines), encoding="utf-8")
    print(df.to_string(index=False))
    print(f"\nWrote {OUT / 'stage2_blip_correct_after_blur.csv'}")


if __name__ == "__main__":
    main()
