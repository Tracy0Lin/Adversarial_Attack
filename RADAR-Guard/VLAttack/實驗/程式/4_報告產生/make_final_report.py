"""Write RESULTS_簡明.md — plain-language summary for all experiments and captioners."""
import json
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "VQA" / "multi_captioner"
CAPS = ("blip", "vitgpt2", "git", "gemma3")
NAMES = {"blip": "BLIP（對照）", "vitgpt2": "ViT-GPT2", "git": "GIT-large-COCO", "gemma3": "gemma3:4b（LLM-API）"}
EXPS = ("exp1", "exp2", "exp3")
EXP_LABEL = {"exp1": "實驗一（混合攻擊）", "exp2": "實驗二（全成功攻擊）", "exp3": "實驗三（僅攻擊圖）"}


def pct(x):
    if x is None:
        return "—"
    return f"{100 * float(x):.1f}%"


def load(exp, cap):
    p = DATA / exp / cap / "summary.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def main():
    lines = [
        "# 多 caption 模型防禦評估 — 簡明結果",
        "",
        f"產生時間：{__import__('datetime').datetime.now():%Y-%m-%d %H:%M}",
        "",
        "## 這次做了什麼",
        "",
        "- 三個獨立實驗情境 × 四種 caption 探針（含 LLM-API gemma3、ViT-GPT2、GIT；BLIP 為舊法對照）。",
        "- 同一套兩階段：±5° 旋轉 caption → SVM 分 Low/Mid/High；Mid 用高斯 θ\\* 緩解 + 答案翻轉 θ_det 偵測。",
        "- 黑盒 VQA 仍是 BLIP VQA。",
        "",
        "## 完整流程（A3）— 你最該看的數字",
        "",
        "| 實驗 | Caption | 攔截 TPR | 誤攔 FPR | J | 防禦前 ASR | 防禦後有效 ASR | 乾淨服務率 | Stage1 AUROC |",
        "|------|---------|----------|----------|---|------------|----------------|------------|--------------|",
    ]
    for exp in EXPS:
        for cap in CAPS:
            s = load(exp, cap)
            if not s:
                lines.append(f"| {EXP_LABEL[exp]} | {NAMES[cap]} | （未完成） | | | | | | |")
                continue
            f = s.get("full_pipeline", {})
            r = s.get("stage1_risk", {})
            lines.append(
                f"| {EXP_LABEL[exp]} | {NAMES[cap]} | {pct(f.get('TPR'))} | {pct(f.get('FPR'))} | "
                f"{f.get('youden_j', 0):.3f} | {pct(f.get('asr_before'))} | {pct(f.get('effective_asr'))} | "
                f"{pct(f.get('clean_service_rate'))} | {r.get('auroc', '—') if r else '—'} |"
            )

    lines += [
        "",
        "## 實驗二消融（各 caption 完整表見 `RESULTS_tables.md`）",
        "",
        "重點對照：",
        "- **A1** 只第一階段攔 High；**A1′** 攔 Mid+High。",
        "- **A2** 全列只做第二階段（翻轉偵測 + 緩解）；**A3** 完整路由。",
        "",
    ]
    for cap in CAPS:
        p = DATA / "exp2" / cap / "pipeline_variants.csv"
        if not p.is_file():
            continue
        import pandas as pd

        v = pd.read_csv(p)
        a3 = v[v["variant"] == "A3_full_pipeline"].iloc[0]
        a1 = v[v["variant"] == "A1_stage1_high_block"].iloc[0]
        a2 = v[v["variant"] == "A2_stage2_only"].iloc[0]
        lines.append(
            f"- **{NAMES[cap]}**：A1 TPR {pct(a1['TPR'])} FPR {pct(a1['FPR'])}；"
            f"A2 TPR {pct(a2['TPR'])} FPR {pct(a2['FPR'])}；"
            f"A3 TPR {pct(a3['TPR'])} FPR {pct(a3['FPR'])}、有效 ASR {pct(a3['effective_asr'])}。"
        )

    lines += [
        "",
        "## 實驗三說明",
        "",
        "只有 1000 張攻擊成功圖，沒有乾淨列，因此**沒有 FPR / AUROC**；Stage1 與 θ 來自實驗二同模型。",
        "表上「攔截 TPR」= 被判為攻擊而攔下的比例；「有效 ASR」= 未被攔且仍騙過 VQA 的比例。",
        "",
        "## 檔案位置",
        "",
        "- 計畫：`PLAN.md`",
        "- 細表：`RESULTS_tables.md`",
        "- 每組：`exp{1,2,3}/<caption>/summary.json`、`pipeline_variants.csv`、`rows_scored.csv`",
        "- 探針：`exp*/probe_gaussian.csv`",
        "",
    ]
    out = DATA / "RESULTS_簡明.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
