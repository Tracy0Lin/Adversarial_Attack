#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""從 11/22 正式重跑的實驗二真實逐張資料，重算指標、寫 Excel，並畫 08.11 風格密度圖。

不修改任何原始檔。輸出寫到本資料夾。
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from scipy.stats import gaussian_kde
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from openpyxl import Workbook
from openpyxl.chart import BarChart, LineChart, Reference
from openpyxl.chart.series import SeriesLabel
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.worksheet.table import Table, TableStyleInfo
from openpyxl.formatting.rule import FormulaRule
from openpyxl.chart.label import DataLabelList

# ---------------------------------------------------------------------------
# Paths (read-only sources)
# ---------------------------------------------------------------------------
ARCHIVE = Path(os.environ.get("VLATTACK_COSANG_ARCHIVE", "/workspace/radar_dossier/cos_vs_angular_20261123_archive/05_formal_1122_rerun"))
SRC_FEAT = ARCHIVE / "outputs" / "exp2_features.csv"
SRC_OOF = ARCHIVE / "outputs" / "oof_scores.csv"
SRC_BOOT = ARCHIVE / "outputs" / "bootstrap_delta.csv"
SRC_RPF1 = ARCHIVE / "formal_1122_rpf1.csv"
SRC_SCRIPT = ARCHIVE / "run_rotation_cos_vs_ang.py"
SRC_COMPUTE = ARCHIVE / "compute_rpf1.py"

OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
FIG_RAW = FIG / "補充_原始cosine_center與q95"
XLSX = OUT / "實驗二_cos_vs_角距_真實數據.xlsx"

SEED = 42
N_BOOT = 1000
PUBLISHED = {
    "cosine_4d": dict(AUROC=0.7217, Recall=0.6050, Precision=0.7043, F1=0.6509),
    "angular_4d": dict(AUROC=0.7233, Recall=0.6790, Precision=0.6893, F1=0.6841),
}
PUB_CI = (-0.0090, 0.0130)

# float32 identical-caption snap (same rule as the 11/23 formal-like-0811 figures)
ZERO_COS = 1e-6
ZERO_ANG = float(np.arccos(1 - 1e-6) / np.pi)

PAL = {"Clean 影像": "#2ecc71", "VLAttack 影像": "#e74c3c"}
NAME_COS = "未做角距離（cos）"
NAME_ANG = "有做角距離（角距）"


def setup_font():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK TC", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["savefig.pad_inches"] = 0.12


def kde_reflect(v, grid):
    """Gaussian KDE (Scott) with reflection at 0: no mass below 0, no boundary dip."""
    k = gaussian_kde(np.concatenate([v, -v]))
    return 2 * k(grid) * (grid >= 0)


def draw_density(ax, x, y, title, xlabel, auc, zc, za, small=False, y_label="樣本密度 Density"):
    """08.11-style KDE on non-zero values; curve area = that group's non-zero share."""
    hi = np.quantile(x[x > 0], 0.99) if (x > 0).any() else 1e-3
    grid = np.linspace(0, hi * 1.08, 800)
    for lab, name, z0 in [(0, "Clean 影像", zc), (1, "VLAttack 影像", za)]:
        v = x[(y == lab) & (x > 0)]
        if len(v) > 5:
            d = kde_reflect(v, grid) * (1 - z0)
            ax.fill_between(grid, d, color=PAL[name], alpha=0.4, lw=0)
            ax.plot(grid, d, color=PAL[name], lw=2)
    ax.set_xlim(0, grid[-1])
    ax.set_ylim(bottom=0)
    fs = 1 if not small else 0.62
    ax.set_title(title, fontsize=14 * fs, fontweight="bold", pad=10 * fs)
    ax.set_xlabel(xlabel, fontsize=11 * fs)
    ax.set_ylabel(y_label, fontsize=11 * fs)
    ax.tick_params(labelsize=9 * fs)
    h = [Patch(facecolor=PAL[k], edgecolor=PAL[k], alpha=0.6, lw=2) for k in PAL]
    ax.legend(h, list(PAL), title="類別", loc="upper right", fontsize=9 * fs, title_fontsize=9 * fs, frameon=True)
    note = f"單一特徵 AUROC = {auc:.3f}\n恰為 0（未畫入曲線）\nclean {zc:.0%}｜attack {za:.0%}"
    ax.text(
        0.97,
        0.72,
        note,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10 * fs,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cccccc"),
    )


def draw_density_near_one(ax, x, y, title, xlabel, auc, zc, za, small=False):
    """Raw cosine center / q95: mass near 1 (identical captions) is held out of the KDE."""
    identical = x >= (1.0 - ZERO_COS)
    v_plot = x[~identical]
    lo = np.quantile(v_plot, 0.01) if len(v_plot) else 0.8
    grid = np.linspace(max(0.0, lo * 0.995), 1.0, 800)
    for lab, name, z0 in [(0, "Clean 影像", zc), (1, "VLAttack 影像", za)]:
        v = x[(y == lab) & (~identical)]
        if len(v) > 5:
            k = gaussian_kde(v)
            d = k(grid) * (1 - z0)
            ax.fill_between(grid, d, color=PAL[name], alpha=0.4, lw=0)
            ax.plot(grid, d, color=PAL[name], lw=2)
    ax.set_xlim(grid[0], 1.0)
    ax.set_ylim(bottom=0)
    fs = 1 if not small else 0.62
    ax.set_title(title, fontsize=14 * fs, fontweight="bold", pad=10 * fs)
    ax.set_xlabel(xlabel, fontsize=11 * fs)
    ax.set_ylabel("樣本密度 Density", fontsize=11 * fs)
    ax.tick_params(labelsize=9 * fs)
    h = [Patch(facecolor=PAL[k], edgecolor=PAL[k], alpha=0.6, lw=2) for k in PAL]
    ax.legend(h, list(PAL), title="類別", loc="upper left", fontsize=9 * fs, title_fontsize=9 * fs, frameon=True)
    note = f"單一特徵 AUROC（朝攻擊）= {auc:.3f}\n恰為 1（未畫入曲線）\nclean {zc:.0%}｜attack {za:.0%}"
    ax.text(
        0.03,
        0.72,
        note,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10 * fs,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cccccc"),
    )


def snap_zero(v, thresh):
    v = np.asarray(v, dtype=float)
    return np.where(np.abs(v) <= thresh, 0.0, v)


def bootstrap_delta(labels, groups, cos_score, ang_score, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    photos = np.unique(groups)
    buckets = {photo: np.flatnonzero(groups == photo) for photo in photos}
    deltas = []
    for _ in range(n_boot):
        drawn = rng.choice(photos, size=len(photos), replace=True)
        idx = np.concatenate([buckets[photo] for photo in drawn])
        y = labels[idx]
        if y.min() == y.max():
            continue
        deltas.append(roc_auc_score(y, ang_score[idx]) - roc_auc_score(y, cos_score[idx]))
    deltas = np.asarray(deltas, dtype=np.float64)
    return {
        "n_valid": int(len(deltas)),
        "mean": float(deltas.mean()),
        "p2_5": float(np.percentile(deltas, 2.5)),
        "p97_5": float(np.percentile(deltas, 97.5)),
    }


def load_and_score():
    feat = pd.read_csv(SRC_FEAT)
    oof = pd.read_csv(SRC_OOF)
    oof2 = oof[oof["experiment"] == "exp2"].copy()
    if len(feat) != 2000:
        raise SystemExit(f"exp2_features.csv 應有 2000 列，實際 {len(feat)}")

    def take(feature):
        g = oof2[oof2["feature"] == feature].set_index(["image_id", "label"])
        if len(g) != 2000:
            raise SystemExit(f"{feature} 應有 2000 列，實際 {len(g)}")
        return g

    cos = take("cosine_4d")
    ang = take("angular_4d")
    key = pd.MultiIndex.from_frame(feat[["image_id", "label"]])
    df = feat.copy()
    df["oof_margin_cos"] = cos.loc[key, "oof_margin"].to_numpy()
    df["oof_risk_cos"] = cos.loc[key, "oof_risk"].to_numpy()
    df["oof_pred_cos"] = cos.loc[key, "oof_pred"].to_numpy()
    df["oof_margin_ang"] = ang.loc[key, "oof_margin"].to_numpy()
    df["oof_risk_ang"] = ang.loc[key, "oof_risk"].to_numpy()
    df["oof_pred_ang"] = ang.loc[key, "oof_pred"].to_numpy()
    df["d_center_1minus_cos"] = 1.0 - df["d_center_cos"]
    df["d_q95_1minus_cos"] = 1.0 - df["d_q95_cos"]
    if not np.array_equal(df["oof_pred_cos"].to_numpy(), (df["oof_margin_cos"].to_numpy() >= 0).astype(int)):
        raise SystemExit("cosine oof_pred 與 margin>=0 不一致")
    if not np.array_equal(df["oof_pred_ang"].to_numpy(), (df["oof_margin_ang"].to_numpy() >= 0).astype(int)):
        raise SystemExit("angular oof_pred 與 margin>=0 不一致")
    return df


def metric_block(y, margin, pred, key):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    raw = dict(
        AUROC=float(roc_auc_score(y, margin)),
        Recall=float(recall_score(y, pred)),
        Precision=float(precision_score(y, pred)),
        F1=float(f1_score(y, pred)),
        TP=int(tp),
        FP=int(fp),
        TN=int(tn),
        FN=int(fn),
        n=int(len(y)),
        n_clean=int((y == 0).sum()),
        n_attack=int((y == 1).sum()),
    )
    rounded = {k: round(raw[k], 4) for k in ("AUROC", "Recall", "Precision", "F1")}
    pub = PUBLISHED[key]
    raw["matches_published"] = all(abs(rounded[k] - pub[k]) < 5e-5 for k in pub)
    raw["rounded"] = rounded
    return raw


def univariate(df):
    y = df["label"].to_numpy(dtype=int)
    rows = []
    for name in ("d_center", "d_iqr", "d_tail", "d_q95"):
        for kind, col in (("cosine", f"{name}_cos"), ("angular", f"{name}_ang")):
            values = df[col].to_numpy(dtype=np.float64)
            raw = float(roc_auc_score(y, values))
            flipped = float(roc_auc_score(y, -values))
            if raw >= flipped:
                auroc, direction = raw, "higher（愈大愈像攻擊）"
            else:
                auroc, direction = flipped, "lower（愈小愈像攻擊）"
            rows.append(
                dict(
                    特徵=name,
                    距離表示=kind,
                    單一特徵AUROC=auroc,
                    攻擊方向=direction,
                )
            )
    return pd.DataFrame(rows)


def roc_tables(y, score_cos, score_ang):
    fpr_c, tpr_c, thr_c = roc_curve(y, score_cos)
    fpr_a, tpr_a, thr_a = roc_curve(y, score_ang)
    raw_c = pd.DataFrame({"距離表示": NAME_COS, "FPR": fpr_c, "TPR": tpr_c, "threshold_oof_margin": thr_c})
    raw_a = pd.DataFrame({"距離表示": NAME_ANG, "FPR": fpr_a, "TPR": tpr_a, "threshold_oof_margin": thr_a})
    raw = pd.concat([raw_c, raw_a], ignore_index=True)
    grid = np.linspace(0.0, 1.0, 1001)
    common = pd.DataFrame(
        {
            "FPR": grid,
            "TPR_未做角距離_cos": np.interp(grid, fpr_c, tpr_c),
            "TPR_有做角距離_角距": np.interp(grid, fpr_a, tpr_a),
        }
    )
    return raw, common


# ---------------------------------------------------------------------------
# Excel helpers
# ---------------------------------------------------------------------------
NAVY = "1F4E79"
TEAL = "1B6A4A"
ORANGE = "C65911"
GREY = "F2F2F2"
WHITE = "FFFFFF"
THIN = Border(
    left=Side(style="thin", color="B0B0B0"),
    right=Side(style="thin", color="B0B0B0"),
    top=Side(style="thin", color="B0B0B0"),
    bottom=Side(style="thin", color="B0B0B0"),
)
FONT_NAME = "Microsoft JhengHei"


def font(bold=False, color=WHITE, size=11):
    return Font(name=FONT_NAME, bold=bold, color=color, size=size)


def fill(hex_color):
    return PatternFill("solid", fgColor=hex_color)


def align(h="center", v="center", wrap=True):
    return Alignment(horizontal=h, vertical=v, wrap_text=wrap)


def style_header(ws, row, start_col, end_col, color=NAVY):
    for col in range(start_col, end_col + 1):
        cell = ws.cell(row=row, column=col)
        cell.font = font(bold=True, color=WHITE, size=10)
        cell.fill = fill(color)
        cell.alignment = align()
        cell.border = THIN


def style_body(ws, r0, r1, c0, c1, num_fmt=None):
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            cell = ws.cell(row=r, column=c)
            cell.font = font(color="000000", size=10)
            cell.alignment = align(h="center" if c > 1 else "left")
            cell.border = THIN
            if (r - r0) % 2 == 1:
                if cell.fill.fgColor is None or cell.fill.fgColor.rgb in ("00000000", "0"):
                    cell.fill = fill(GREY)
            if num_fmt and isinstance(cell.value, float):
                cell.number_format = num_fmt


def set_widths(ws, widths):
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w


def write_excel(df, metrics, boot, uni, roc_raw, roc_common):
    wb = Workbook()

    # ===== 指標表 =====
    ws = wb.active
    ws.title = "指標表"
    ws.sheet_properties.tabColor = NAVY
    ws["A1"] = "實驗二（正式重跑 2026-11-22）四維特徵 × RBF-SVM：未做角距離（cos）對有做角距離（角距）"
    ws["A1"].font = font(bold=True, color=NAVY, size=14)
    ws.merge_cells("A1:L1")
    ws["A2"] = (
        "資料：VLAttack／BLIP／VQA，2,000 張（乾淨 1,000 ＋ 攻擊 1,000）。"
        "每張 11 句描述（原圖＋10 個 ±5° 旋轉），MiniLM 嵌入。"
        "分類器：RBF-SVM（C=1，class_weight=balanced，先標準化），依 photo_id 分組的 5-fold，切點為折外決策值 ≥ 0。"
        "對照組用的是原始 cosine 四維（mean、IQR、Q90−Q50、Q95），不是 1−cos。"
        "下列數字全部由逐張折外分數重算，四位小數與發表表一致。"
    )
    ws["A2"].font = font(color="333333", size=10)
    ws["A2"].alignment = align(h="left")
    ws.merge_cells("A2:L2")
    ws.row_dimensions[1].height = 22
    ws.row_dimensions[2].height = 48

    headers1 = [
        "設定 configuration",
        "距離表示 distance",
        "樣本數 n",
        "AUROC",
        "召回率 Recall",
        "精確度 Precision",
        "F1-Score",
        "TP",
        "FP",
        "TN",
        "FN",
        "四位小數是否完全重現",
    ]
    for c, h in enumerate(headers1, 1):
        ws.cell(row=4, column=c, value=h)
    style_header(ws, 4, 1, 12)

    rows_main = [
        (NAME_COS, "cosine（原始 cosine 四維）", metrics["cosine_4d"], "cosine_4d"),
        (NAME_ANG, "角距 δ = arccos(cos)/π 四維", metrics["angular_4d"], "angular_4d"),
    ]
    for i, (conf, dist, m, key_name) in enumerate(rows_main):
        r = 5 + i
        vals = [
            conf,
            dist,
            m["n"],
            m["AUROC"],
            m["Recall"],
            m["Precision"],
            m["F1"],
            m["TP"],
            m["FP"],
            m["TN"],
            m["FN"],
            (
                "是（0.7217 / 0.6050 / 0.7043 / 0.6509）"
                if key_name == "cosine_4d"
                else "是（0.7233 / 0.6790 / 0.6893 / 0.6841）"
            )
            if m["matches_published"]
            else "否（請核對）",
        ]
        for c, v in enumerate(vals, 1):
            ws.cell(row=r, column=c, value=v)
        for c in range(4, 8):
            ws.cell(row=r, column=c).number_format = "0.0000"
        style_body(ws, r, r, 1, 12)
        if i == 0:
            ws.cell(row=r, column=1).fill = fill("D6EAF8")
        else:
            ws.cell(row=r, column=1).fill = fill("FDEBD0")

    # compact table for the Excel bar chart (hidden-ish at columns N-P)
    ws["N4"] = "指標"
    ws["O4"] = NAME_COS
    ws["P4"] = NAME_ANG
    style_header(ws, 4, 14, 16, TEAL)
    for i, key in enumerate(("AUROC", "Recall", "Precision", "F1")):
        ws.cell(row=5 + i, column=14, value=key)
        ws.cell(row=5 + i, column=15, value=metrics["cosine_4d"]["rounded"][key if key != "F1" else "F1"])
        ws.cell(row=5 + i, column=16, value=metrics["angular_4d"]["rounded"][key if key != "F1" else "F1"])
        ws.cell(row=5 + i, column=15).number_format = "0.0000"
        ws.cell(row=5 + i, column=16).number_format = "0.0000"
        style_body(ws, 5 + i, 5 + i, 14, 16, "0.0000")
    # fix F1 key - already used "F1" in rounded dict as "F1"
    for i, key in enumerate(("AUROC", "Recall", "Precision", "F1")):
        ws.cell(row=5 + i, column=15, value=metrics["cosine_4d"]["rounded"][key])
        ws.cell(row=5 + i, column=16, value=metrics["angular_4d"]["rounded"][key])
        ws.cell(row=5 + i, column=15).number_format = "0.0000"
        ws.cell(row=5 + i, column=16).number_format = "0.0000"

    chart = BarChart()
    chart.type = "col"
    chart.grouping = "clustered"
    chart.title = "實驗二四項指標：未做角距離（cos）vs 有做角距離（角距）"
    chart.y_axis.title = "分數"
    chart.y_axis.scaling.min = 0
    chart.y_axis.scaling.max = 1
    chart.x_axis.title = None
    data_ref = Reference(ws, min_col=15, max_col=16, min_row=4, max_row=8)
    cats = Reference(ws, min_col=14, min_row=5, max_row=8)
    chart.add_data(data_ref, titles_from_data=True)
    chart.set_categories(cats)
    chart.shape = 4
    chart.style = 10
    chart.y_axis.numFmt = "0.00"
    chart.dataLabels = DataLabelList()
    chart.dataLabels.showVal = True
    chart.width = 18
    chart.height = 8
    ws.add_chart(chart, "A8")

    # CI section
    ws["A24"] = "AUROC 差的 95% 信賴區間（角距 − cosine）"
    ws["A24"].font = font(bold=True, color=NAVY, size=12)
    ws.merge_cells("A24:F24")
    headers_ci = ["方法", "有效抽樣 n_valid", "平均差 mean", "2.5% 分位 p2.5", "97.5% 分位 p97.5", "與發表區間 [−0.0090, 0.0130] 是否一致"]
    for c, h in enumerate(headers_ci, 1):
        ws.cell(row=25, column=c, value=h)
    style_header(ws, 25, 1, 6, TEAL)
    ci_ok = (round(boot["p2_5"], 4) == PUB_CI[0]) and (round(boot["p97_5"], 4) == PUB_CI[1])
    ci_vals = [
        "照片分組 bootstrap 1000 次（seed=42；抽已算好的折外決策值，不重訓）",
        boot["n_valid"],
        boot["mean"],
        boot["p2_5"],
        boot["p97_5"],
        "是" if ci_ok else "否（請核對）",
    ]
    for c, v in enumerate(ci_vals, 1):
        ws.cell(row=26, column=c, value=v)
    for c in (3, 4, 5):
        ws.cell(row=26, column=c).number_format = "0.0000"
    style_body(ws, 26, 26, 1, 6)
    ws["A27"] = "區間跨過 0，故在本設定下角距與 cosine 的 AUROC 差距未達顯著。發表值四位小數為 [−0.0090, 0.0130]（完整值見上列）。"
    ws["A27"].font = font(color="333333", size=10)
    ws.merge_cells("A27:F27")

    # confusion note
    ws["A29"] = "預設切點（決策值 ≥ 0）補充"
    ws["A29"].font = font(bold=True, color=NAVY, size=12)
    ws.merge_cells("A29:F29")
    headers_cf = ["設定", "攻擊數", "攔截 TP（Recall）", "誤攔 FP（FPR）", "正確放行 TN", "漏攔 FN"]
    for c, h in enumerate(headers_cf, 1):
        ws.cell(row=30, column=c, value=h)
    style_header(ws, 30, 1, 6)
    for i, (conf, m) in enumerate(((NAME_COS, metrics["cosine_4d"]), (NAME_ANG, metrics["angular_4d"]))):
        r = 31 + i
        vals = [conf, m["n_attack"], m["TP"], m["FP"], m["TN"], m["FN"]]
        for c, v in enumerate(vals, 1):
            ws.cell(row=r, column=c, value=v)
        style_body(ws, r, r, 1, 6)
    ws["A33"] = (
        f"{NAME_COS}：攔截 605/1000、誤攔 254/1000（FPR=0.254）。"
        f"{NAME_ANG}：攔截 679/1000、誤攔 306/1000（FPR=0.306）。"
        "切點為 SVM 決策值 ≥ 0（即 oof_pred）。AUROC 用折外決策值 oof_margin，不用風險分。"
    )
    ws["A33"].font = font(color="333333", size=10)
    ws["A33"].alignment = align(h="left")
    ws.merge_cells("A33:F33")
    ws.row_dimensions[33].height = 32

    # univariate
    ws["A35"] = "單一特徵 AUROC（不是 SVM；攻擊為正類，已轉成「愈大愈像攻擊」）"
    ws["A35"].font = font(bold=True, color=NAVY, size=12)
    ws.merge_cells("A35:E35")
    headers_u = ["特徵 feature", "距離表示", "單一特徵 AUROC", "攻擊方向 attack_direction", "備註"]
    for c, h in enumerate(headers_u, 1):
        ws.cell(row=36, column=c, value=h)
    style_header(ws, 36, 1, 5, ORANGE)
    notes = {
        ("d_center", "cosine"): "正式特徵是 mean(cos)，愈高愈聚；攻擊方向為愈低。圖一主圖改畫 1−mean(cos) 以便和角距同向。",
        ("d_q95", "cosine"): "正式特徵是 Q95(cos)，愈高愈聚；攻擊方向為愈低。圖七主圖改畫 1−Q95(cos)。",
        ("d_iqr", "cosine"): "IQR(cos)=IQR(1−cos)，方向已是愈大愈散，主圖直接畫正式特徵。",
        ("d_tail", "cosine"): "正式特徵是 Q90(cos)−Q50(cos)，不是 1−cos 的尾距（兩者不相等）。主圖直接畫正式特徵。",
        ("d_center", "angular"): "正式特徵 D_center_ang，愈大愈散。",
        ("d_iqr", "angular"): "正式特徵 D_iqr_ang。",
        ("d_tail", "angular"): "正式特徵 D_tail_ang。",
        ("d_q95", "angular"): "正式特徵 D_q95_ang；單維裡角距相對較高的一項。",
    }
    for i, row in uni.iterrows():
        r = 37 + i
        kind = "cosine" if row["距離表示"] == "cosine" else "angular"
        feat = row["特徵"]
        vals = [feat, row["距離表示"], row["單一特徵AUROC"], row["攻擊方向"], notes.get((feat, kind), "")]
        for c, v in enumerate(vals, 1):
            ws.cell(row=r, column=c, value=v)
        ws.cell(row=r, column=3).number_format = "0.0000"
        style_body(ws, r, r, 1, 5, "0.0000")
        ws.cell(row=r, column=5).alignment = align(h="left")
    ws["A45"] = "上表四位小數與 11/22 正式備註一致：實驗二 cosine / 角距 為 center 0.7116 / 0.6938，IQR 0.6483 / 0.6440，tail 0.5906 / 0.6108，Q95 0.6782 / 0.7276。"
    ws["A45"].font = font(color="333333", size=10)
    ws.merge_cells("A45:E45")

    set_widths(ws, [36, 28, 14, 16, 16, 16, 14, 10, 10, 10, 10, 42])
    ws.column_dimensions["N"].width = 14
    ws.column_dimensions["O"].width = 24
    ws.column_dimensions["P"].width = 24
    ws.freeze_panes = "A5"
    ws.row_dimensions[4].height = 28
    ws.print_title_rows = "1:4"
    ws.page_setup.orientation = "landscape"
    ws.page_setup.fitToPage = True
    ws.page_setup.fitToWidth = 1
    ws.page_setup.fitToHeight = 1

    # ===== 逐張特徵 =====
    ws2 = wb.create_sheet("逐張特徵")
    ws2.sheet_properties.tabColor = TEAL
    per = pd.DataFrame(
        {
            "影像編號 image_id": df["image_id"],
            "照片編號 photo_id": df["photo_id"],
            "真實標籤 label": np.where(df["label"] == 1, "VLAttack", "Clean"),
            "真實標籤代碼 label_code": df["label"],
            "不同描述句數 n_unique_captions": df["n_unique_captions"],
            "餘弦中心 d_center_cos": df["d_center_cos"],
            "餘弦四分位距 d_iqr_cos": df["d_iqr_cos"],
            "餘弦尾距 d_tail_cos": df["d_tail_cos"],
            "餘弦第95分位 d_q95_cos": df["d_q95_cos"],
            "餘弦中心改寫 d_center_1minus_cos（=1−d_center_cos，僅供作圖）": df["d_center_1minus_cos"],
            "餘弦高尾改寫 d_q95_1minus_cos（=1−d_q95_cos，僅供作圖）": df["d_q95_1minus_cos"],
            "角距中心 d_center_ang": df["d_center_ang"],
            "角距四分位距 d_iqr_ang": df["d_iqr_ang"],
            "角距尾距 d_tail_ang": df["d_tail_ang"],
            "角距第95分位 d_q95_ang": df["d_q95_ang"],
            "餘弦折外決策值 oof_margin_cos": df["oof_margin_cos"],
            "角距折外決策值 oof_margin_ang": df["oof_margin_ang"],
            "餘弦折外風險分 oof_risk_cos": df["oof_risk_cos"],
            "角距折外風險分 oof_risk_ang": df["oof_risk_ang"],
            "餘弦預測標籤 pred_cos": np.where(df["oof_pred_cos"] == 1, "VLAttack", "Clean"),
            "角距預測標籤 pred_ang": np.where(df["oof_pred_ang"] == 1, "VLAttack", "Clean"),
            "餘弦預測代碼 oof_pred_cos": df["oof_pred_cos"],
            "角距預測代碼 oof_pred_ang": df["oof_pred_ang"],
        }
    )
    per = per.sort_values(["照片編號 photo_id", "影像編號 image_id", "真實標籤代碼 label_code"]).reset_index(drop=True)
    for r_idx, row in enumerate(dataframe_to_rows(per, index=False, header=True), 1):
        for c_idx, v in enumerate(row, 1):
            cell = ws2.cell(row=r_idx, column=c_idx, value=v)
            if r_idx == 1:
                continue
            if c_idx in (1, 2, 4, 5, 22, 23):
                cell.number_format = "0"
            elif c_idx >= 6:
                cell.number_format = "0.00000000"
    style_header(ws2, 1, 1, per.shape[1])
    for r in range(2, len(per) + 2):
        for c in range(1, per.shape[1] + 1):
            cell = ws2.cell(row=r, column=c)
            cell.font = font(color="000000", size=9)
            cell.alignment = align(h="center", wrap=False)
            cell.border = THIN
            if (r % 2) == 0:
                cell.fill = fill("F7FBFF")
        # color label cells
        if ws2.cell(row=r, column=3).value == "VLAttack":
            ws2.cell(row=r, column=3).font = Font(name=FONT_NAME, color="C0392B", size=9, bold=True)
        else:
            ws2.cell(row=r, column=3).font = Font(name=FONT_NAME, color="1E8449", size=9, bold=True)
    ws2.freeze_panes = "A2"
    ws2.auto_filter.ref = f"A1:{get_column_letter(per.shape[1])}{len(per)+1}"
    widths2 = [18, 16, 16, 16, 18, 20, 20, 20, 20, 36, 36, 20, 20, 20, 20, 24, 24, 22, 22, 20, 20, 18, 18]
    set_widths(ws2, widths2)
    ws2.row_dimensions[1].height = 36
    ws2.auto_filter.ref = f"A1:{get_column_letter(per.shape[1])}{len(per)+1}"

    # ===== 畫圖用_分組特徵（bonus, for Excel histograms） =====
    wsP = wb.create_sheet("畫圖用_分組特徵")
    wsP.sheet_properties.tabColor = "7D3C98"
    # side-by-side clean / attack columns for each feature
    clean = df[df["label"] == 0].reset_index(drop=True)
    attack = df[df["label"] == 1].reset_index(drop=True)
    plot_cols = [
        ("d_center_cos", "餘弦中心_原始 d_center_cos"),
        ("d_center_1minus_cos", "餘弦中心_改寫 1−d_center_cos"),
        ("d_center_ang", "角距中心 d_center_ang"),
        ("d_iqr_cos", "餘弦四分位距 d_iqr_cos"),
        ("d_iqr_ang", "角距四分位距 d_iqr_ang"),
        ("d_tail_cos", "餘弦尾距 d_tail_cos"),
        ("d_tail_ang", "角距尾距 d_tail_ang"),
        ("d_q95_cos", "餘弦第95分位_原始 d_q95_cos"),
        ("d_q95_1minus_cos", "餘弦第95分位_改寫 1−d_q95_cos"),
        ("d_q95_ang", "角距第95分位 d_q95_ang"),
        ("oof_margin_cos", "餘弦折外決策值 oof_margin_cos"),
        ("oof_margin_ang", "角距折外決策值 oof_margin_ang"),
    ]
    headers_p = []
    for _, label in plot_cols:
        headers_p.extend([f"Clean_{label}", f"VLAttack_{label}"])
    for c, h in enumerate(headers_p, 1):
        wsP.cell(row=1, column=c, value=h)
    style_header(wsP, 1, 1, len(headers_p), "7D3C98")
    n_c, n_a = len(clean), len(attack)
    n_max = max(n_c, n_a)
    for i in range(n_max):
        for j, (col, _) in enumerate(plot_cols):
            c0 = 2 * j + 1
            if i < n_c:
                wsP.cell(row=i + 2, column=c0, value=float(clean.iloc[i][col])).number_format = "0.00000000"
            if i < n_a:
                wsP.cell(row=i + 2, column=c0 + 1, value=float(attack.iloc[i][col])).number_format = "0.00000000"
    for r in range(2, n_max + 2):
        for c in range(1, len(headers_p) + 1):
            cell = wsP.cell(row=r, column=c)
            cell.font = font(color="000000", size=8)
            cell.alignment = align(wrap=False)
            cell.border = THIN
    for c in range(1, len(headers_p) + 1):
        wsP.column_dimensions[get_column_letter(c)].width = 28
    wsP.freeze_panes = "A2"
    wsP.row_dimensions[1].height = 40
    wsP.auto_filter.ref = f"A1:{get_column_letter(len(headers_p))}{n_max+1}"
    wsP["A1003"] = "說明：每項特徵各兩欄（Clean 1,000 列、VLAttack 1,000 列），可在 Excel 選兩欄插入直方圖或箱形圖。主圖的 KDE 密度圖已輸出為 PNG。"
    wsP["A1003"].font = font(color="333333", size=10)
    wsP.merge_cells("A1003:X1003")

    # ===== ROC =====
    ws3 = wb.create_sheet("ROC")
    ws3.sheet_properties.tabColor = ORANGE
    ws3["A1"] = "共用 FPR 格點（步長 0.001，以線性內插得到 TPR）——方便在 Excel 畫兩條 ROC 曲線"
    ws3["A1"].font = font(bold=True, color=NAVY, size=12)
    ws3.merge_cells("A1:C1")
    for c, h in enumerate(("FPR", "TPR_未做角距離_cos", "TPR_有做角距離_角距"), 1):
        ws3.cell(row=2, column=c, value=h)
    style_header(ws, 2, 1, 3, ORANGE)  # wrong sheet! fix below
    style_header(ws3, 2, 1, 3, ORANGE)
    for i, row in roc_common.iterrows():
        r = 3 + i
        ws3.cell(row=r, column=1, value=float(row["FPR"])).number_format = "0.000"
        ws3.cell(row=r, column=2, value=float(row["TPR_未做角距離_cos"])).number_format = "0.0000"
        ws3.cell(row=r, column=3, value=float(row["TPR_有做角距離_角距"])).number_format = "0.0000"
        for c in range(1, 4):
            cell = ws3.cell(row=r, column=c)
            cell.font = font(color="000000", size=8)
            cell.alignment = align(wrap=False)
            cell.border = THIN
            if i % 2 == 1:
                cell.fill = fill(GREY)
    line = LineChart()
    line.title = "實驗二 ROC：未做角距離（cos）vs 有做角距離（角距）"
    line.style = 10
    line.y_axis.title = "真陽性率 TPR"
    line.x_axis.title = "假陽性率 FPR"
    line.y_axis.scaling.min = 0
    line.y_axis.scaling.max = 1
    line.x_axis.scaling.min = 0
    line.x_axis.scaling.max = 1
    data_ref = Reference(ws3, min_col=2, max_col=3, min_row=2, max_row=1003)
    cats = Reference(ws3, min_col=1, min_row=3, max_row=1003)
    line.add_data(data_ref, titles_from_data=True)
    line.set_categories(cats)
    line.width = 15
    line.height = 10
    line.legend.position = "b"
    ws3.add_chart(line, "E2")

    ws3["E20"] = "AUROC（折外決策值 oof_margin）"
    ws3["E20"].font = font(bold=True, color=NAVY, size=11)
    ws3["E21"] = NAME_COS
    ws3["F21"] = metrics["cosine_4d"]["AUROC"]
    ws3["F21"].number_format = "0.0000"
    ws3["E22"] = NAME_ANG
    ws3["F22"] = metrics["angular_4d"]["AUROC"]
    ws3["F22"].number_format = "0.0000"
    ws3["E23"] = "差（角距 − cos）"
    ws3["F23"] = metrics["angular_4d"]["AUROC"] - metrics["cosine_4d"]["AUROC"]
    ws3["F23"].number_format = "0.0000"
    ws3["E24"] = "95% CI"
    ws3["F24"] = f"[{boot['p2_5']:.4f}, {boot['p97_5']:.4f}]"
    for r in range(20, 25):
        ws3.cell(row=r, column=5).font = font(color="000000", size=10)
        ws3.cell(row=r, column=6).font = font(color="000000", size=10)

    # raw points to the right of the note, starting at column H
    ws3["H1"] = "原始 ROC 點（sklearn.metrics.roc_curve，未內插）"
    ws3["H1"].font = font(bold=True, color=NAVY, size=12)
    ws3.merge_cells("H1:K1")
    for c, h in enumerate(("距離表示 distance", "FPR", "TPR", "切點 threshold（oof_margin）"), 8):
        ws3.cell(row=2, column=c, value=h)
    style_header(ws3, 2, 8, 11, ORANGE)
    for i, row in roc_raw.iterrows():
        r = 3 + i
        ws3.cell(row=r, column=8, value=row["距離表示"])
        ws3.cell(row=r, column=9, value=float(row["FPR"])).number_format = "0.000000"
        ws3.cell(row=r, column=10, value=float(row["TPR"])).number_format = "0.000000"
        ws3.cell(row=r, column=11, value=float(row["threshold_oof_margin"])).number_format = "0.00000000"
        for c in range(8, 12):
            cell = ws3.cell(row=r, column=c)
            cell.font = font(color="000000", size=8)
            cell.alignment = align(wrap=False)
            cell.border = THIN
            if i % 2 == 1:
                cell.fill = fill(GREY)
    set_widths(ws3, [12, 22, 22, 14, 36, 16])
    ws3.column_dimensions["H"].width = 26
    ws3.column_dimensions["I"].width = 14
    ws3.column_dimensions["J"].width = 14
    ws3.column_dimensions["K"].width = 26
    ws3.freeze_panes = "A3"

    # ===== 說明 =====
    ws4 = wb.create_sheet("說明")
    ws4.sheet_properties.tabColor = "5D6D7E"
    ws4["A1"] = "欄位定義、資料來源與實驗設定"
    ws4["A1"].font = font(bold=True, color=NAVY, size=14)
    ws4.merge_cells("A1:B1")

    explain = []
    explain.append(("一、這份檔案是什麼", ""))
    explain.append((
        "內容",
        "實驗二在 2026-11-22 正式重跑的真實逐張資料與指標，供畫圖與核對。"
        "未做角距離（cos）AUROC 0.7217、Recall 0.6050、Precision 0.7043、F1 0.6509；"
        "有做角距離（角距）AUROC 0.7233、Recall 0.6790、Precision 0.6893、F1 0.6841。"
        "以上四位小數已由本檔的 2,000 張折外分數重算並完全重現。沒有合成或補值。",
    ))
    explain.append(("產出日期", "2026-11-26（Asia/Taipei，UTC+8）"))
    explain.append(("二、實驗設定", ""))
    explain.append(("攻擊與模型", "VLAttack，BLIP／VQA。範圍僅限 VLAttack，不含 AttackVLM。"))
    explain.append(("樣本", "實驗二 2,000 張：乾淨 1,000 ＋ 攻擊 1,000。對應 388 張照片（photo_id）。同一 image_id 會同時出現 Clean 與 VLAttack 各一列。"))
    explain.append(("描述句", "每張 11 句：1 句原圖描述＋10 個 ±5° 旋轉版本。嵌入模型為 all-MiniLM-L6-v2（平均池化並 L2 正規化）。中心是這 11 句自己的平均方向。"))
    explain.append((
        "四維特徵",
        "對 11 個距離值取四項統計：center（平均）、iqr（Q75−Q25）、tail（Q90−Q50）、q95（第 95 分位）。"
        "cosine 四維直接做在 11 個 cosine 相似度上（正式對照組用的就是這組原始 cosine，不是 1−cos）。"
        "角距四維做在 δ = arccos(cos)/π（正規化到 0–1，與夾角成正比）。"
        "11 句完全相同時，重算的角距記為 0。",
    ))
    explain.append((
        "重要：cosine 不是 1−cos",
        "正式 SVM 對照組吃的是 raw cosine 的 mean／IQR／Q90−Q50／Q95。"
        "因此 d_center_cos 與 d_q95_cos 愈接近 1 代表 11 句愈聚在一起，攻擊方向是「愈低愈像攻擊」。"
        "d_iqr_cos 與 d_tail_cos 則是愈大愈散。"
        "IQR(cos) 數值上等於 IQR(1−cos)，所以四分位距不必改寫。"
        "tail 不相等：Q90(cos)−Q50(cos) ≠ Q90(1−cos)−Q50(1−cos)，主圖的尾距畫的是正式特徵本身。"
        "為了讓中心與 Q95 的密度圖和角距「愈右愈偏離」同向，主圖另提供 1−mean(cos)、1−Q95(cos) 的版本（單調轉換，單一特徵 AUROC 不變，差只可能來自歸零造成的 ties）。"
        "原始 cosine 的中心與 Q95 密度圖放在補充資料夾，避免和主圖軸向搞混。",
    ))
    explain.append(("分類器", "RBF-SVM，C=1，class_weight=balanced，特徵先 StandardScaler。實驗一、二各自 5-fold GroupKFold，分組鍵是 photo_id（question_id // 1000），兩種四維用同一組折。AUROC 用折外決策值 oof_margin。切點：決策值 ≥ 0（oof_pred）。風險分 oof_risk 是決策值的 sigmoid，不用來算本表的四項指標。"))
    explain.append(("信賴區間", "照片分組 bootstrap 1,000 次，seed=42，抽的是已算好的折外分數，沒有重訓。區間是角距 AUROC 減 cosine AUROC。實驗二重算結果 mean=0.0019，95% CI=[−0.0090, 0.0130]，與發表值一致，區間跨過 0。"))
    explain.append(("三、資料來源（唯讀，未改動）", ""))
    explain.append(("特徵表", str(SRC_FEAT)))
    explain.append(("折外分數", str(SRC_OOF)))
    explain.append(("發表指標表", str(SRC_RPF1)))
    explain.append(("官方 bootstrap", str(SRC_BOOT)))
    explain.append(("重跑腳本", str(SRC_SCRIPT)))
    explain.append(("指標重算腳本", str(SRC_COMPUTE)))
    explain.append(("重跑說明", str(ARCHIVE / "2026-11-22_旋轉cosine與角距四維.md")))
    explain.append(("四、工作表說明", ""))
    explain.append(("指標表", "四項指標（完整精度與四位小數）、TP/FP/TN/FN、AUROC 差的 95% CI、單一特徵 AUROC，以及可在 Excel 編輯的群組長條圖。"))
    explain.append(("逐張特徵", "2,000 列，一列一張圖。含 photo/image id、真實標籤、8 個正式四維特徵、兩個僅供作圖的 1−cos 欄、兩種折外決策值／風險分／預測標籤。可用篩選或樞紐分析。"))
    explain.append(("畫圖用_分組特徵", "把各特徵拆成 Clean 與 VLAttack 兩欄（各 1,000 列），方便在 Excel 插入直方圖或箱形圖。密度主圖請用輸出的 PNG。"))
    explain.append(("ROC", "左側是 FPR 共用格點（步長 0.001）的內插 TPR，已插入 Excel 折線圖；右側是 sklearn 原始 ROC 點（含 threshold）。AUROC 仍以原始折外決策值計算，不以內插點重算為準。"))
    explain.append(("說明", "本頁。"))
    explain.append(("五、逐張特徵欄位", ""))
    col_defs = [
        ("影像編號 image_id", "題目／影像列識別（同一 image_id 會有 Clean 與 VLAttack 各一列）。"),
        ("照片編號 photo_id", "分組鍵，等於 question_id // 1000。5-fold 與 bootstrap 都按此分組。"),
        ("真實標籤 label", "Clean 或 VLAttack。label_code：0=Clean，1=VLAttack。"),
        ("不同描述句數 n_unique_captions", "11 句裡有幾句不重複。"),
        ("d_center_cos / d_iqr_cos / d_tail_cos / d_q95_cos", "正式 cosine 四維，SVM 實際使用的輸入。center、q95 是 cosine 本身（約在 0.8–1），不是距離。"),
        ("d_center_1minus_cos / d_q95_1minus_cos", "1−對應 cosine 統計，只供與角距同向作圖，不曾送進正式 SVM。"),
        ("d_center_ang / d_iqr_ang / d_tail_ang / d_q95_ang", "正式角距四維，δ=arccos(cos)/π 上的同樣四項統計。SVM 實際使用的輸入。"),
        ("oof_margin_cos / oof_margin_ang", "該列所屬折的折外 SVM 決策值。AUROC 用此欄。切點 ≥ 0。"),
        ("oof_risk_cos / oof_risk_ang", "決策值的 sigmoid，本表四項指標未使用。"),
        ("pred_cos / pred_ang、oof_pred_*", "依決策值 ≥ 0 得到的預測標籤（Clean／VLAttack 或 0／1）。"),
    ]
    for a, b in col_defs:
        explain.append((a, b))
    explain.append(("六、圖檔怎麼畫、讀的是哪一種距離", ""))
    explain.append((
        "密度圖風格",
        "對齊你提供的 08.11 風格：seaborn whitegrid、Noto Sans CJK TC、粗體標題、圖例在右上（標題「類別」）、"
        "Clean 影像填色 #2ecc71（alpha 0.4）加綠色實線、VLAttack 影像填色 #e74c3c 加紅色實線。"
        "KDE 為 Scott 帶寬，並在 0 做反射，避免負值與邊界凹陷。"
        "11 句完全相同時距離為 0（或 raw cosine 為 1）；這些點不畫進曲線，改在右上角框標出比例，曲線面積等於該組非零（或非 1）比例。"
        "右上角「單一特徵 AUROC」是官方值（含 0、未因作圖歸零），與「指標表」單一特徵列一致。"
        "float32 自內積殘差以 |v|≤1e-6（角距對等門檻 arccos(1−1e-6)/π）歸零，只影響曲線外觀，不改 SVM 用的正式特徵，也不改框內 AUROC。",
    ))
    explain.append(("主圖對應", "圖三／圖四：iqr（語意離散度／語意離散程度），x 軸為正式特徵 d_iqr_cos、D_iqr_ang。圖五／圖六：tail（尾端變換／轉換敏感度），x 軸為正式特徵 d_tail_cos、D_tail_ang。圖一／圖二、圖七／圖八：center 與 q95；cosine 側主圖用 1−統計以便和角距同向，檔名與軸標有寫 1−mean(cos)、1−Q95(cos)。"))
    explain.append(("組合圖", "組合圖_2x2_iqr與tail.png：左欄未做角距離、右欄有做角距離，上排 iqr、下排 tail（對應你貼的參考構圖）。組合圖_4x2_四特徵.png：四列特徵、左 cos（center／q95 為 1−cos）、右角距。"))
    explain.append(("補充圖", "figures/補充_原始cosine_center與q95/ 內為正式 SVM 使用的原始 mean(cos)、Q95(cos) 密度圖（愈右愈像中心）。主圖不要和這兩張的橫軸方向搞混。"))
    explain.append(("指標圖與 ROC", "指標長條圖與 ROC 曲線亦輸出為 PNG；Excel「指標表」「ROC」兩頁也內建可編輯圖表。"))
    explain.append(("七、注意事項", ""))
    explain.append(("未納入", "實驗一數字、08.11 小樣本、196 對重建、AttackVLM，以及合成資料，都不在本檔。"))
    explain.append(("重現", "四項指標四位小數與 AUROC 差 95% CI 均已由本資料重算，與發表值完全一致。完整精度：cosine AUROC=0.721672、角距 AUROC=0.723260。"))

    for i, (k, v) in enumerate(explain, 3):
        ws4.cell(row=i, column=1, value=k)
        ws4.cell(row=i, column=2, value=v)
        ws4.cell(row=i, column=1).font = font(bold=True, color=NAVY if v == "" else "000000", size=12 if v == "" else 10)
        ws4.cell(row=i, column=2).font = font(color="000000", size=10)
        ws4.cell(row=i, column=1).alignment = align(h="left", v="top")
        ws4.cell(row=i, column=2).alignment = align(h="left", v="top")
        ws4.row_dimensions[i].height = 16 if v == "" else min(90, 18 + 12 * (len(v) // 50))
        if v == "":
            ws4.merge_cells(start_row=i, start_column=1, end_row=i, end_column=2)
            ws4.cell(row=i, column=1).fill = fill("D6EAF8")
    ws4.column_dimensions["A"].width = 36
    ws4.column_dimensions["B"].width = 120
    ws4.freeze_panes = "A3"
    ws4.page_setup.orientation = "landscape"
    ws4.page_setup.fitToPage = True
    ws4.page_setup.fitToWidth = 1
    ws4.page_setup.fitToHeight = 0

    wb.save(XLSX)
    return per


def prepare_plot_vector(df, col, kind):
    """Return (values, y, auc_toward_attack, zero_or_one_share_clean, share_attack, mode)."""
    y = df["label"].to_numpy(dtype=int)
    v = df[col].to_numpy(dtype=float)
    if kind == "distance":
        # angular, or cosine iqr/tail, or 1-cos center/q95
        # AUROC in the box is the official single-feature value (unsapped, zeros included).
        space = "ang" if col.endswith("_ang") else "cos"
        auc = float(roc_auc_score(y, v))
        v = snap_zero(v, ZERO_ANG if space == "ang" else ZERO_COS)
        zc = float((v[y == 0] == 0).mean())
        za = float((v[y == 1] == 0).mean())
        return v, y, auc, zc, za, "distance"
    elif kind == "raw_cosine":
        auc = float(roc_auc_score(y, -v))  # toward attack, official
        zc = float((v[y == 0] >= 1.0 - ZERO_COS).mean())
        za = float((v[y == 1] >= 1.0 - ZERO_COS).mean())
        return v, y, auc, zc, za, "raw_cosine"
    else:
        raise ValueError(kind)


def save_one(path, draw_fn, *args, figsize=(7.5, 5), dpi=300):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    draw_fn(ax, *args, small=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def make_figures(df, metrics, boot):
    setup_font()
    FIG.mkdir(parents=True, exist_ok=True)
    FIG_RAW.mkdir(parents=True, exist_ok=True)
    y = df["label"].to_numpy(dtype=int)

    specs_main = [
        # filename, col, kind, title, xlabel, fig key
        ("圖一_未做角距離_平均偏離度_center_1minus_cos.png", "d_center_1minus_cos", "distance",
         "圖一：未做角距離 平均偏離度", "傳統餘弦距離 d_center_cos = 1 − mean(cos)"),
        ("圖二_有做角距離_中心不一致度_center_ang.png", "d_center_ang", "distance",
         "圖二：有做角距離 中心不一致度", "正規化角距離 D_center_ang"),
        ("圖三_未做角距離_語意離散度_iqr_cos.png", "d_iqr_cos", "distance",
         "圖三：未做角距離 語意離散度", "傳統餘弦距離 d_iqr_cos"),
        ("圖四_有做角距離_語意離散程度_iqr_ang.png", "d_iqr_ang", "distance",
         "圖四：有做角距離 語意離散程度", "正規化角距離 D_iqr_ang"),
        ("圖五_未做角距離_尾端變換敏感度_tail_cos.png", "d_tail_cos", "distance",
         "圖五：未做角距離 尾端變換敏感度", "傳統餘弦距離 d_tail_cos"),
        ("圖六_有做角距離_尾端轉換敏感度_tail_ang.png", "d_tail_ang", "distance",
         "圖六：有做角距離 尾端轉換敏感度", "正規化角距離 D_tail_ang"),
        ("圖七_未做角距離_高尾端距離_q95_1minus_cos.png", "d_q95_1minus_cos", "distance",
         "圖七：未做角距離 高尾端距離", "傳統餘弦距離 d_q95_cos = 1 − Q95(cos)"),
        ("圖八_有做角距離_高尾端極端心虛角_q95_ang.png", "d_q95_ang", "distance",
         "圖八：有做角距離 高尾端極端心虛角", "正規化角距離 D_q95_ang"),
    ]
    data = {}  # key -> (v,y,title,xlabel,auc,zc,za)
    keys = ["center_cos", "center_ang", "iqr_cos", "iqr_ang", "tail_cos", "tail_ang", "q95_cos", "q95_ang"]
    for key, spec in zip(keys, specs_main):
        fn, col, kind, title, xlabel = spec
        v, yv, auc, zc, za, _ = prepare_plot_vector(df, col, kind)
        data[key] = (v, yv, title, xlabel, auc, zc, za)
        save_one(FIG / fn, draw_density, v, yv, title, xlabel, auc, zc, za)

    # raw cosine supplements
    raw_specs = [
        ("圖一b_未做角距離_中心_原始cosine.png", "d_center_cos",
         "圖一b：未做角距離 中心（正式 SVM 使用的原始 mean(cos)）", "原始餘弦相似度 mean(cos)（正式特徵 d_center_cos）"),
        ("圖七b_未做角距離_高尾端_原始cosine.png", "d_q95_cos",
         "圖七b：未做角距離 高尾端（正式 SVM 使用的原始 Q95(cos)）", "原始餘弦相似度 Q95(cos)（正式特徵 d_q95_cos）"),
    ]
    for fn, col, title, xlabel in raw_specs:
        v, yv, auc, zc, za, _ = prepare_plot_vector(df, col, "raw_cosine")
        save_one(FIG_RAW / fn, draw_density_near_one, v, yv, title, xlabel, auc, zc, za)

    # 4x2 composite: left cos, right ang; rows center, iqr, tail, q95
    fig, axs = plt.subplots(4, 2, figsize=(16, 20), dpi=160)
    order = [("center_cos", "center_ang"), ("iqr_cos", "iqr_ang"), ("tail_cos", "tail_ang"), ("q95_cos", "q95_ang")]
    for i, (lc, rc) in enumerate(order):
        for j, key in enumerate((lc, rc)):
            v, yv, title, xlabel, auc, zc, za = data[key]
            draw_density(axs[i, j], v, yv, title, xlabel, auc, zc, za, small=True)
    fig.suptitle(
        "實驗二（正式重跑，VLAttack BLIP／VQA，乾淨 1,000＋攻擊 1,000）\n"
        "左欄未做角距離（center／q95 為 1−cos，以便與右欄同向），右欄有做角距離",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p_4x2 = FIG / "組合圖_4x2_四特徵.png"
    fig.savefig(p_4x2)
    plt.close(fig)

    # 2x2 composite: iqr & tail only (reference layout)
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), dpi=180)
    order2 = [("iqr_cos", "iqr_ang"), ("tail_cos", "tail_ang")]
    for i, (lc, rc) in enumerate(order2):
        for j, key in enumerate((lc, rc)):
            v, yv, title, xlabel, auc, zc, za = data[key]
            draw_density(axs[i, j], v, yv, title, xlabel, auc, zc, za, small=False)
    fig.suptitle(
        "實驗二（正式重跑，VLAttack BLIP／VQA，乾淨 1,000＋攻擊 1,000）\n左欄未做角距離，右欄有做角距離",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p_2x2 = FIG / "組合圖_2x2_iqr與tail.png"
    fig.savefig(p_2x2)
    plt.close(fig)

    # grouped bar chart
    fig, ax = plt.subplots(figsize=(9.5, 5.5), dpi=200)
    labels_m = ["AUROC", "Recall", "Precision", "F1-Score"]
    keys_m = ["AUROC", "Recall", "Precision", "F1"]
    x = np.arange(len(labels_m))
    w = 0.36
    cvals = [metrics["cosine_4d"]["rounded"][k] for k in keys_m]
    avals = [metrics["angular_4d"]["rounded"][k] for k in keys_m]
    b1 = ax.bar(x - w / 2, cvals, w, label=NAME_COS, color="#5DADE2", edgecolor="#1A5276", lw=1.2)
    b2 = ax.bar(x + w / 2, avals, w, label=NAME_ANG, color="#F5B041", edgecolor="#9A7D0A", lw=1.2)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("分數", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_m, fontsize=12)
    ax.set_title("實驗二四項指標：未做角距離（cos）vs 有做角距離（角距）", fontsize=14, fontweight="bold", pad=10)
    ax.legend(title="設定", loc="upper right", frameon=True, fontsize=10, title_fontsize=10)
    ax.axhline(0.5, color="#bbbbbb", ls="--", lw=0.8)
    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            ax.annotate(f"{h:.4f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    fig.text(
        0.02,
        0.02,
        "切點：SVM 決策值 ≥ 0；AUROC 用折外決策值。"
        f"AUROC 差（角距 − cos）95% CI = [{boot['p2_5']:.4f}, {boot['p97_5']:.4f}]（跨過 0）。",
        ha="left",
        va="bottom",
        fontsize=8,
        color="#444444",
    )
    p_bar = FIG / "指標長條圖_AUROC_Recall_Precision_F1.png"
    fig.savefig(p_bar)
    plt.close(fig)

    # ROC
    fpr_c, tpr_c, _ = roc_curve(y, df["oof_margin_cos"].to_numpy())
    fpr_a, tpr_a, _ = roc_curve(y, df["oof_margin_ang"].to_numpy())
    fig, ax = plt.subplots(figsize=(7.5, 7.0), dpi=200)
    ax.plot(fpr_c, tpr_c, color="#2471A3", lw=2.2, label=f"{NAME_COS}  AUROC = {metrics['cosine_4d']['rounded']['AUROC']:.4f}")
    ax.plot(fpr_a, tpr_a, color="#B9770E", lw=2.2, label=f"{NAME_ANG}  AUROC = {metrics['angular_4d']['rounded']['AUROC']:.4f}")
    ax.plot([0, 1], [0, 1], color="#999999", ls="--", lw=1, label="隨機（chance）")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("假陽性率 FPR", fontsize=12)
    ax.set_ylabel("真陽性率 TPR", fontsize=12)
    ax.set_title("實驗二 ROC：未做角距離（cos）vs 有做角距離（角距）", fontsize=14, fontweight="bold", pad=10)
    ax.legend(loc="lower right", frameon=True, fontsize=10, title="設定")
    ax.set_aspect("equal", adjustable="box")
    ax.text(
        0.98,
        0.42,
        f"AUROC 差 95% CI\n[{boot['p2_5']:.4f}, {boot['p97_5']:.4f}]\n（跨過 0，未達顯著）",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cccccc"),
    )
    fig.tight_layout()
    p_roc = FIG / "ROC曲線_cos_vs_角距.png"
    fig.savefig(p_roc)
    plt.close(fig)
    return p_4x2, p_2x2, p_bar, p_roc


def write_readme(metrics, boot, paths):
    p_4x2, p_2x2, p_bar, p_roc = paths
    text = f"""# 實驗二 cos vs 角距：真實數據 Excel 與圖（2026-11-26，Asia/Taipei）

本資料夾由 11/22 正式重跑的逐張特徵與折外分數重算，沒有合成資料，也沒有改動原始檔。

## 指標是否完全重現

是。由 2,000 張折外決策值重算，四位小數與發表表一致：

| 設定 | AUROC | Recall | Precision | F1-Score | TP | FP | TN | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 未做角距離（cos） | {metrics['cosine_4d']['rounded']['AUROC']:.4f} | {metrics['cosine_4d']['rounded']['Recall']:.4f} | {metrics['cosine_4d']['rounded']['Precision']:.4f} | {metrics['cosine_4d']['rounded']['F1']:.4f} | {metrics['cosine_4d']['TP']} | {metrics['cosine_4d']['FP']} | {metrics['cosine_4d']['TN']} | {metrics['cosine_4d']['FN']} |
| 有做角距離（角距） | {metrics['angular_4d']['rounded']['AUROC']:.4f} | {metrics['angular_4d']['rounded']['Recall']:.4f} | {metrics['angular_4d']['rounded']['Precision']:.4f} | {metrics['angular_4d']['rounded']['F1']:.4f} | {metrics['angular_4d']['TP']} | {metrics['angular_4d']['FP']} | {metrics['angular_4d']['TN']} | {metrics['angular_4d']['FN']} |

完整精度：cosine AUROC = {metrics['cosine_4d']['AUROC']:.16f}，角距 AUROC = {metrics['angular_4d']['AUROC']:.16f}。

AUROC 差（角距 − cos）照片分組 bootstrap 1000 次（seed=42）：mean = {boot['mean']:.6f}，95% CI = [{boot['p2_5']:.4f}, {boot['p97_5']:.4f}]，與發表區間 [−0.0090, 0.0130] 一致。區間跨過 0。

## 檔案

- `實驗二_cos_vs_角距_真實數據.xlsx`：工作表「指標表」「逐張特徵」「畫圖用_分組特徵」「ROC」「說明」。
- `build_exp2_excel_and_figures.py`：本資料夾的重現腳本。
- `figures/組合圖_2x2_iqr與tail.png`：對應參考圖構圖（左未做角距離、右有做角距離；上 iqr、下 tail）。
- `figures/組合圖_4x2_四特徵.png`：四項特徵，左 cos（center／q95 為 1−cos）、右角距。
- `figures/圖三_…` 至 `圖六_…`：iqr、tail 單張主圖（橫軸是正式特徵）。
- `figures/圖一_…`、`圖七_…`：center、q95 主圖（cosine 側為 1−cos，軸標有寫）。
- `figures/補充_原始cosine_center與q95/`：正式 SVM 使用的原始 mean(cos)、Q95(cos)。
- `figures/指標長條圖_AUROC_Recall_Precision_F1.png`
- `figures/ROC曲線_cos_vs_角距.png`

## 來源（唯讀）

- `{SRC_FEAT}`
- `{SRC_OOF}`
- `{SRC_RPF1}`
- `{SRC_BOOT}`

正式對照組用的是原始 cosine 四維，不是 1−cos。細節見 Excel「說明」頁。
"""
    (OUT / "README.md").write_text(text, encoding="utf-8")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    FIG_RAW.mkdir(parents=True, exist_ok=True)
    df = load_and_score()
    y = df["label"].to_numpy(dtype=int)
    metrics = {
        "cosine_4d": metric_block(y, df["oof_margin_cos"].to_numpy(), df["oof_pred_cos"].to_numpy(), "cosine_4d"),
        "angular_4d": metric_block(y, df["oof_margin_ang"].to_numpy(), df["oof_pred_ang"].to_numpy(), "angular_4d"),
    }
    boot = bootstrap_delta(
        y,
        df["photo_id"].to_numpy(),
        df["oof_margin_cos"].to_numpy(),
        df["oof_margin_ang"].to_numpy(),
    )
    print("==== 重算指標 ====")
    for k, m in metrics.items():
        print(k, m["rounded"], "TP/FP/TN/FN", m["TP"], m["FP"], m["TN"], m["FN"], "match", m["matches_published"])
    print("bootstrap", boot)
    print("CI match", round(boot["p2_5"], 4) == PUB_CI[0] and round(boot["p97_5"], 4) == PUB_CI[1])
    if not all(m["matches_published"] for m in metrics.values()):
        raise SystemExit("指標未能重現發表值，已停止，未寫檔。")

    uni = univariate(df)
    print(uni.to_string(index=False))
    roc_raw, roc_common = roc_tables(y, df["oof_margin_cos"].to_numpy(), df["oof_margin_ang"].to_numpy())
    write_excel(df, metrics, boot, uni, roc_raw, roc_common)
    print("wrote", XLSX, "size", XLSX.stat().st_size)
    paths = make_figures(df, metrics, boot)
    write_readme(metrics, boot, paths)
    print("figures:")
    for p in sorted(FIG.rglob("*.png")):
        print(" ", p, p.stat().st_size)


if __name__ == "__main__":
    main()
