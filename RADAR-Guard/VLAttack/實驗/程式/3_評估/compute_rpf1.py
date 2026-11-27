"""從 11/22 正式重跑的折外預測 (oof_scores.csv) 算 AUROC / Recall / Precision / F1。
預設切點：SVM 決策值 >= 0 (oof_pred)。AUROC 用折外決策值 (oof_margin)。不重訓。"""
import os
import sys, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score
src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(os.environ.get("VLATTACK_OOF_SCORES", "/workspace/radar_dossier/code/VLAttack_final/BLIP_attack/output/VQA/rotation_cos_vs_ang/oof_scores.csv"))
d = pd.read_csv(src)
rows = []
name = {"cosine_4d": "未做角距離 (cosine 四維)", "angular_4d": "有做角距離 (角距四維)"}
for (e, f), g in d.groupby(["experiment", "feature"], sort=False):
    rows.append(dict(experiment=e, configuration=name[f], n=len(g),
        AUROC=round(roc_auc_score(g.label, g.oof_margin), 4),
        Recall=round(recall_score(g.label, g.oof_pred), 4),
        Precision=round(precision_score(g.label, g.oof_pred), 4),
        F1=round(f1_score(g.label, g.oof_pred), 4)))
out = pd.DataFrame(rows); print(out.to_string(index=False))
out.to_csv(Path(__file__).with_name("formal_1122_rpf1.csv"), index=False, encoding="utf-8-sig")
