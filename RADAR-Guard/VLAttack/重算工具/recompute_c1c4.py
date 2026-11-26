#!/usr/bin/env python3
"""Recompute table 4.28 C1-C4 counts and try to explain 困難題. Write-only under check_c1c4/."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("/workspace/radar_dossier/check_c1c4")
SCORED = Path("/workspace/radar/BLIP_attack/output/VQA/multi_captioner")
ASR = Path("/workspace/radar_dossier/asr_redefine/per_question.csv")
ANN_VAL = Path("/workspace/vqa_ann/v2_mscoco_val2014_annotations.json")
ANN_TRAIN = Path("/workspace/vqa_ann/v2_mscoco_train2014_annotations.json")

TABLE = {
    "exp1": {"n": {"C1": 137, "C2": 78, "C3": 82, "C4": 164},
             "hard": {"C1": 32, "C2": 30, "C3": 22, "C4": 46}},
    "exp2": {"n": {"C1": 220, "C2": 175, "C3": 163, "C4": 442},
             "hard": {"C1": 53, "C2": 57, "C3": 37, "C4": 128}},
    "exp3": {"n": {"C1": 198, "C2": 162, "C3": 149, "C4": 491},
             "hard": {"C1": 50, "C2": 52, "C3": 26, "C4": 126}},
}

def norm(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip().str.lower()


def load_exp(exp: str) -> pd.DataFrame:
    p = SCORED / exp / "blip" / "sigma08_fixed" / "rows_scored.csv"
    df = pd.read_csv(p)
    df["exp"] = exp
    df["qid"] = df["image_id"].astype(int)
    df["photo_id"] = df["photo_id"].astype(int)
    for c in ["pred_ref", "pred_0.8", "original_prediction", "served_full"]:
        df[c + "_n"] = norm(df[c])
    df["fooled_png_vs_orig"] = (df["label"] == 1) & (df["pred_ref_n"] != df["original_prediction_n"])
    df["log_attack_success"] = df["log_attack_success"].astype(bool)
    df["fooled_before"] = df["fooled_before"].astype(bool)
    # clean pred_ref for same image_id (if present)
    clean = df[df["label"] == 0][["qid", "pred_ref_n"]].rename(columns={"pred_ref_n": "clean_pred_ref_n"})
    df = df.merge(clean, on="qid", how="left")
    df["fooled_png_vs_cleanref"] = (df["label"] == 1) & df["clean_pred_ref_n"].notna() & (df["pred_ref_n"] != df["clean_pred_ref_n"])
    df["blur_ne_orig"] = df["pred_0.8_n"] != df["original_prediction_n"]
    df["blur_ne_cleanref"] = df["clean_pred_ref_n"].notna() & (df["pred_0.8_n"] != df["clean_pred_ref_n"])
    df["blur_ne_atkref"] = df["pred_0.8_n"] != df["pred_ref_n"]
    df["log_and_png"] = df["log_attack_success"] & df["fooled_before"]
    df["log_and_png_orig"] = df["log_attack_success"] & df["fooled_png_vs_orig"]
    return df


def assign_c(atk: pd.DataFrame, pop_mask: pd.Series, blur_col: str) -> pd.Series:
    """Chapter C1-C4 on a chosen population and blur-vs-original column."""
    cat = pd.Series(index=atk.index, dtype="object")
    m = pop_mask
    cat[m & (atk["bin"] == "low")] = "C1"
    cat[m & (atk["bin"] == "mid") & atk[blur_col]] = "C2"
    cat[m & (atk["bin"] == "mid") & ~atk[blur_col]] = "C3"
    cat[m & (atk["bin"] == "high")] = "C4"
    return cat


def counts(cat: pd.Series) -> dict:
    vc = cat.value_counts(dropna=True)
    return {k: int(vc.get(k, 0)) for k in ["C1", "C2", "C3", "C4"]}


def match_table(got: dict, exp: str) -> bool:
    return got == TABLE[exp]["n"]


def load_asr() -> pd.DataFrame:
    a = pd.read_csv(ASR)
    a = a[a["model"] == "BLIP-VQA"].copy()
    a["qid"] = a["question_id"].astype(int)
    return a


def load_ann_for(qids: set) -> dict:
    """Load VQAv2 anns for needed question_ids only."""
    out = {}
    for path in (ANN_VAL, ANN_TRAIN):
        print(f"loading {path.name} ...", flush=True)
        data = json.load(open(path))
        for ann in data["annotations"]:
            qid = int(ann["question_id"])
            if qid in qids and qid not in out:
                answers = [x["answer"].strip().lower() for x in ann["answers"]]
                confs = [x.get("answer_confidence", "") for x in ann["answers"]]
                ctr = Counter(answers)
                top_ans, top_n = ctr.most_common(1)[0]
                out[qid] = {
                    "answer_type": ann.get("answer_type"),
                    "question_type": ann.get("question_type"),
                    "mc_answer": str(ann.get("multiple_choice_answer", "")).strip().lower(),
                    "n_unique": len(ctr),
                    "top_n": top_n,
                    "top_ans": top_ans,
                    "soft_top": min(top_n / 3.0, 1.0),
                    "n_maybe": sum(c != "yes" for c in confs),
                    "n_conf_no": sum(c == "no" for c in confs),
                }
        del data
    return out


def main():
    asr = load_asr()
    frames = {e: load_exp(e) for e in ("exp1", "exp2", "exp3")}

    # --- basic agreement between flags ---
    flag_rows = []
    for exp, df in frames.items():
        atk = df[df["label"] == 1].copy()
        flag_rows.append({
            "exp": exp, "n_atk": len(atk),
            "fooled_before_col": int(atk["fooled_before"].sum()),
            "fooled_png_vs_orig": int(atk["fooled_png_vs_orig"].sum()),
            "fooled_png_vs_cleanref": int(atk["fooled_png_vs_cleanref"].sum()) if atk["clean_pred_ref_n"].notna().any() else None,
            "log_true": int(atk["log_attack_success"].sum()),
            "log_and_png_fooled_before": int(atk["log_and_png"].sum()),
            "log_and_png_vs_orig": int(atk["log_and_png_orig"].sum()),
            "log_only": int((atk["log_attack_success"] & ~atk["fooled_before"]).sum()),
            "png_only": int((~atk["log_attack_success"] & atk["fooled_before"]).sum()),
            "orig_ne_cleanref": int((atk["original_prediction_n"] != atk["clean_pred_ref_n"]).sum()) if atk["clean_pred_ref_n"].notna().any() else None,
            "bin_low": int((atk["bin"] == "low").sum()),
            "bin_mid": int((atk["bin"] == "mid").sum()),
            "bin_high": int((atk["bin"] == "high").sum()),
        })
    flag_df = pd.DataFrame(flag_rows)
    print("\n=== FLAG COUNTS ===")
    print(flag_df.to_string(index=False))
    flag_df.to_csv(OUT / "flag_counts.csv", index=False)

    # disagreements log vs png
    disag = []
    for exp, df in frames.items():
        atk = df[df["label"] == 1]
        d = atk[atk["log_attack_success"] != atk["fooled_before"]]
        for _, r in d.iterrows():
            disag.append({
                "exp": exp, "qid": r["qid"], "photo_id": r["photo_id"], "bin": r["bin"],
                "log": bool(r["log_attack_success"]), "fooled_before": bool(r["fooled_before"]),
                "original_prediction": r["original_prediction"], "pred_ref": r["pred_ref"],
                "pred_0.8": r["pred_0.8"], "clean_pred_ref": r.get("clean_pred_ref_n"),
            })
    disag_df = pd.DataFrame(disag)
    disag_df.to_csv(OUT / "log_vs_png_disagreements.csv", index=False)
    print("\n=== LOG vs PNG disagreements ===")
    print(disag_df.to_string(index=False) if len(disag_df) else "(none)")

    # --- C1-C4 under several population x blur definitions ---
    pops = {
        "all_attack": lambda a: pd.Series(True, index=a.index),
        "fooled_before": lambda a: a["fooled_before"],
        "fooled_png_vs_orig": lambda a: a["fooled_png_vs_orig"],
        "log_only": lambda a: a["log_attack_success"],
        "log_and_png": lambda a: a["log_and_png"],
        "log_and_png_vs_orig": lambda a: a["log_and_png_orig"],
        "fooled_png_vs_cleanref": lambda a: a["fooled_png_vs_cleanref"],
    }
    blur_cols = {
        "blur_ne_orig": "blur_ne_orig",
        "blur_ne_cleanref": "blur_ne_cleanref",
        "blur_ne_atkref": "blur_ne_atkref",
    }

    recs = []
    best_for_exp = {}
    assigned = {}  # exp -> (pop, blur, cat series on atk index)
    for exp, df in frames.items():
        atk = df[df["label"] == 1].copy()
        target = TABLE[exp]["n"]
        for pop_name, pop_fn in pops.items():
            pop = pop_fn(atk)
            for blur_name, blur_col in blur_cols.items():
                if blur_col == "blur_ne_cleanref" and atk["clean_pred_ref_n"].isna().all():
                    continue
                cat = assign_c(atk, pop, blur_col)
                got = counts(cat)
                ok = got == target
                recs.append({
                    "exp": exp, "population": pop_name, "blur_rule": blur_name,
                    "C1": got["C1"], "C2": got["C2"], "C3": got["C3"], "C4": got["C4"],
                    "sum": sum(got.values()),
                    "match_table": ok,
                    "delta": {k: got[k] - target[k] for k in got},
                })
                if ok:
                    assigned.setdefault(exp, []).append((pop_name, blur_name, atk, cat))
        # also print closest
        exp_recs = [r for r in recs if r["exp"] == exp]
        def l1(r):
            return sum(abs(r[k] - target[k]) for k in ["C1", "C2", "C3", "C4"])
        exp_recs_sorted = sorted(exp_recs, key=l1)
        best_for_exp[exp] = exp_recs_sorted[0]

    rec_df = pd.DataFrame(recs)
    rec_df["delta"] = rec_df["delta"].astype(str)
    rec_df.to_csv(OUT / "c1c4_definition_sweep.csv", index=False)
    print("\n=== DEFINITION SWEEP (matches and closest) ===")
    matches = rec_df[rec_df["match_table"]]
    print("EXACT MATCHES:")
    print(matches.to_string(index=False) if len(matches) else "(none)")
    print("\nCLOSEST PER EXP:")
    for exp, r in best_for_exp.items():
        print(exp, r)

    # default documented rule for later hard-proxy tests:
    # exp1: log_and_png, blur_ne_orig (S6 + 書面報告)
    # exp2/3: all_attack, blur_ne_orig (書面報告: 全部 1000 張)
    default_assigned = {}
    for exp, df in frames.items():
        atk = df[df["label"] == 1].copy()
        if exp == "exp1":
            pop = atk["log_and_png"]
        else:
            pop = pd.Series(True, index=atk.index)
        cat = assign_c(atk, pop, "blur_ne_orig")
        default_assigned[exp] = (atk, cat)
        atk = atk.copy()
        atk["chapter_cat"] = cat
        atk[atk["chapter_cat"].notna()][["qid", "photo_id", "bin", "fooled_before", "log_attack_success",
            "original_prediction", "pred_ref", "pred_0.8", "chapter_cat"]].to_csv(
            OUT / f"rows_{exp}_default_cats.csv", index=False)

    print("\n=== DEFAULT RULE COUNTS (exp1=log&PNG; exp2/3=all attack; blur!=original_prediction) ===")
    for exp, (atk, cat) in default_assigned.items():
        got = counts(cat)
        tgt = TABLE[exp]["n"]
        print(exp, got, "table", tgt, "match", got == tgt, "sum", sum(got.values()))

    # join ASR soft scores
    print("\n=== JOIN ASR / try hard proxies ===")
    asr_atk = asr.copy()
    # per_question has 2000 rows per exp (clean+attack?). filter later
    print("asr rows", len(asr_atk), "exps", asr_atk["exp"].value_counts().to_dict())
    print("asr cols", asr_atk.columns.tolist())

    qids = set()
    for exp, df in frames.items():
        qids.update(df["qid"].astype(int).tolist())
    print("unique qids in scored rows", len(qids))

    # unique qid stats for 2538 investigation
    uniq_rows = []
    all_atk_q = {}
    for exp, df in frames.items():
        atk = df[df["label"] == 1]
        cln = df[df["label"] == 0]
        all_atk_q[exp] = set(atk["qid"].astype(int))
        uniq_rows.append({
            "exp": exp,
            "atk_qids": atk["qid"].nunique(),
            "clean_qids": cln["qid"].nunique() if len(cln) else 0,
            "atk_photos": atk["photo_id"].nunique(),
        })
    print("qid sets:", {e: len(s) for e, s in all_atk_q.items()})
    print("exp1∩exp2", len(all_atk_q["exp1"] & all_atk_q["exp2"]))
    print("exp1∩exp3", len(all_atk_q["exp1"] & all_atk_q["exp3"]))
    print("exp2∩exp3", len(all_atk_q["exp2"] & all_atk_q["exp3"]))
    print("union atk 1+2", len(all_atk_q["exp1"] | all_atk_q["exp2"]))
    print("union atk 1+2+3", len(all_atk_q["exp1"] | all_atk_q["exp2"] | all_atk_q["exp3"]))
    # populations that might equal 2538
    e1_all = all_atk_q["exp1"]
    e1_fooled = set(frames["exp1"].query("label==1 and fooled_before")["qid"].astype(int))
    e1_logpng = set(frames["exp1"].query("label==1 and log_and_png")["qid"].astype(int))
    candidates = {
        "e1all+e2+e3": e1_all | all_atk_q["exp2"] | all_atk_q["exp3"],
        "e1fooled+e2+e3": e1_fooled | all_atk_q["exp2"] | all_atk_q["exp3"],
        "e1logpng+e2+e3": e1_logpng | all_atk_q["exp2"] | all_atk_q["exp3"],
        "e1all+e2": e1_all | all_atk_q["exp2"],
        "e1all+e3": e1_all | all_atk_q["exp3"],
        "len e1all+e2+e3 rows (with multiplicity)": None,
    }
    print("union sizes:", {k: (len(v) if v is not None else 'n/a') for k, v in candidates.items()})
    print("multiplicity 461+1000+1000", 461+1000+1000, "1000+1000+1000", 3000,
          "1000+1000+538", 2538, "1000+1000+539", 2539, "fooled+e2+e3 multiplicity",
          len(e1_fooled)+1000+1000)

    # Load annotations
    ann = load_ann_for(qids)
    print("anns found", len(ann), "of", len(qids), "missing", len(qids - set(ann)))

    # attach to default cats and try proxies
    proxy_defs = []
    hard_match_rows = []
    missing_ann_rows = []

    for exp, (atk, cat) in default_assigned.items():
        a = atk.copy()
        a["chapter_cat"] = cat
        # asr join (attacked rows)
        sub = asr_atk[(asr_atk["exp"] == exp)]
        # per_question may have only attack or both; image-level join
        # If 2000 rows, there may be clean+attack distinguished? per_question doesn't have label.
        # For exp3 maybe 1000 attack + 1000? It said 2000 per exp. For exp3 no clean in scored? 
        # asr compute probably always scored the attacked PNG and the clean answer.
        # There's one row per question (the attack evaluation), 1000 per exp? But it said 2000...
        # Wait: 2000 per exp for BLIP might include something else, or clean and attack as separate? 
        # We'll inspect uniquely.
        a = a.merge(sub[["qid", "clean_answer", "attacked_answer", "clean_soft", "atk_soft",
                          "clean_strict", "atk_strict", "old_log", "old_png", "old_orig",
                          "new_success", "top_human"]], on="qid", how="left", suffixes=("", "_asr"))
        def ann_get(qid, key, default=None):
            rec = ann.get(int(qid))
            return rec[key] if rec else default
        a["ann_found"] = a["qid"].map(lambda q: int(q) in ann)
        a["answer_type"] = a["qid"].map(lambda q: ann_get(q, "answer_type"))
        a["question_type"] = a["qid"].map(lambda q: ann_get(q, "question_type"))
        a["top_n"] = a["qid"].map(lambda q: ann_get(q, "top_n"))
        a["n_unique"] = a["qid"].map(lambda q: ann_get(q, "n_unique"))
        a["soft_top"] = a["qid"].map(lambda q: ann_get(q, "soft_top"))
        a["n_maybe"] = a["qid"].map(lambda q: ann_get(q, "n_maybe"))
        a["mc_answer"] = a["qid"].map(lambda q: ann_get(q, "mc_answer"))
        use = a[a["chapter_cat"].notna()].copy()
        # hard proxies on the C1-C4 population
        proxies = {
            "clean_soft<1": use["clean_soft"] < 1,
            "clean_soft==0": use["clean_soft"] == 0,
            "clean_soft<0.5": use["clean_soft"] < 0.5,
            "clean_soft<=0.33": use["clean_soft"] <= 0.34,
            "soft_top<1": use["soft_top"] < 1,
            "soft_top<=0.66": use["soft_top"] <= 0.67,
            "top_n<=6": use["top_n"] <= 6,
            "top_n<=5": use["top_n"] <= 5,
            "top_n<=4": use["top_n"] <= 4,
            "top_n<=7": use["top_n"] <= 7,
            "top_n<=8": use["top_n"] <= 8,
            "top_n<=9": use["top_n"] <= 9,
            "n_unique>=3": use["n_unique"] >= 3,
            "n_unique>=4": use["n_unique"] >= 4,
            "n_unique>=5": use["n_unique"] >= 5,
            "n_maybe>=1": use["n_maybe"] >= 1,
            "n_maybe>=2": use["n_maybe"] >= 2,
            "n_maybe>=3": use["n_maybe"] >= 3,
            "clean_strict==0": use["clean_strict"] == 0,
            "answer_type=number": use["answer_type"] == "number",
            "answer_type=other": use["answer_type"] == "other",
            "answer_type=yes/no": use["answer_type"] == "yes/no",
            "new_success": use["new_success"] == True,
        }
        target_h = TABLE[exp]["hard"]
        for pname, mask in proxies.items():
            mask = mask.fillna(False)
            got_h = {}
            for c in ["C1", "C2", "C3", "C4"]:
                got_h[c] = int(((use["chapter_cat"] == c) & mask).sum())
            l1 = sum(abs(got_h[c] - target_h[c]) for c in got_h)
            hard_match_rows.append({
                "exp": exp, "proxy": pname,
                "C1": got_h["C1"], "C2": got_h["C2"], "C3": got_h["C3"], "C4": got_h["C4"],
                "sum": sum(got_h.values()),
                "table": f"{target_h['C1']}/{target_h['C2']}/{target_h['C3']}/{target_h['C4']}",
                "l1": l1, "exact": l1 == 0,
            })
        use.to_csv(OUT / f"rows_{exp}_with_ann.csv", index=False)
        missing_ann_rows.append({"exp": exp, "n_pop": len(use), "ann_missing": int((~use["ann_found"]).sum()),
                                 "asr_missing": int(use["clean_soft"].isna().sum())})

    hard_df = pd.DataFrame(hard_match_rows)
    hard_df.to_csv(OUT / "hard_proxy_sweep.csv", index=False)
    print("\n=== HARD PROXY closest ===")
    print(pd.DataFrame(missing_ann_rows).to_string(index=False))
    for exp in ("exp1", "exp2", "exp3"):
        sub = hard_df[hard_df["exp"] == exp].sort_values("l1")
        print(f"\n{exp} top 8 proxies:")
        print(sub.head(8).to_string(index=False))
        exact = sub[sub["exact"]]
        if len(exact):
            print("EXACT", exact["proxy"].tolist())

    # Also try hard proxies on ALL-attack population for exp1 (for 2538 investigation)
    # Global unique annotation hard rate
    if ann:
        recs_ann = list(ann.values())
        def rate(pred):
            return sum(1 for r in recs_ann if pred(r)), len(recs_ann)
        print("\n=== annotation-wide rates on scored unique qids ===")
        for name, pred in [
            ("soft_top<1", lambda r: r["soft_top"] < 1),
            ("top_n<=6", lambda r: r["top_n"] <= 6),
            ("top_n<=5", lambda r: r["top_n"] <= 5),
            ("n_unique>=3", lambda r: r["n_unique"] >= 3),
            ("n_maybe>=1", lambda r: r["n_maybe"] >= 1),
        ]:
            h, n = rate(pred)
            print(f"  {name}: {h}/{n} = {h/n:.3f}")

    # If we pool default populations with multiplicity
    all_pop_q = []
    for exp, (atk, cat) in default_assigned.items():
        use = atk[cat.notna()]
        all_pop_q.extend(use["qid"].tolist())
    print("default pop multiplicity", len(all_pop_q), "unique", len(set(all_pop_q)))

    # Write a compact markdown summary (facts only; narrative in agent report)
    (OUT / "NOTES.txt").write_text(
        "See flag_counts.csv, c1c4_definition_sweep.csv, hard_proxy_sweep.csv, "
        "log_vs_png_disagreements.csv, rows_exp*_default_cats.csv\n",
        encoding="utf-8",
    )
    print("\nDone. Outputs in", OUT)


if __name__ == "__main__":
    main()
