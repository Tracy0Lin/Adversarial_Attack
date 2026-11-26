"""ASR redefinition (option 3): reference = VQAv2 human answers. CPU only, reads existing outputs."""
import json, re, sys
from pathlib import Path
import numpy as np, pandas as pd

OUT = Path(__file__).resolve().parent
BASE = Path("/workspace/radar_dossier/code/VLAttack_final/BLIP_attack/output/VQA")
ANN = Path("/workspace/vqa_ann")
SRC = {
 ("BLIP-VQA", "exp1"): BASE/"multi_captioner/exp1/blip/sigma08_fixed/rows_scored.csv",
 ("BLIP-VQA", "exp2"): BASE/"multi_captioner/exp2/blip/sigma08_fixed/rows_scored.csv",
 ("BLIP-VQA", "exp3"): BASE/"multi_captioner/exp3/blip/sigma08_fixed/rows_scored.csv",
 ("Florence-2", "exp1"): BASE/"florence_vlattack/exp1/florence2/vqa_florence2_sigma08_fixed/rows_scored.csv",
 ("Florence-2", "exp2"): BASE/"florence_vlattack/exp2/florence2/vqa_florence2_sigma08_fixed/rows_scored.csv",
 ("Florence-2", "exp3"): BASE/"florence_vlattack/exp3/florence2/vqa_florence2_sigma08_fixed/rows_scored.csv",
}
REPORT = {  # 報告 rev3 數字: (初始, 僅第一階段, 完整)
 ("BLIP-VQA","exp1"): (46.2, 29.8, 24.1), ("BLIP-VQA","exp2"): (100, 55.8, 40.9), ("BLIP-VQA","exp3"): (100, 50.9, 36.0),
 ("Florence-2","exp1"): (61.7, 37.3, 32.8), ("Florence-2","exp2"): (100, 56.5, 46.7), ("Florence-2","exp3"): (100, 56.1, 48.7)}
REPORT_S2 = {("BLIP-VQA","exp2"): 53.9, ("Florence-2","exp2"): 68.7}

# ---- official VQA normalization (vqaEval.py) ----
contractions = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't", "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't", "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's", "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've", "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't", "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've", "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've", "somebody'd": "somebodyd", "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll": "somebody'll", "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've", "someone'dve": "someone'd've", "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've", "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's", "thered": "there'd", "thered've": "there'd've", "there'dve": "there'd've", "therere": "there're", "theres": "there's", "theyd": "they'd", "theyd've": "they'd've", "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't", "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't", "whatll": "what'll", "whatre": "what're", "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's", "whereve": "where've", "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll", "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've", "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've", "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've", "youll": "you'll", "youre": "you're", "youve": "you've"}
manualMap = {'none': '0','zero': '0','one': '1','two': '2','three': '3','four': '4','five': '5','six': '6','seven': '7','eight': '8','nine': '9','ten': '10'}
articles = ['a', 'an', 'the']
periodStrip = re.compile(r"(?!<=\d)(\.)(?!\d)")
commaStrip = re.compile(r"(\d)(,)(\d)")
punct = [';', r"/", '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\', '_', '-', '>', '<', '@', '`', ',', '?', '!']
def proc_punct(t):
    out = t
    for p in punct:
        if (p + ' ' in t or ' ' + p in t) or (re.search(commaStrip, t) is not None):
            out = out.replace(p, '')
        else:
            out = out.replace(p, ' ')
    return periodStrip.sub("", out, re.UNICODE)
def proc_digit(t):
    words = []
    for w in t.lower().split():
        w = manualMap.setdefault(w, w)
        if w not in articles: words.append(w)
    return " ".join(contractions.get(w, w) for w in words)
def norm(a):
    a = str(a).replace('\n', ' ').replace('\t', ' ').strip()
    return proc_digit(proc_punct(a)).strip()

# ---- annotations ----
qa = {}
for f in ("v2_mscoco_train2014_annotations.json", "v2_mscoco_val2014_annotations.json"):
    for a in json.load(open(ANN/f))["annotations"]:
        qa[a["question_id"]] = ([norm(x["answer"]) for x in a["answers"]], norm(a["multiple_choice_answer"]), f[12:21])
def soft(ans, qid):
    hs = qa[qid][0]; a = norm(ans)
    return min(sum(h == a for h in hs) / 3.0, 1.0)
def soft_official(ans, qid):  # 10 選 9 平均
    hs = qa[qid][0]; a = norm(ans)
    return float(np.mean([min(sum(h == a for j, h in enumerate(hs) if j != i) / 3.0, 1.0) for i in range(len(hs))]))
def strict(ans, qid): return float(norm(ans) == qa[qid][1])

VARIANTS = {"A0_無防禦": "none", "A1_僅第一階段(High攔截)": "s1_high", "A1b_Mid+High攔截": "s1_midhigh",
            "A2_僅第二階段(含θdet攔截)": "s2_only", "A2b_僅第二階段(全部σ0.8修復)": "s2_mitigate_only",
            "A3m_完整(Low放行/Mid修復/High攔截)": "full_mitigate_only", "A3_完整+Midθdet攔截": "full"}
def decide(df, mode, ts=0.8, td=0.8):  # 與 analyze.decide 相同
    ref = df["pred_ref"].values; n = len(df)
    flip = (df[f"pred_{td}"] != df["pred_ref"]).values
    mit = df[f"pred_{ts}"].values; b = df["bin"].values
    low, mid, high = b == "low", b == "mid", b == "high"
    blocked = np.zeros(n, bool); served = ref.copy()
    if mode == "s1_high": blocked = high
    elif mode == "s1_midhigh": blocked = mid | high
    elif mode == "s2_only": blocked = flip; served = np.where(blocked, ref, mit)
    elif mode == "s2_mitigate_only": served = mit
    elif mode == "full": blocked = high | (mid & flip); served = np.where(mid & ~blocked, mit, ref)
    elif mode == "full_mitigate_only": blocked = high; served = np.where(mid, mit, ref)
    return blocked, served

rng = np.random.default_rng(42)
def photo_boot(photos, x, B=1000):
    ph = np.asarray(photos); x = np.asarray(x, float); u = np.unique(ph)
    idx = {p: np.where(ph == p)[0] for p in u}; vals = []
    for _ in range(B):
        s = rng.choice(u, len(u)); ii = np.concatenate([idx[p] for p in s]); vals.append(x[ii].mean())
    return np.percentile(vals, [2.5, 97.5]) * 100

rows, abl, flips_all, per_q = [], [], [], []
missing = {}
for (model, exp), path in SRC.items():
    df = pd.read_csv(path, dtype=str, keep_default_na=False); df["label"] = df.label.astype(int)
    df["qid"] = df.image_id.astype(int)
    miss = sorted(set(df.qid) - set(qa)); missing[f"{model}/{exp}"] = len(miss)
    df = df[df.qid.isin(qa)]
    atk = df[df.label == 1].set_index("image_id"); cln = df[df.label == 0].set_index("image_id")
    has_clean = len(cln) > 0
    a = atk.copy()
    a["clean_src"] = "PNG重推論(pred_ref of _1_ori)" if has_clean else "log original_prediction"
    a["clean_ans"] = cln.loc[a.index, "pred_ref"] if has_clean else a["original_prediction"]
    q = a.qid.values
    for col, src in (("clean", "clean_ans"), ("atk", "pred_ref")):
        a[f"{col}_soft"] = [soft(x, i) for x, i in zip(a[src], q)]
        a[f"{col}_off"] = [soft_official(x, i) for x, i in zip(a[src], q)]
        a[f"{col}_strict"] = [strict(x, i) for x, i in zip(a[src], q)]
    # 舊定義
    old_log = a.log_attack_success.str.lower().eq("true")
    old_png = a.pred_ref != a.clean_ans
    old_orig = a.pred_ref != a.original_prediction
    N_all = len(a)
    defs = {
      "soft>0 (主)": (a.clean_soft > 0, a.atk_soft == 0, lambda s, i: soft(s, i) == 0),
      "soft≥0.5": (a.clean_soft >= 0.5, a.atk_soft == 0, lambda s, i: soft(s, i) == 0),
      "soft=1": (a.clean_soft >= 1, a.atk_soft == 0, lambda s, i: soft(s, i) == 0),
      "門檻0.5(乾淨≥0.5,攻擊後<0.5)": (a.clean_soft >= 0.5, a.atk_soft < 0.5, lambda s, i: soft(s, i) < 0.5),
      "門檻1(乾淨=1,攻擊後<1)": (a.clean_soft >= 1, a.atk_soft < 1, lambda s, i: soft(s, i) < 1),
      "官方10選9 soft>0": (a.clean_off > 0, a.atk_off == 0, lambda s, i: soft_official(s, i) == 0),
      "strict(=多數答案)": (a.clean_strict == 1, a.atk_strict == 0, lambda s, i: strict(s, i) == 0),
    }
    # report-comparable old full numbers from current pipeline (served != original_prediction, /1000)
    full_df = df.copy()
    for vname, mode in VARIANTS.items():
        bl, sv = decide(df, mode)
        d = df.assign(blocked=bl, served=sv)
        da = d[d.label == 1].set_index("image_id").loc[a.index]
        dc = d[d.label == 0].set_index("image_id") if has_clean else None
        old_eff = int(((~da.blocked) & (da.served != da.original_prediction)).sum())
        for dname, (den, succ, wrongfn) in defs.items():
            m = den.values
            sub = a[m]; dsub = da[m]
            served_wrong = np.array([wrongfn(s, i) for s, i in zip(dsub.served, sub.qid)])
            atk_succ = succ.values[m]
            still = (~dsub.blocked.values) & served_wrong & atk_succ
            induced = (~dsub.blocked.values) & served_wrong & ~atk_succ
            rescued_blk = dsub.blocked.values & atk_succ
            rec = dict(model=model, exp=exp, variant=vname, definition=dname, clean_ref=a.clean_src.iloc[0],
                       N_all=N_all, photos_all=a.photo_id.nunique(), N_den=int(m.sum()), photos_den=sub.photo_id.nunique(),
                       dropped_clean_wrong=int((~m).sum()),
                       attack_success_before=int(atk_succ.sum()), ASR_before_pct=round(atk_succ.mean()*100, 2),
                       still_success_after=int(still.sum()), ASR_after_pct=round(still.mean()*100, 2),
                       defense_induced_errors=int(induced.sum()), defense_induced_pct=round(induced.mean()*100, 2),
                       attack_success_blocked=int(rescued_blk.sum()),
                       old_eff_success_report_style=old_eff, old_eff_ASR_pct=round(old_eff / N_all * 100, 2))
            if dname == "soft>0 (主)":
                lo, hi = photo_boot(sub.photo_id.values, still); rec["ASR_after_CI95"] = f"[{lo:.1f}, {hi:.1f}]"
                lo, hi = photo_boot(sub.photo_id.values, atk_succ); rec["ASR_before_CI95"] = f"[{lo:.1f}, {hi:.1f}]"
            # clean side cost
            if has_clean:
                c = dc.loc[sub.index]
                c_wrong = np.array([wrongfn(s, i) for s, i in zip(c.served, sub.qid)])
                rec.update(clean_N=len(c), clean_blocked=int(c.blocked.sum()),
                           clean_served_wrong=int(((~c.blocked) & c_wrong).sum()),
                           clean_cost_pct=round((c.blocked.values | c_wrong).mean() * 100, 2))
                # all 1000 clean rows (incl. originally-wrong) for reference
                call = d[d.label == 0]
                rec.update(clean_all_blocked=int(call.blocked.sum()))
            abl.append(rec)
    # headline row (A0) with old defs
    m = (a.clean_soft > 0).values
    rows.append(dict(model=model, exp=exp, N_all=N_all, photos_all=a.photo_id.nunique(),
        old_log=int(old_log.sum()), old_png=int(old_png.sum()), old_orig=int(old_orig.sum()),
        clean_ref=a.clean_src.iloc[0], N_den=int(m.sum()), photos_den=a[m].photo_id.nunique(),
        dropped_clean_wrong=int((~m).sum()), new_success=int((a.atk_soft[m] == 0).sum())))
    # flip analysis vs old PNG definition (A0)
    t = a.assign(old_png=old_png, new=((a.clean_soft > 0) & (a.atk_soft == 0)))
    for qid, r in t.iterrows():
        cat = None
        if r.old_png and not r.new:
            if r.clean_soft == 0: cat = "乾淨答案本來就錯→移出分母"
            elif norm(r.pred_ref) == norm(r.clean_ans): cat = "正規化後相同(如 two/2、大小寫、冠詞)"
            else: cat = "攻擊後答案不同但仍被人工接受(同義/部分正確)"
        elif (not r.old_png) and r.new: cat = "舊定義不算成功、新定義算成功"
        elif (not r.old_png) and r.clean_soft == 0: cat = "乾淨答案本來就錯(舊也不算成功)→移出分母"
        if cat: flips_all.append(dict(model=model, exp=exp, question_id=qid, photo_id=r.photo_id, category=cat,
                 question=r.get("question", ""), clean_answer=r.clean_ans, attacked_answer=r.pred_ref,
                 clean_soft=round(r.clean_soft, 3), atk_soft=round(r.atk_soft, 3),
                 human_answers="|".join(qa[int(qid)][0]), top_human=qa[int(qid)][1]))
        per_q.append(dict(model=model, exp=exp, question_id=qid, photo_id=r.photo_id, split=qa[int(qid)][2],
            clean_answer=r.clean_ans, attacked_answer=r.pred_ref, clean_soft=round(r.clean_soft,3), atk_soft=round(r.atk_soft,3),
            clean_strict=r.clean_strict, atk_strict=r.atk_strict, old_log=bool(old_log[qid]), old_png=bool(old_png[qid]),
            old_orig=bool(old_orig[qid]), new_success=bool(t.new[qid]), top_human=qa[int(qid)][1]))

pd.DataFrame(rows).to_csv(OUT/"headline_A0.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(abl).to_csv(OUT/"ablation_all_definitions.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(flips_all).to_csv(OUT/"flips_vs_old_png.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(per_q).to_csv(OUT/"per_question.csv", index=False, encoding="utf-8-sig")
json.dump({"missing_qids": missing, "report_numbers": {f"{k[0]}/{k[1]}": v for k, v in REPORT.items()}}, open(OUT/"meta.json", "w"), ensure_ascii=False, indent=1)
print(pd.DataFrame(rows).to_string()); print(missing)
