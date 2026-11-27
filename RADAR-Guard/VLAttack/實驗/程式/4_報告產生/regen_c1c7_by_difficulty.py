# RECONSTRUCTED SCRIPT (2026-11-27) -- NOT the original generator.
# The one-off script that originally wrote question_difficulty/c1c7_by_difficulty.csv is lost.
# This reconstruction reproduces all 72 rows of that csv exactly (0 mismatches) from the rows_scored.csv files + difficulty labels.
# 還原版腳本：原始產生程式已遺失；本腳本依同一規則重建，72 列結果與原檔完全一致。

"""Regenerate question_difficulty/c1c7_by_difficulty.csv from rows_scored.csv + difficulty labels.
Rule (實驗主表, 2026-11-11 version described in 成果報告/2026-11-09_難度與C1到C7.md):
  high risk -> blocked; low risk -> served = pred_ref; mid risk -> served = pred_0.8.
  victim answer = original_prediction (BLIP) or ground_truth (Florence).
  C1..C5: label==1 & fooled_before; C6/C7: label==0; X: label==1 & ~fooled_before & not high & served != victim.
Difficulty is looked up by question_id == image_id on the clean original image.

Reconstructed 2026-11-27 for verification (the original one-off script that wrote c1c7_by_difficulty.csv was not found on TL).
Reproduces all 72 rows of the original csv exactly.

Usage:  python regen_c1c7_by_difficulty.py <實驗/資料 folder> [out.csv]
Inputs (repo layout, under <實驗/資料>; paths updated 2026-11-27 for the restructured repo):
  BLIP/主流程_multi_captioner/exp{1,2,3}_blip/rows_scored.csv            (BLIP, default theta)
  Florence/florence_victim/exp{1,2,3}/florence2/vqa_florence2/rows_scored.csv
  Florence/florence_vlattack/exp{1,2,3}/florence2/vqa_florence2/rows_scored.csv
  題目難度/labels.csv, 題目難度/florence/labels.csv
"""
import pandas as pd, sys, os
D=sys.argv[1] if len(sys.argv)>1 else '.'
OUT=sys.argv[2] if len(sys.argv)>2 else 'c1c7_by_difficulty_regen.csv'
TFL=D
def rows_path(study,exp):
    n=exp[-1]
    if study=='BLIP': return f'{TFL}/BLIP/主流程_multi_captioner/exp{n}_blip/rows_scored.csv'
    if study=='FlorenceVictim': return f'{TFL}/Florence/florence_victim/exp{n}/florence2/vqa_florence2/rows_scored.csv'
    return f'{TFL}/Florence/florence_vlattack/exp{n}/florence2/vqa_florence2/rows_scored.csv'
lab={'BLIP':pd.read_csv(f'{D}/題目難度/labels.csv',dtype=str),'FlorenceVictim':pd.read_csv(f'{D}/題目難度/florence/labels.csv',dtype=str),'FlorenceVLAttack':pd.read_csv(f'{D}/題目難度/florence/labels.csv',dtype=str)}
rows=[]
for study in ('BLIP','FlorenceVictim','FlorenceVLAttack'):
    L=lab[study].set_index('question_id')['difficulty_3'].to_dict()
    for exp in ('exp1','exp2','exp3'):
        df=pd.read_csv(rows_path(study,exp),dtype={'image_id':str},keep_default_na=False)
        df['label']=df['label'].astype(int)
        for c in ('fooled_before',): df[c]=df[c].astype(str).str.lower().eq('true')
        victim=df['original_prediction'] if study=='BLIP' else df['ground_truth']
        served=df['pred_ref'].where(df['bin']=='low', df['pred_0.8']).where(df['bin']!='high', None)
        diff=served.astype(str)!=victim.astype(str)
        cat=pd.Series('',index=df.index)
        atk=(df.label==1)&df.fooled_before
        cat[atk&(df.bin=='high')]='C5'
        cat[atk&(df.bin=='low')&diff]='C1'; cat[atk&(df.bin=='mid')&diff]='C2'
        cat[atk&(df.bin=='low')&~diff]='C3'; cat[atk&(df.bin=='mid')&~diff]='C4'
        cl=df.label==0
        cat[cl&(df.bin=='mid')&diff]='C6'; cat[cl&(df.bin=='high')&diff&False]='C7'
        cat[(df.label==1)&~df.fooled_before&(df.bin!='high')&diff]='X'
        d3=df['image_id'].map(L).fillna('unlabeled')
        for c in ['C1','C2','C3','C4','C5','C6','C7','X']:
            m=cat==c
            rows.append([study,exp,c,int(m.sum()),int((d3[m]=='easy').sum()),int((d3[m]=='medium').sum()),int((d3[m]=='hard').sum()),int((d3[m]=='unlabeled').sum())])
out=pd.DataFrame(rows,columns=['study','exp','category','n','easy','medium','hard','unlabeled'])
out.to_csv(OUT,index=False)
orig=pd.read_csv(f'{D}/題目難度/c1c7_by_difficulty.csv')
print('orig rows',len(orig),'regen rows',len(out))
mg=orig.merge(out,on=['study','exp','category'],how='outer',suffixes=('_o','_r'))
cols=['n','easy','medium','hard','unlabeled']
bad=mg[(mg[[c+'_o' for c in cols]].fillna(-1).values!=mg[[c+'_r' for c in cols]].fillna(-1).values).any(axis=1)]
print('mismatching rows:',len(bad)); print(bad.to_string())
