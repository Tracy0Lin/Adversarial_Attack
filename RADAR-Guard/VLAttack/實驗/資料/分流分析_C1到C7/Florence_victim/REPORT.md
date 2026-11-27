# Florence-2-large-ft 受害者（florence_victim，VQA=Florence）

## 定義

- **攻擊成功（原本）**：`fooled_before`（對抗圖 VQA ≠ 乾淨正確答案）
- **二階後仍攻擊成功**：未封鎖且 `served_full` ≠ `original_prediction`
- **二階後防禦成功**（攻擊行）：已 fooled 但未再成功（含封鎖或答回正確）
- **乾淨圖變攻擊成功**：`label=0` 且未封鎖但 `served_full` ≠ `original_prediction`
- 風險箱：`risk≤0.3` 低、`≥0.7` 高，其間為中（與 `analyze.py` 一致）

## exp1

### C1 — 攻擊成功→低風險→二階後仍攻擊成功

- 行數 **315**，照片 **117**，佔 fooled **31.5%**
- 平均特徵：d_center=0.01488, rewrite=0.146, risk=0.269

| qid | photo | bin | pred_ref | served | 圖片 |
|-----|-------|-----|----------|--------|------|
| 133001 | 133 | low | abandoned | abandoned | `Florence_victim/exp1/C1/133001/133001_1_ori.png` |
| 139001 | 139 | low | arnett | arnett | `Florence_victim/exp1/C1/139001/139001_1_ori.png` |
| 164003 | 164 | low | etzetztied | etzetztied | `Florence_victim/exp1/C1/164003/164003_1_ori.png` |
| 196000 | 196 | low | tape | tape | `Florence_victim/exp1/C1/196000/196000_1_ori.png` |
| 257000 | 257 | low | shelly | shelly | `Florence_victim/exp1/C1/257000/257000_1_ori.png` |
| 283001 | 283 | low | captain ack | captain ack | `Florence_victim/exp1/C1/283001/283001_1_ori.png` |
| 285002 | 285 | low | no | no | `Florence_victim/exp1/C1/285002/285002_1_ori.png` |
| 294010 | 294 | low | dollhouse character | dollhouse character | `Florence_victim/exp1/C1/294010/294010_1_ori.png` |
| 360001 | 360 | low | skateboarding | skateboarding | `Florence_victim/exp1/C1/360001/360001_1_ori.png` |
| 395002 | 395 | low | silhouette | silhouette | `Florence_victim/exp1/C1/395002/395002_1_ori.png` |

### C2 — 攻擊成功→中風險→二階後仍攻擊成功

- 行數 **13**，照片 **11**，佔 fooled **1.3%**
- 平均特徵：d_center=0.0407, rewrite=0.662, risk=0.551

| qid | photo | bin | pred_ref | served | 圖片 |
|-----|-------|-----|----------|--------|------|
| 502001 | 502 | mid | yes | yes | `Florence_victim/exp1/C2/502001/502001_1_ori.png` |
| 536009 | 536 | mid | answering does not require reading text in the image | answering does not require reading text in the image | `Florence_victim/exp1/C2/536009/536009_1_ori.png` |
| 262284002 | 262284 | mid | yes | yes | `Florence_victim/exp1/C2/262284002/262284002_1_ori.png` |
| 262386003 | 262386 | mid | unanswerable | answering does not require reading text in the image | `Florence_victim/exp1/C2/262386003/262386003_1_ori.png` |
| 262605001 | 262605 | mid | unanswerable | unanswerable | `Florence_victim/exp1/C2/262605001/262605001_1_ori.png` |
| 393282004 | 393282 | mid | no | no | `Florence_victim/exp1/C2/393282004/393282004_1_ori.png` |
| 393288001 | 393288 | mid | red | red | `Florence_victim/exp1/C2/393288001/393288001_1_ori.png` |
| 393372019 | 393372 | mid | yes | yes | `Florence_victim/exp1/C2/393372019/393372019_1_ori.png` |
| 393513002 | 393513 | mid | unanswerable | unanswerable | `Florence_victim/exp1/C2/393513002/393513002_1_ori.png` |
| 524536005 | 524536 | mid | graduation | graduation | `Florence_victim/exp1/C2/524536005/524536005_1_ori.png` |

### C3 — 攻擊成功→低風險→二階後防禦成功

*本實驗無此類樣本。*

### C4 — 攻擊成功→中風險→二階後防禦成功

- 行數 **401**，照片 **148**，佔 fooled **40.1%**
- 平均特徵：d_center=0.05195, rewrite=0.624, risk=0.531

| qid | photo | bin | pred_ref | served | 圖片 |
|-----|-------|-----|----------|--------|------|
| 42001 | 42 | mid | unanswerable | unanswerable | `Florence_victim/exp1/C4/42001/42001_1_ori.png` |
| 73001 | 73 | mid | bike | bike | `Florence_victim/exp1/C4/73001/73001_1_ori.png` |
| 74001 | 74 | mid | 1 | 1 | `Florence_victim/exp1/C4/74001/74001_1_ori.png` |
| 133000 | 133 | mid | ivory | ivory | `Florence_victim/exp1/C4/133000/133000_1_ori.png` |
| 136002 | 136 | mid | ops | ops | `Florence_victim/exp1/C4/136002/136002_1_ori.png` |
| 139003 | 139 | mid | vary | vary | `Florence_victim/exp1/C4/139003/139003_1_ori.png` |
| 143001 | 143 | mid | three | three | `Florence_victim/exp1/C4/143001/143001_1_ori.png` |
| 164001 | 164 | mid | arkson | arkson | `Florence_victim/exp1/C4/164001/164001_1_ori.png` |
| 192001 | 192 | mid | unanswerable | unanswerable | `Florence_victim/exp1/C4/192001/192001_1_ori.png` |
| 196004 | 196 | mid | no | no | `Florence_victim/exp1/C4/196004/196004_1_ori.png` |

### C5 — 攻擊成功→高風險（一階）

- 行數 **271**，照片 **112**，佔 fooled **27.1%**
- 平均特徵：d_center=0.04725, rewrite=0.706, risk=0.728

| qid | photo | bin | pred_ref | served | 圖片 |
|-----|-------|-----|----------|--------|------|
| 42000 | 42 | high | ans | ans | `Florence_victim/exp1/C5/42000/42000_1_ori.png` |
| 133002 | 133 | high | collar | collar | `Florence_victim/exp1/C5/133002/133002_1_ori.png` |
| 136000 | 136 | high | artistic image of a pair of giraffes on display in a museum | artistic image of a pair of giraffes on display in a museum | `Florence_victim/exp1/C5/136000/136000_1_ori.png` |
| 139002 | 139 | high | bright red and white | bright red and white | `Florence_victim/exp1/C5/139002/139002_1_ori.png` |
| 143000 | 143 | high | dark gray | dark gray | `Florence_victim/exp1/C5/143000/143000_1_ori.png` |
| 192000 | 192 | high | unanswerable | unanswerable | `Florence_victim/exp1/C5/192000/192000_1_ori.png` |
| 196001 | 196 | high | no | no | `Florence_victim/exp1/C5/196001/196001_1_ori.png` |
| 241003 | 241 | high | no | no | `Florence_victim/exp1/C5/241003/241003_1_ori.png` |
| 285000 | 285 | high | no | no | `Florence_victim/exp1/C5/285000/285000_1_ori.png` |
| 294002 | 294 | high | laughing and having fun.thus the answer is laughing | laughing and having fun.thus the answer is laughing | `Florence_victim/exp1/C5/294002/294002_1_ori.png` |

### C6 — 乾淨圖→中風險→二階後變攻擊成功

- 行數 **21**，照片 **18**，佔 clean **2.1%**
- 平均特徵：d_center=0.06038, rewrite=0.719, risk=0.479

| qid | photo | bin | pred_ref | served | 圖片 |
|-----|-------|-----|----------|--------|------|
| 192000 | 192 | mid | baseball | answering does not require reading text in the image | `Florence_victim/exp1/C6/192000/192000_1_ori.png` |
| 257004 | 257 | mid | nike | unanswerable | `Florence_victim/exp1/C6/257004/257004_1_ori.png` |
| 536009 | 536 | mid | no | yes | `Florence_victim/exp1/C6/536009/536009_1_ori.png` |
| 131276000 | 131276 | mid | yes | answering does not require reading text in the image | `Florence_victim/exp1/C6/131276000/131276000_1_ori.png` |
| 131390000 | 131390 | mid | yes | answering does not require reading text in the image | `Florence_victim/exp1/C6/131390000/131390000_1_ori.png` |
| 131611007 | 131611 | mid | no | answering does not require reading text in the image | `Florence_victim/exp1/C6/131611007/131611007_1_ori.png` |
| 262175002 | 262175 | mid | hat | pineapple hat | `Florence_victim/exp1/C6/262175002/262175002_1_ori.png` |
| 262235004 | 262235 | mid | woman | answering does not require reading text in the image | `Florence_victim/exp1/C6/262235004/262235004_1_ori.png` |
| 262460000 | 262460 | mid | shirts | unanswerable | `Florence_victim/exp1/C6/262460000/262460000_1_ori.png` |
| 262608004 | 262608 | mid | no | yes | `Florence_victim/exp1/C6/262608004/262608004_1_ori.png` |

### C7 — 乾淨圖→高風險→二階後變攻擊成功

*本實驗無此類樣本。*

## exp2

### C1 — 攻擊成功→低風險→二階後仍攻擊成功

- 行數 **315**，照片 **117**，佔 fooled **31.5%**
- 平均特徵：d_center=0.01488, rewrite=0.146, risk=0.269

| qid | photo | bin | pred_ref | served | 圖片 |
|-----|-------|-----|----------|--------|------|
| 133001 | 133 | low | abandoned | abandoned | `Florence_victim/exp2/C1/133001/133001_1_ori.png` |
| 139001 | 139 | low | arnett | arnett | `Florence_victim/exp2/C1/139001/139001_1_ori.png` |
| 164003 | 164 | low | etzetztied | etzetztied | `Florence_victim/exp2/C1/164003/164003_1_ori.png` |
| 196000 | 196 | low | tape | tape | `Florence_victim/exp2/C1/196000/196000_1_ori.png` |
| 257000 | 257 | low | shelly | shelly | `Florence_victim/exp2/C1/257000/257000_1_ori.png` |
| 283001 | 283 | low | captain ack | captain ack | `Florence_victim/exp2/C1/283001/283001_1_ori.png` |
| 285002 | 285 | low | no | no | `Florence_victim/exp2/C1/285002/285002_1_ori.png` |
| 294010 | 294 | low | dollhouse character | dollhouse character | `Florence_victim/exp2/C1/294010/294010_1_ori.png` |
| 360001 | 360 | low | skateboarding | skateboarding | `Florence_victim/exp2/C1/360001/360001_1_ori.png` |
| 395002 | 395 | low | silhouette | silhouette | `Florence_victim/exp2/C1/395002/395002_1_ori.png` |

### C2 — 攻擊成功→中風險→二階後仍攻擊成功

- 行數 **13**，照片 **11**，佔 fooled **1.3%**
- 平均特徵：d_center=0.0407, rewrite=0.662, risk=0.551

| qid | photo | bin | pred_ref | served | 圖片 |
|-----|-------|-----|----------|--------|------|
| 502001 | 502 | mid | yes | yes | `Florence_victim/exp2/C2/502001/502001_1_ori.png` |
| 536009 | 536 | mid | answering does not require reading text in the image | answering does not require reading text in the image | `Florence_victim/exp2/C2/536009/536009_1_ori.png` |
| 262284002 | 262284 | mid | yes | yes | `Florence_victim/exp2/C2/262284002/262284002_1_ori.png` |
| 262386003 | 262386 | mid | unanswerable | answering does not require reading text in the image | `Florence_victim/exp2/C2/262386003/262386003_1_ori.png` |
| 262605001 | 262605 | mid | unanswerable | unanswerable | `Florence_victim/exp2/C2/262605001/262605001_1_ori.png` |
| 393282004 | 393282 | mid | no | no | `Florence_victim/exp2/C2/393282004/393282004_1_ori.png` |
| 393288001 | 393288 | mid | red | red | `Florence_victim/exp2/C2/393288001/393288001_1_ori.png` |
| 393372019 | 393372 | mid | yes | yes | `Florence_victim/exp2/C2/393372019/393372019_1_ori.png` |
| 393513002 | 393513 | mid | unanswerable | unanswerable | `Florence_victim/exp2/C2/393513002/393513002_1_ori.png` |
| 524536005 | 524536 | mid | graduation | graduation | `Florence_victim/exp2/C2/524536005/524536005_1_ori.png` |

### C3 — 攻擊成功→低風險→二階後防禦成功

*本實驗無此類樣本。*

### C4 — 攻擊成功→中風險→二階後防禦成功

- 行數 **401**，照片 **148**，佔 fooled **40.1%**
- 平均特徵：d_center=0.05195, rewrite=0.624, risk=0.531

| qid | photo | bin | pred_ref | served | 圖片 |
|-----|-------|-----|----------|--------|------|
| 42001 | 42 | mid | unanswerable | unanswerable | `Florence_victim/exp2/C4/42001/42001_1_ori.png` |
| 73001 | 73 | mid | bike | bike | `Florence_victim/exp2/C4/73001/73001_1_ori.png` |
| 74001 | 74 | mid | 1 | 1 | `Florence_victim/exp2/C4/74001/74001_1_ori.png` |
| 133000 | 133 | mid | ivory | ivory | `Florence_victim/exp2/C4/133000/133000_1_ori.png` |
| 136002 | 136 | mid | ops | ops | `Florence_victim/exp2/C4/136002/136002_1_ori.png` |
| 139003 | 139 | mid | vary | vary | `Florence_victim/exp2/C4/139003/139003_1_ori.png` |
| 143001 | 143 | mid | three | three | `Florence_victim/exp2/C4/143001/143001_1_ori.png` |
| 164001 | 164 | mid | arkson | arkson | `Florence_victim/exp2/C4/164001/164001_1_ori.png` |
| 192001 | 192 | mid | unanswerable | unanswerable | `Florence_victim/exp2/C4/192001/192001_1_ori.png` |
| 196004 | 196 | mid | no | no | `Florence_victim/exp2/C4/196004/196004_1_ori.png` |

### C5 — 攻擊成功→高風險（一階）

- 行數 **271**，照片 **112**，佔 fooled **27.1%**
- 平均特徵：d_center=0.04725, rewrite=0.706, risk=0.728

| qid | photo | bin | pred_ref | served | 圖片 |
|-----|-------|-----|----------|--------|------|
| 42000 | 42 | high | ans | ans | `Florence_victim/exp2/C5/42000/42000_1_ori.png` |
| 133002 | 133 | high | collar | collar | `Florence_victim/exp2/C5/133002/133002_1_ori.png` |
| 136000 | 136 | high | artistic image of a pair of giraffes on display in a museum | artistic image of a pair of giraffes on display in a museum | `Florence_victim/exp2/C5/136000/136000_1_ori.png` |
| 139002 | 139 | high | bright red and white | bright red and white | `Florence_victim/exp2/C5/139002/139002_1_ori.png` |
| 143000 | 143 | high | dark gray | dark gray | `Florence_victim/exp2/C5/143000/143000_1_ori.png` |
| 192000 | 192 | high | unanswerable | unanswerable | `Florence_victim/exp2/C5/192000/192000_1_ori.png` |
| 196001 | 196 | high | no | no | `Florence_victim/exp2/C5/196001/196001_1_ori.png` |
| 241003 | 241 | high | no | no | `Florence_victim/exp2/C5/241003/241003_1_ori.png` |
| 285000 | 285 | high | no | no | `Florence_victim/exp2/C5/285000/285000_1_ori.png` |
| 294002 | 294 | high | laughing and having fun.thus the answer is laughing | laughing and having fun.thus the answer is laughing | `Florence_victim/exp2/C5/294002/294002_1_ori.png` |

### C6 — 乾淨圖→中風險→二階後變攻擊成功

- 行數 **21**，照片 **18**，佔 clean **2.1%**
- 平均特徵：d_center=0.06038, rewrite=0.719, risk=0.479

| qid | photo | bin | pred_ref | served | 圖片 |
|-----|-------|-----|----------|--------|------|
| 192000 | 192 | mid | baseball | answering does not require reading text in the image | `Florence_victim/exp2/C6/192000/192000_1_ori.png` |
| 257004 | 257 | mid | nike | unanswerable | `Florence_victim/exp2/C6/257004/257004_1_ori.png` |
| 536009 | 536 | mid | no | yes | `Florence_victim/exp2/C6/536009/536009_1_ori.png` |
| 131276000 | 131276 | mid | yes | answering does not require reading text in the image | `Florence_victim/exp2/C6/131276000/131276000_1_ori.png` |
| 131390000 | 131390 | mid | yes | answering does not require reading text in the image | `Florence_victim/exp2/C6/131390000/131390000_1_ori.png` |
| 131611007 | 131611 | mid | no | answering does not require reading text in the image | `Florence_victim/exp2/C6/131611007/131611007_1_ori.png` |
| 262175002 | 262175 | mid | hat | pineapple hat | `Florence_victim/exp2/C6/262175002/262175002_1_ori.png` |
| 262235004 | 262235 | mid | woman | answering does not require reading text in the image | `Florence_victim/exp2/C6/262235004/262235004_1_ori.png` |
| 262460000 | 262460 | mid | shirts | unanswerable | `Florence_victim/exp2/C6/262460000/262460000_1_ori.png` |
| 262608004 | 262608 | mid | no | yes | `Florence_victim/exp2/C6/262608004/262608004_1_ori.png` |

### C7 — 乾淨圖→高風險→二階後變攻擊成功

*本實驗無此類樣本。*

## exp3

### C1 — 攻擊成功→低風險→二階後仍攻擊成功

- 行數 **541**，照片 **185**，佔 fooled **54.1%**
- 平均特徵：d_center=0.0095, rewrite=0.096, risk=0.27

| qid | photo | bin | pred_ref | served | 圖片 |
|-----|-------|-----|----------|--------|------|
| 569003 | 569 | low | a picture of a woman with a kite in her hand | a picture of a woman with a kite in her hand | `Florence_victim/exp3/C1/569003/569003_1_ori.png` |
| 590000 | 590 | low | unanswerable | unanswerable | `Florence_victim/exp3/C1/590000/590000_1_ori.png` |
| 623000 | 623 | low | arkshire | arkshire | `Florence_victim/exp3/C1/623000/623000_1_ori.png` |
| 626000 | 626 | low | no | no | `Florence_victim/exp3/C1/626000/626000_1_ori.png` |
| 632001 | 632 | low | sunroom | sunroom | `Florence_victim/exp3/C1/632001/632001_1_ori.png` |
| 675000 | 675 | low | touching | touching | `Florence_victim/exp3/C1/675000/675000_1_ori.png` |
| 692000 | 692 | low | dribbling | dribbling | `Florence_victim/exp3/C1/692000/692000_1_ori.png` |
| 699002 | 699 | low | off white | off white | `Florence_victim/exp3/C1/699002/699002_1_ori.png` |
| 711002 | 711 | low | pie | pie | `Florence_victim/exp3/C1/711002/711002_1_ori.png` |
| 724000 | 724 | low | unanswerable | unanswerable | `Florence_victim/exp3/C1/724000/724000_1_ori.png` |

### C2 — 攻擊成功→中風險→二階後仍攻擊成功

- 行數 **10**，照片 **8**，佔 fooled **1.0%**
- 平均特徵：d_center=0.07363, rewrite=0.71, risk=0.55

| qid | photo | bin | pred_ref | served | 圖片 |
|-----|-------|-----|----------|--------|------|
| 661003 | 661 | mid | yes | yes | `Florence_victim/exp3/C2/661003/661003_1_ori.png` |
| 775001 | 775 | mid | answering does not require reading text in the image | people | `Florence_victim/exp3/C2/775001/775001_1_ori.png` |
| 1063003 | 1063 | mid | day time | day time | `Florence_victim/exp3/C2/1063003/1063003_1_ori.png` |
| 1089002 | 1089 | mid | unanswerable | scofield | `Florence_victim/exp3/C2/1089002/1089002_1_ori.png` |
| 1146009 | 1146 | mid | 2 | 2 | `Florence_victim/exp3/C2/1146009/1146009_1_ori.png` |
| 87489002 | 87489 | mid | 626 | 626 | `Florence_victim/exp3/C2/87489002/87489002_1_ori.png` |
| 132024001 | 132024 | mid | yes | yes | `Florence_victim/exp3/C2/132024001/132024001_1_ori.png` |
| 262895002 | 262895 | mid | unanswerable | unanswerable | `Florence_victim/exp3/C2/262895002/262895002_1_ori.png` |

### C3 — 攻擊成功→低風險→二階後防禦成功

*本實驗無此類樣本。*

### C4 — 攻擊成功→中風險→二階後防禦成功

- 行數 **273**，照片 **131**，佔 fooled **27.3%**
- 平均特徵：d_center=0.06712, rewrite=0.788, risk=0.527

| qid | photo | bin | pred_ref | served | 圖片 |
|-----|-------|-----|----------|--------|------|
| 564000 | 564 | mid | no | no | `Florence_victim/exp3/C4/564000/564000_1_ori.png` |
| 569007 | 569 | mid | build a garden | build a garden | `Florence_victim/exp3/C4/569007/569007_1_ori.png` |
| 590001 | 590 | mid | brace bracelets on the face and arm. they are in black and white | brace bracelets on the face and arm. they are in black and white | `Florence_victim/exp3/C4/590001/590001_1_ori.png` |
| 636001 | 636 | mid | a picture of the inside of the toilet bowl is shown.thus the answer is empty | a picture of the inside of the toilet bowl is shown.thus the answer is empty | `Florence_victim/exp3/C4/636001/636001_1_ori.png` |
| 641000 | 641 | mid | four | four | `Florence_victim/exp3/C4/641000/641000_1_ori.png` |
| 661002 | 661 | mid | curbside view of a person holding a sandwich in their left arm | curbside view of a person holding a sandwich in their left arm | `Florence_victim/exp3/C4/661002/661002_1_ori.png` |
| 675001 | 675 | mid | mindshead | mindshead | `Florence_victim/exp3/C4/675001/675001_1_ori.png` |
| 699001 | 699 | mid | stuff on the image | stuff on the image | `Florence_victim/exp3/C4/699001/699001_1_ori.png` |
| 715002 | 715 | mid | 54 | 54 | `Florence_victim/exp3/C4/715002/715002_1_ori.png` |
| 757000 | 757 | mid | amusamusamus is the name of the set of elephants in the picture | amusamusamus is the name of the set of elephants in the picture | `Florence_victim/exp3/C4/757000/757000_1_ori.png` |

### C5 — 攻擊成功→高風險（一階）

- 行數 **176**，照片 **107**，佔 fooled **17.6%**
- 平均特徵：d_center=0.0627, rewrite=0.877, risk=0.723

| qid | photo | bin | pred_ref | served | 圖片 |
|-----|-------|-----|----------|--------|------|
| 564002 | 564 | high | red | red | `Florence_victim/exp3/C5/564002/564002_1_ori.png` |
| 569000 | 569 | high | no | no | `Florence_victim/exp3/C5/569000/569000_1_ori.png` |
| 632002 | 632 | high | yes | yes | `Florence_victim/exp3/C5/632002/632002_1_ori.png` |
| 693001 | 693 | high | aurel | aurel | `Florence_victim/exp3/C5/693001/693001_1_ori.png` |
| 699000 | 699 | high | a picture of a young girl | a picture of a young girl | `Florence_victim/exp3/C5/699000/699000_1_ori.png` |
| 724001 | 724 | high | no | no | `Florence_victim/exp3/C5/724001/724001_1_ori.png` |
| 730001 | 730 | high | yes | yes | `Florence_victim/exp3/C5/730001/730001_1_ori.png` |
| 757004 | 757 | high | (australian) | (australian) | `Florence_victim/exp3/C5/757004/757004_1_ori.png` |
| 761000 | 761 | high | medicine belt | medicine belt | `Florence_victim/exp3/C5/761000/761000_1_ori.png` |
| 772002 | 772 | high | no | no | `Florence_victim/exp3/C5/772002/772002_1_ori.png` |

### C6 — 乾淨圖→中風險→二階後變攻擊成功

*本實驗無此類樣本。*

### C7 — 乾淨圖→高風險→二階後變攻擊成功

*本實驗無此類樣本。*
