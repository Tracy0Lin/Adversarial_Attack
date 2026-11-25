# BLIP 受害者（multi_captioner，VQA=BLIP）

## 定義

- **攻擊成功（原本）**：`fooled_before`（對抗圖 VQA ≠ 乾淨正確答案）
- **二階後仍攻擊成功**：未封鎖且 `served_full` ≠ `original_prediction`
- **二階後防禦成功**（攻擊行）：已 fooled 但未再成功（含封鎖或答回正確）
- **乾淨圖變攻擊成功**：`label=0` 且未封鎖但 `served_full` ≠ `original_prediction`
- 風險箱：`risk≤0.3` 低、`≥0.7` 高，其間為中（與 `analyze.py` 一致）

## exp1

### C1 — 攻擊成功→低風險→二階後仍攻擊成功

- 行數 **137**，照片 **72**，佔 fooled **29.72%**
- 平均特徵：d_center=0.01613, rewrite=0.201, risk=0.27

| qid | photo | bin | pred_ref | served | 圖片 |
|-----|-------|-----|----------|--------|------|
| 74000 | 74 | low | yes | yes | `BLIP_victim/exp1/C1/74000/74000_1_ori.png` |
| 143000 | 143 | low | black | black | `BLIP_victim/exp1/C1/143000/143000_1_ori.png` |
| 164000 | 164 | low | 2 | 2 | `BLIP_victim/exp1/C1/164000/164000_1_ori.png` |
| 196000 | 196 | low | rice | rice | `BLIP_victim/exp1/C1/196000/196000_1_ori.png` |
| 283002 | 283 | low | stone | stone | `BLIP_victim/exp1/C1/283002/283002_1_ori.png` |
| 294010 | 294 | low | no | no | `BLIP_victim/exp1/C1/294010/294010_1_ori.png` |
| 357001 | 357 | low | no | no | `BLIP_victim/exp1/C1/357001/357001_1_ori.png` |
| 359001 | 359 | low | no | no | `BLIP_victim/exp1/C1/359001/359001_1_ori.png` |
| 415002 | 415 | low | no | no | `BLIP_victim/exp1/C1/415002/415002_1_ori.png` |
| 472001 | 472 | low | yes | yes | `BLIP_victim/exp1/C1/472001/472001_1_ori.png` |

### C2 — 攻擊成功→中風險→二階後仍攻擊成功

- 行數 **25**，照片 **21**，佔 fooled **5.42%**
- 平均特徵：d_center=0.06686, rewrite=0.912, risk=0.515

| qid | photo | bin | pred_ref | served | 圖片 |
|-----|-------|-----|----------|--------|------|
| 131171002 | 131171 | mid | no | no | `BLIP_victim/exp1/C2/131171002/131171002_1_ori.png` |
| 131390001 | 131390 | mid | no | no | `BLIP_victim/exp1/C2/131390001/131390001_1_ori.png` |
| 131611015 | 131611 | mid | yes | yes | `BLIP_victim/exp1/C2/131611015/131611015_1_ori.png` |
| 262162028 | 262162 | mid | twin | twin | `BLIP_victim/exp1/C2/262162028/262162028_1_ori.png` |
| 262227003 | 262227 | mid | green | green | `BLIP_victim/exp1/C2/262227003/262227003_1_ori.png` |
| 262325003 | 262325 | mid | plate | plate | `BLIP_victim/exp1/C2/262325003/262325003_1_ori.png` |
| 262371000 | 262371 | mid | no | no | `BLIP_victim/exp1/C2/262371000/262371000_1_ori.png` |
| 262386014 | 262386 | mid | unknown | unknown | `BLIP_victim/exp1/C2/262386014/262386014_1_ori.png` |
| 262594001 | 262594 | mid | black | black | `BLIP_victim/exp1/C2/262594001/262594001_1_ori.png` |
| 262605008 | 262605 | mid | no | no | `BLIP_victim/exp1/C2/262605008/262605008_1_ori.png` |

### C3 — 攻擊成功→低風險→二階後防禦成功

*本實驗無此類樣本。*

### C4 — 攻擊成功→中風險→二階後防禦成功

- 行數 **135**，照片 **74**，佔 fooled **29.28%**
- 平均特徵：d_center=0.08075, rewrite=0.866, risk=0.52

| qid | photo | bin | pred_ref | served | 圖片 |
|-----|-------|-----|----------|--------|------|
| 133000 | 133 | mid | white | white | `BLIP_victim/exp1/C4/133000/133000_1_ori.png` |
| 136002 | 136 | mid | cat | cat | `BLIP_victim/exp1/C4/136002/136002_1_ori.png` |
| 164002 | 164 | mid | yes | yes | `BLIP_victim/exp1/C4/164002/164002_1_ori.png` |
| 192002 | 192 | mid | dodgers | cubs | `BLIP_victim/exp1/C4/192002/192002_1_ori.png` |
| 241003 | 241 | mid | no | no | `BLIP_victim/exp1/C4/241003/241003_1_ori.png` |
| 395000 | 395 | mid | red | red | `BLIP_victim/exp1/C4/395000/395000_1_ori.png` |
| 397000 | 397 | mid | leaves | leaves | `BLIP_victim/exp1/C4/397000/397000_1_ori.png` |
| 488001 | 488 | mid | badminton | badminton | `BLIP_victim/exp1/C4/488001/488001_1_ori.png` |
| 502003 | 502 | mid | no | no | `BLIP_victim/exp1/C4/502003/502003_1_ori.png` |
| 536000 | 536 | mid | bathroom | bathroom | `BLIP_victim/exp1/C4/536000/536000_1_ori.png` |

### C5 — 攻擊成功→高風險（一階）

- 行數 **164**，照片 **78**，佔 fooled **35.57%**
- 平均特徵：d_center=0.12673, rewrite=0.91, risk=0.765

| qid | photo | bin | pred_ref | served | 圖片 |
|-----|-------|-----|----------|--------|------|
| 42001 | 42 | high | no | no | `BLIP_victim/exp1/C5/42001/42001_1_ori.png` |
| 73001 | 73 | high | bike | bike | `BLIP_victim/exp1/C5/73001/73001_1_ori.png` |
| 133001 | 133 | high | no | no | `BLIP_victim/exp1/C5/133001/133001_1_ori.png` |
| 139003 | 139 | high | red | red | `BLIP_victim/exp1/C5/139003/139003_1_ori.png` |
| 241000 | 241 | high | bench | bench | `BLIP_victim/exp1/C5/241000/241000_1_ori.png` |
| 257003 | 257 | high | no | no | `BLIP_victim/exp1/C5/257003/257003_1_ori.png` |
| 294000 | 294 | high | no | no | `BLIP_victim/exp1/C5/294000/294000_1_ori.png` |
| 397001 | 397 | high | america | america | `BLIP_victim/exp1/C5/397001/397001_1_ori.png` |
| 428000 | 428 | high | scarf | scarf | `BLIP_victim/exp1/C5/428000/428000_1_ori.png` |
| 536005 | 536 | high | 2 | 2 | `BLIP_victim/exp1/C5/536005/536005_1_ori.png` |

### C6 — 乾淨圖→中風險→二階後變攻擊成功

- 行數 **1**，照片 **1**，佔 clean **0.1%**
- 平均特徵：d_center=0.15822, rewrite=0.6, risk=0.641

| qid | photo | bin | pred_ref | served | 圖片 |
|-----|-------|-----|----------|--------|------|
| 240301002 | 240301 | mid | tired | resting | `BLIP_victim/exp1/C6/240301002/240301002_1_ori.png` |

### C7 — 乾淨圖→高風險→二階後變攻擊成功

*本實驗無此類樣本。*

## exp2

### C1 — 攻擊成功→低風險→二階後仍攻擊成功

- 行數 **220**，照片 **126**，佔 fooled **22.04%**
- 平均特徵：d_center=0.00903, rewrite=0.103, risk=0.268

| qid | photo | bin | pred_ref | served | 圖片 |
|-----|-------|-----|----------|--------|------|
| 74000 | 74 | low | yes | yes | `BLIP_victim/exp2/C1/74000/74000_1_ori.png` |
| 143000 | 143 | low | black | black | `BLIP_victim/exp2/C1/143000/143000_1_ori.png` |
| 164000 | 164 | low | 2 | 2 | `BLIP_victim/exp2/C1/164000/164000_1_ori.png` |
| 196000 | 196 | low | rice | rice | `BLIP_victim/exp2/C1/196000/196000_1_ori.png` |
| 283002 | 283 | low | stone | stone | `BLIP_victim/exp2/C1/283002/283002_1_ori.png` |
| 294010 | 294 | low | no | no | `BLIP_victim/exp2/C1/294010/294010_1_ori.png` |
| 357001 | 357 | low | no | no | `BLIP_victim/exp2/C1/357001/357001_1_ori.png` |
| 359001 | 359 | low | no | no | `BLIP_victim/exp2/C1/359001/359001_1_ori.png` |
| 415002 | 415 | low | no | no | `BLIP_victim/exp2/C1/415002/415002_1_ori.png` |
| 520000 | 520 | low | birds | birds | `BLIP_victim/exp2/C1/520000/520000_1_ori.png` |

### C2 — 攻擊成功→中風險→二階後仍攻擊成功

- 行數 **59**，照片 **51**，佔 fooled **5.91%**
- 平均特徵：d_center=0.07269, rewrite=0.873, risk=0.552

| qid | photo | bin | pred_ref | served | 圖片 |
|-----|-------|-----|----------|--------|------|
| 472001 | 472 | mid | yes | yes | `BLIP_victim/exp2/C2/472001/472001_1_ori.png` |
| 569001 | 569 | mid | green | green | `BLIP_victim/exp2/C2/569001/569001_1_ori.png` |
| 623002 | 623 | mid | no | no | `BLIP_victim/exp2/C2/623002/623002_1_ori.png` |
| 636002 | 636 | mid | no | no | `BLIP_victim/exp2/C2/636002/636002_1_ori.png` |
| 699002 | 699 | mid | white | gray | `BLIP_victim/exp2/C2/699002/699002_1_ori.png` |
| 711000 | 711 | mid | floral | floral | `BLIP_victim/exp2/C2/711000/711000_1_ori.png` |
| 730004 | 730 | mid | no | no | `BLIP_victim/exp2/C2/730004/730004_1_ori.png` |
| 785003 | 785 | mid | no | no | `BLIP_victim/exp2/C2/785003/785003_1_ori.png` |
| 885027 | 885 | mid | woman | woman | `BLIP_victim/exp2/C2/885027/885027_1_ori.png` |
| 985001 | 985 | mid | no | no | `BLIP_victim/exp2/C2/985001/985001_1_ori.png` |

### C3 — 攻擊成功→低風險→二階後防禦成功

*本實驗無此類樣本。*

### C4 — 攻擊成功→中風險→二階後防禦成功

- 行數 **279**，照片 **165**，佔 fooled **27.96%**
- 平均特徵：d_center=0.07313, rewrite=0.863, risk=0.549

| qid | photo | bin | pred_ref | served | 圖片 |
|-----|-------|-----|----------|--------|------|
| 136002 | 136 | mid | cat | cat | `BLIP_victim/exp2/C4/136002/136002_1_ori.png` |
| 164002 | 164 | mid | yes | yes | `BLIP_victim/exp2/C4/164002/164002_1_ori.png` |
| 192002 | 192 | mid | dodgers | cubs | `BLIP_victim/exp2/C4/192002/192002_1_ori.png` |
| 241003 | 241 | mid | no | no | `BLIP_victim/exp2/C4/241003/241003_1_ori.png` |
| 294006 | 294 | mid | no | no | `BLIP_victim/exp2/C4/294006/294006_1_ori.png` |
| 397000 | 397 | mid | leaves | leaves | `BLIP_victim/exp2/C4/397000/397000_1_ori.png` |
| 536000 | 536 | mid | bathroom | bathroom | `BLIP_victim/exp2/C4/536000/536000_1_ori.png` |
| 564000 | 564 | mid | no | no | `BLIP_victim/exp2/C4/564000/564000_1_ori.png` |
| 569004 | 569 | mid | no | no | `BLIP_victim/exp2/C4/569004/569004_1_ori.png` |
| 632006 | 632 | mid | no | no | `BLIP_victim/exp2/C4/632006/632006_1_ori.png` |

### C5 — 攻擊成功→高風險（一階）

- 行數 **440**，照片 **208**，佔 fooled **44.09%**
- 平均特徵：d_center=0.12312, rewrite=0.942, risk=0.741

| qid | photo | bin | pred_ref | served | 圖片 |
|-----|-------|-----|----------|--------|------|
| 42001 | 42 | high | no | no | `BLIP_victim/exp2/C5/42001/42001_1_ori.png` |
| 73001 | 73 | high | bike | bike | `BLIP_victim/exp2/C5/73001/73001_1_ori.png` |
| 133000 | 133 | high | white | white | `BLIP_victim/exp2/C5/133000/133000_1_ori.png` |
| 139003 | 139 | high | red | red | `BLIP_victim/exp2/C5/139003/139003_1_ori.png` |
| 241000 | 241 | high | bench | bench | `BLIP_victim/exp2/C5/241000/241000_1_ori.png` |
| 257003 | 257 | high | no | no | `BLIP_victim/exp2/C5/257003/257003_1_ori.png` |
| 294000 | 294 | high | no | no | `BLIP_victim/exp2/C5/294000/294000_1_ori.png` |
| 395000 | 395 | high | red | red | `BLIP_victim/exp2/C5/395000/395000_1_ori.png` |
| 397001 | 397 | high | america | america | `BLIP_victim/exp2/C5/397001/397001_1_ori.png` |
| 428000 | 428 | high | scarf | scarf | `BLIP_victim/exp2/C5/428000/428000_1_ori.png` |

### C6 — 乾淨圖→中風險→二階後變攻擊成功

- 行數 **1**，照片 **1**，佔 clean **0.1%**
- 平均特徵：d_center=0.10351, rewrite=0.6, risk=0.407

| qid | photo | bin | pred_ref | served | 圖片 |
|-----|-------|-----|----------|--------|------|
| 524850002 | 524850 | mid | yes | no | `BLIP_victim/exp2/C6/524850002/524850002_1_ori.png` |

### C7 — 乾淨圖→高風險→二階後變攻擊成功

*本實驗無此類樣本。*

## exp3

### C1 — 攻擊成功→低風險→二階後仍攻擊成功

- 行數 **197**，照片 **133**，佔 fooled **19.78%**
- 平均特徵：d_center=0.00795, rewrite=0.114, risk=0.269

| qid | photo | bin | pred_ref | served | 圖片 |
|-----|-------|-----|----------|--------|------|
| 1369001 | 1369 | low | robe | robe | `—` |
| 1436000 | 1436 | low | 0 | 0 | `—` |
| 1448002 | 1448 | low | yes | yes | `—` |
| 1563001 | 1563 | low | no | no | `—` |
| 1577004 | 1577 | low | red | red | `—` |
| 1584000 | 1584 | low | yes | yes | `—` |
| 1682001 | 1682 | low | no | no | `—` |
| 1757001 | 1757 | low | no | no | `—` |
| 1840000 | 1840 | low | purple | purple | `—` |
| 1856001 | 1856 | low | white | white | `—` |

### C2 — 攻擊成功→中風險→二階後仍攻擊成功

- 行數 **40**，照片 **38**，佔 fooled **4.02%**
- 平均特徵：d_center=0.07229, rewrite=0.843, risk=0.541

| qid | photo | bin | pred_ref | served | 圖片 |
|-----|-------|-----|----------|--------|------|
| 1525010 | 1525 | mid | no | no | `—` |
| 1591001 | 1591 | mid | show | show | `—` |
| 1599002 | 1599 | mid | no | no | `—` |
| 1779002 | 1779 | mid | no | no | `—` |
| 2139002 | 2139 | mid | flag | flag | `—` |
| 132622003 | 132622 | mid | yes | yes | `—` |
| 132652013 | 132652 | mid | nowhere | nowhere | `—` |
| 132682000 | 132682 | mid | no | no | `—` |
| 132706001 | 132706 | mid | no | no | `—` |
| 132814001 | 132814 | mid | zebra | none | `—` |

### C3 — 攻擊成功→低風險→二階後防禦成功

*本實驗無此類樣本。*

### C4 — 攻擊成功→中風險→二階後防禦成功

- 行數 **270**，照片 **189**，佔 fooled **27.11%**
- 平均特徵：d_center=0.07145, rewrite=0.888, risk=0.548

| qid | photo | bin | pred_ref | served | 圖片 |
|-----|-------|-----|----------|--------|------|
| 1398001 | 1398 | mid | gray | gray | `—` |
| 1436003 | 1436 | mid | standing | standing | `—` |
| 1490001 | 1490 | mid | 0 | 0 | `—` |
| 1525000 | 1525 | mid | hospital | hospital | `—` |
| 1563002 | 1563 | mid | green | green | `—` |
| 1590000 | 1590 | mid | vase | vase | `—` |
| 1591003 | 1591 | mid | 0 | 0 | `—` |
| 1592002 | 1592 | mid | pink | pink | `—` |
| 1682010 | 1682 | mid | no | no | `—` |
| 1841000 | 1841 | mid | no | no | `—` |

### C5 — 攻擊成功→高風險（一階）

- 行數 **489**，照片 **249**，佔 fooled **49.1%**
- 平均特徵：d_center=0.12839, rewrite=0.942, risk=0.741

| qid | photo | bin | pred_ref | served | 圖片 |
|-----|-------|-----|----------|--------|------|
| 387001 | 387 | high | apple | apple | `—` |
| 1393002 | 1393 | high | no | no | `—` |
| 1425002 | 1425 | high | no | no | `—` |
| 1436005 | 1436 | high | nothing | nothing | `—` |
| 1464000 | 1464 | high | no | no | `—` |
| 1525001 | 1525 | high | 0 | 0 | `—` |
| 1532003 | 1532 | high | 7 | 7 | `—` |
| 1577003 | 1577 | high | no | no | `—` |
| 1590002 | 1590 | high | cat | cat | `—` |
| 1655000 | 1655 | high | no | no | `—` |

### C6 — 乾淨圖→中風險→二階後變攻擊成功

*本實驗無此類樣本。*

### C7 — 乾淨圖→高風險→二階後變攻擊成功

*本實驗無此類樣本。*
