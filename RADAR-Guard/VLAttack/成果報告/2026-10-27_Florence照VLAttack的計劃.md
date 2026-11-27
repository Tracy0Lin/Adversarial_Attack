# 用 Florence 照 VLAttack 跑結果的計劃

日期：2026-10-27

現在 `成果報告/2026-10-26_Florence三實驗.md` 裡的數字已經做完，但那是 `FlorencePGD`：改照片和回答都用同一份 `microsoft/Florence-2-large-ft`，改的時候直接看它還說不說得出正確答案。實驗一因此是 100%。那份表留著，當作另一種攻擊，不要改寫成 VLAttack。

若要的是和 BLIP 同一種 VLAttack，必須重跑。BLIP 是改照片時看預訓練 BLIP 的印象，改完才問 BLIP-VQA。Florence 要對成兩份官方權重，不能只換模型名字。

## 要跑成什麼

| | 改照片時看誰 | 改完誰回答 |
| --- | --- | --- |
| BLIP 的 VLAttack（已有） | 預訓練 BLIP `model_base_14M.pth` 的中間印象 | 另一份 BLIP-VQA `model_vqa.pth` |
| 這次要做的 Florence | 預訓練 `microsoft/Florence-2-large`（沒有 `-ft`）的中間印象 | 另一份 `microsoft/Florence-2-large-ft` |
| 已經做完、不要再當成 VLAttack | 同一份 `Florence-2-large-ft`，而且看的是正確答案還在不在 | 還是 `Florence-2-large-ft` |

改照片時只把預訓練 Florence 對這張圖的印象推離原圖。不要把問題的正確答案送進損失，也不要在改的過程中問 `-ft`。照片定稿之後，才用 `num_beams=3` 問 `Florence-2-large-ft`。答案不等於 `ground_truth` 才算攻擊成功。

`ground_truth` 沿用現在的篩選：`-ft` 看原圖說出的那句，必須出現在 VQAv2 標註裡，而且是票數最高的那句。篩選問答繼續用 `vqa()`、`num_beams=3`。可沿用已經建好的可攻擊池，不必為了換攻擊方式重篩，除非要擴大池子。

超參和 BLIP 那次對齊：100 steps、alpha 0.002、L∞ = 0.125。這次倉庫的 BLIP 只做了圖像這一段，文字攻擊是關掉的，Florence 也先只做圖像。

Florence 沒有 BLIP 的 `Gen_feats`。要新寫一段，從 `Florence-2-large` 的視覺編碼取出中間特徵，再用同一種「區塊特徵距離」去推像素。這不是把 `attack_florence_pgd.py` 的模型名稱改掉。

輸出寫到新目錄（例如 `BLIP_attack/output/VQA/florence_vlattack/`），不要覆蓋 `florence_victim/` 裡的 PGD 圖和 log。

## 三個實驗

和 BLIP 同一種切法。分母都用 1000 張攻擊圖。

1. **實驗一，自然情境。** 從池子依序打，打完前 1000 題就停。失敗的留下。這一組的初始 ASR 不該事先當成 100%。另外配這 1000 題的乾淨圖。
2. **實驗二，壓力測試。** 從實驗一取出攻擊成功的題，湊滿 1000 張成功，配上對應乾淨圖。若實驗一成功不到 1000，就繼續往池子後面打，直到成功張數夠。
3. **實驗三，純攻擊。** 再取 1000 張攻擊成功，照片不和實驗一、二重疊，題目也不共用同一張照片。存檔之後用 `num_beams=3` 再問一次；若答回 `ground_truth`，這張不算成功，換下一張再進防禦。實驗三的初始 ASR 以這次再問為準，目標是 1000/1000。

每次問原圖、問對抗圖、防禦時再問，都用 `num_beams=3`。log 裡不要再寫死 `num_beams=1`。

## 防禦和報表

防禦沿用現在的兩階段，不必重做方法：

- 第一階段看 caption 偏移，分成低、中、高風險。高風險擋下，低風險原樣放行，中風險進第二階段。
- 第二階段高斯 σ = 0.8。
- 乾淨圖和攻擊圖都跟 `ground_truth` 比。攻擊圖答案不等於 `ground_truth`、而且沒被擋下，算仍成功。乾淨圖被判高風險，或中風險糊完之後答案不等於 `ground_truth`，算良性誤判。

主表用這個規則（程式裡的 `A3_full_mitigate_only`）。`summary.json` 的 `effective_asr` 是另一種算法：中風險只要糊完答案和沒糊的不一樣就擋下。可以附在同一份報告，但要分開標名稱。

報告對照三列，不要混：

| 列 | 初始 ASR 預期 |
| --- | --- |
| BLIP 的 VLAttack（已有） | 實驗一約 46%，不是 100% |
| Florence 照上面重跑 | 實驗一由數據決定，事先不寫成 100% |
| 已完成的 FlorencePGD | 實驗一 100%，實驗三兩階段後 70.5%（`2026-10-26` 那份） |

## 建議順序

在有 GPU 的機器上做。本機沒有 NVIDIA GPU，不能跑這段。

1. 載入兩份權重，確認預訓練那份能取出中間特徵、`-ft` 能用 `num_beams=3` 答 VQA。先用幾張乾淨圖對一下，確認問答和現有 `ground_truth` 一致。
2. 做特徵距離的 PGD，小批（例如 20 題）看初始 ASR 是不是落在「有成功、也有失敗」，而不是整批 100%。若小批全成功，先檢查是不是又把 `-ft` 的答案損失加回去了。
3. 實驗一打滿 1000，寫 log 和 PNG。
4. 依成功與否建實驗二、實驗三。實驗三先做 beams=3 再問，失敗的先換掉。
5. 對三組跑現有的探針、caption、analyze（σ = 0.8）。
6. 另寫一份結果報告。`2026-10-26_Florence三實驗.md` 不要改成這次的數字。

## 可另外做、但不能代替上面

把已經打好的 BLIP VLAttack 圖片直接拿去問 `Florence-2-large-ft`。這只是看 BLIP 的攻擊圖會不會順便騙到 Florence，論文的做法不是這一條。目錄和表格都標成「轉移」，不要標成 Florence 上的 VLAttack。
