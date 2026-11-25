# 資料集與配對說明

每個 ZIP 內的 `DATASET_INFO.json` 列出實際圖片數量；`data/SHA256SUMS.txt` 可驗證 ZIP 完整性。ZIP 內使用相對路徑。分類器的標籤為 Clean=0、MF-tt=1；圖片資料夾中的 `class0` 是既有攻擊腳本的索引目錄，不能單獨解讀成分類標籤。

| ZIP | 內容 | 圖片數 |
| --- | --- | ---: |
| `shared_inputs.zip` | COCO 來源圖、配對的 SD 1.4 目標圖、原始配對清單 | 1,000＋1,000 |
| `blip_dataset.zip` | 自然情境 Clean／MF-ii／MF-tt；壓力情境 Clean／Attack | 每組 1,000，共 5,000 |
| `vitgpt2_dataset.zip` | 自然情境 Clean／MF-ii／MF-tt；壓力情境 Clean／Attack | 每組 1,000，共 5,000 |
| `git_dataset.zip` | 自然情境 Clean／MF-ii／MF-tt；確認成功樣本的 Clean／Attack 及原始選樣清單 | 每組 1,000，共 5,000 |

自然情境使用原始配對樣本；壓力情境使用已成功攻擊的樣本與相應正常圖，兩者不可混成同一份統計。BLIP、ViT-GPT2 的純攻擊情境沿用各自壓力情境的 `stress/attack`；GIT 的純攻擊情境沿用 `stress/attack/class0`，結果清單在 `results/`。ViT-GPT2 的兩組攻擊圖片已逐檔 SHA-256 核對相同；BLIP 也已逐檔核對相同。

原始清單及部分結果 JSON 保留遠端實驗時的 `/workspace/projects/couser/...` 絕對路徑，僅供追溯，移到其他主機後不能直接以其中路徑開檔。讀取圖片時請使用 ZIP 中的相對路徑；重跑實驗要重新指定本機資料根目錄。來源影像與生成目標圖涉及上游資料及模型條款，使用或再散布時應核對對應來源規範。
