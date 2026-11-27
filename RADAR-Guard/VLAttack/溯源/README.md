# 溯源

| 檔案 | 內容 |
|---|---|
| `新舊路徑對照.tsv` | 2026-11-27 整理資料夾時，每個舊路徑搬到哪個新路徑。`文件/成果報告/` 等歷史文件裡的路徑都是舊名，請用這張表對照 |
| `SCRIPT_ORIGIN_MAP.tsv` | repo 內每支腳本與原電腦檔案的對照：SHA-256（LF 換行版）、狀態（逐位元相同、只差換行、內容不同、找不到原檔、還原版）、原始路徑 |
| `SHA256SUMS_scripts.txt` | 同一批檔案的 SHA-256 清單，可用 `sha256sum -c` 核對（需先把換行轉成 LF） |

整理資料夾後，這兩份清單的路徑已改成新路徑。其中 3 個檔案因為內容改了路徑設定，雜湊已重新計算：

- `實驗/程式/2_訓練/train_stage1_svm.py`
- `實驗/程式/4_報告產生/regen_c1c7_by_difficulty.py`
- `系統/1_系統完整程式/vlattack_engine.py`

2026-11-28 為了讓腳本能在別的電腦重跑，另有 10 支腳本把寫死的原電腦絕對路徑改成「先讀環境變數，沒設才用原路徑」，每處只改 1 行並加上 `import os`。雜湊已重新計算；其中原本與原檔相同的 7 支，狀態改為 `MODIFIED_ENV_PATH`：

| 腳本 | 環境變數 |
|---|---|
| `實驗/程式/Florence/攻擊/florence_vlattack/attack_vlattack.py` | `VLATTACK_IMG_ROOT`、`VLATTACK_BLIP_PRETRAIN` |
| `實驗/程式/Florence/攻擊/florence_vlattack/attack_feature.py`、`check_step1.py`、`florence_victim/replace_exp3_one.py`、`實驗/程式/5_題目難度/build_florence_samples.py` | `VLATTACK_IMG_ROOT` |
| `實驗/程式/1_攻擊與資料建立/create_answer_list.py` | `VLATTACK_ANN_ROOT` |
| `實驗/程式/1_攻擊與資料建立/analyze_attack_sweet_spot.py` | `VLATTACK_VQA_OUT` |
| `實驗/程式/3_評估/run_rotation_cos_vs_ang.py` | `VLATTACK_MINILM_DIR` |
| `實驗/程式/3_評估/compute_rpf1.py` | `VLATTACK_OOF_SCORES` |
| `實驗/程式/4_報告產生/實驗二_cos_vs_角距_圖表/build_exp2_excel_and_figures.py` | `VLATTACK_COSANG_ARCHIVE` |

不設環境變數時，行為與原檔完全相同。其餘檔案的內容和雜湊都沒有變。`實驗設計_VLAttack/2_訓練腳本/` 裡 4 個與 `3_評估腳本/` 完全相同的副本已刪除，清單中也移除了它們的列。
