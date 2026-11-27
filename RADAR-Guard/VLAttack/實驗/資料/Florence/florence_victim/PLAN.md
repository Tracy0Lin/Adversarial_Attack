# Florence-2-large-ft 受害者全流程（路線 B）

建立日：2026-10-22  
根目錄：`BLIP_attack/output/VQA/florence_victim/`  
程式：`BLIP_attack/output/florence_victim/`

## 設計決策

| 項目 | 選擇 |
|------|------|
| 模型 | **單一 checkpoint**：`microsoft/Florence-2-large-ft` |
| Caption | `<CAPTION>` |
| VQA | **prompt = 問題原文**（官方 ft 討論區：無 `<VQA>` 前綴） |
| 攻擊 | **白盒 = 黑盒 = Florence**：PGD 最大化 seq2seq 對「正確答案」的 teacher-forcing loss，L∞=0.125，steps=100，alpha=0.002（與 BLIP 線 PGD 超參一致；非 BLIP pretrain VLAttack） |
| 建集 | 與 `prepare_vqa.py` 同精神：預測命中標註且為最高權重答案 → 可攻擊池 |
| 記錄 | `screening_log.jsonl` 全題；`attack_dual_defense_log.json` 與 BLIP 線欄位相容 |
| 防禦 | 沿用 `multi_captioner` 的 `run_probe` / `run_captions` / `analyze`，資料根目錄改為本目錄 |

## 實驗

| 實驗 | 目錄 | 內容 |
|------|------|------|
| 一 | `exp1_mixed1000/` | 自然攻擊成功比例 + 兩階段防禦 |
| 二 | `exp2_success1000/` | 僅 Florence 攻擊成功 + 乾淨對 |
| 三 | `exp3_success1000/` | 僅攻擊成功、照片與一二不重疊 |

## 執行順序（建議）

```bash
cd BLIP_attack/output/florence_victim
# 1) 掃描 VQAv2 val、寫入可攻擊池與 screening log
python prepare_eligible.py --max_pool 5000

# 2) 實驗一：攻擊至 1000 筆 log
python attack_florence_pgd.py --experiment_root ../VQA/florence_victim/exp1_mixed1000 --num_samples 1000

# 3) 高斯探針 + caption + 分析
python run_defense_pipeline.py --exp exp1

# 4) 實驗二 / 三建集與攻擊（見 build_exp2.py / attack_supervisor_exp3.py）
```

環境：需 `%LOCALAPPDATA%\vlattack_florence_pkgs`（transformers 4.46 + timm + einops），見 `florence_engine.py`。
