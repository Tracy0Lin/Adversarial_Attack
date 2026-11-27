import os
import json
import pandas as pd
import re
from tabulate import tabulate

def analyze_pure_attack_stealth(base_path):
    all_results = []
    
    if not os.path.exists(base_path):
        print(f"找不到路徑: {base_path}")
        return

    # 正則表達式：嚴格匹配 steps(數字)_alpha(數字) 資料夾
    folder_pattern = re.compile(r'^steps\d+_alpha\d+\.?\d*$', re.IGNORECASE)
    subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    
    for subdir in subdirs:
        if not folder_pattern.match(subdir):
            continue

        log_path = os.path.join(base_path, subdir, "attack_dual_defense_log.json")
        
        if os.path.exists(log_path):
            try:
                with open(log_path, 'r', encoding='utf-8') as f:
                    logs = json.load(f)
                
                if not logs or not isinstance(logs, list):
                    continue
                if 'attack_success' not in logs[0]:
                    continue

                valid_logs = [l for l in logs if 'attack_success' in l]
                count = len(valid_logs)
                
                # ===== 1. 純攻擊破壞力 (Attack Power) =====
                asr = sum(1 for log in valid_logs if log['attack_success']) / count 
                
                # ===== 2. 三位一體視覺隱蔽性 (Image Stealth) =====
                avg_ssim = sum(log['ssim']['original_vs_attack'] for log in valid_logs) / count
                avg_lpips = sum(log['lpips']['original_vs_attack'] for log in valid_logs) / count
                avg_clip = sum(log['clip']['original_vs_attack'] if 'clip' in log else 0.0 for log in valid_logs) / count
                
                # 轉換 LPIPS 使得「感知隱蔽性」數值越大越好 (LPIPS 越低，感知保真度越高)
                perceptual_stealth = max(0.0, 1.0 - avg_lpips)
                
                # 綜合隱蔽性總得分 (結合底層像素像素 + 中層視覺感知 + 高層多模態語意)
                stealth_score = (avg_ssim + perceptual_stealth + avg_clip) / 3.0 

                # ===== 3. 拋棄硬限制！改採「自然權衡評分公式」 =====
                # 💡 這裡將權重設為：攻擊效能佔 50%，視覺與語意隱蔽性佔 50%
                # 這樣最科學，完全交由多目標優化的自然分布來決定排名
                attack_stealth_index = (asr * 0.5) + (stealth_score * 0.5)

                all_results.append({
                    "Hyperparam": subdir, 
                    "Base ASR (%)": round(asr * 100, 2),
                    "Avg SSIM (Pixel)": round(avg_ssim, 4),
                    "Avg LPIPS (Percept())": round(avg_lpips, 4),
                    "Avg CLIP (Semantic)": round(avg_clip, 4), 
                    "Stealth_Score": round(stealth_score, 4),
                    "Attack_Stealth_Index": round(attack_stealth_index, 4), # 綜合平衡排名指標
                    "Samples": count
                })
            except Exception as e:
                print(f"解析資料夾 {subdir} 出錯: {e}")

    if not all_results:
        print("❌ 找不到符合新型「三位一體」格式的正式實驗 Log。")
        return

    df = pd.DataFrame(all_results)
    
    # 💡 依據無損自然的「純攻擊隱蔽性指數」進行降序排序，揪出黃金平衡組合
    df = df.sort_values(by="Attack_Stealth_Index", ascending=False)

    print("\n" + "="*85)
    print("Steps vs Alpha")
    print("="*85)
    print(tabulate(df, headers='keys', tablefmt='plain', showindex=False, stralign='left', numalign='left'))
    print("="*85)

    # 儲存 CSV。這份 CSV 數據完整、有層次，非常適合拿來直接匯入 Excel 畫折線圖！
    output_file = "attack_stealth_natural_summary.csv"
    df.to_csv(output_file, index=False)
    print(f"📊 權衡數據已成功儲存至: {output_file}\n")

if __name__ == "__main__":
    base_dir = os.environ.get("VLATTACK_VQA_OUT", r"C:\Users\tracy_aboikg6\OneDrive\Desktop\program\VLAttack\BLIP_attack\output\VQA")
    analyze_pure_attack_stealth(base_dir)