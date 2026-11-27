import json
import os

# 設定你的 JSON 檔案路徑
FILE_PATH = 'output/VQA/attack_dual_defense_log.json'

def get_answer_type(ans):
    """判斷題型分類"""
    ans = str(ans).lower().strip()
    if ans in ['yes', 'no']: return 'Yes/No'
    number_words = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'none']
    if ans.isdigit() or ans in number_words: return 'Number'
    return 'Other'

def print_compare_table():
    if not os.path.exists(FILE_PATH):
        print(f"❌ 找不到檔案: {FILE_PATH}")
        return

    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 建立統計字典 (包含縮放與高斯)
    stats = {key: {'total': 0, 'nodef': 0, 'scale_mild': 0, 'scale_strong': 0, 'gauss_mild': 0, 'gauss_strong': 0} 
             for key in ['Overall', 'Yes/No', 'Number', 'Other']}

    for item in data:
        gt = item.get('ground_truth')
        atype = get_answer_type(gt)
        
        # 讀取攻擊成功狀態 (1代表攻擊成功，模型仍答錯)
        nodef = 1 if item.get('attack_success') else 0
        scale_m = 1 if item.get('mild_defense', {}).get('attack_success') else 0
        scale_s = 1 if item.get('strong_defense', {}).get('attack_success') else 0
        
        # 使用 .get() 避免舊資料沒有高斯欄位而報錯
        gauss_m = 1 if item.get('gaussian_mild', {}).get('attack_success') else 0
        gauss_s = 1 if item.get('gaussian_strong', {}).get('attack_success') else 0

        keys_to_update = ['Overall', atype]
        for k in keys_to_update:
            stats[k]['total'] += 1
            stats[k]['nodef'] += nodef
            stats[k]['scale_mild'] += scale_m
            stats[k]['scale_strong'] += scale_s
            stats[k]['gauss_mild'] += gauss_m
            stats[k]['gauss_strong'] += gauss_s

    total_samples = stats['Overall']['total']

    # ================= 繪製超寬表格 =================
    line_len = 115
    print("=" * line_len)
    print(f"📊 VLAttack 多重防禦策略 PK 報告 (樣本數: {total_samples})")
    print("=" * line_len)
    
    # 表頭對齊
    print(f"{'類型 (Type)':<13} | {'總數':>5} | {'無防禦 ASR':>10} | {'縮放(弱) ASR':>11} | {'縮放(強) ASR':>11} | {'高斯(弱) ASR':>11} | {'高斯(強) ASR':>11}")
    print("-" * line_len)

    for key in ['Overall', 'Yes/No', 'Number', 'Other']:
        v = stats[key]
        if v['total'] == 0: continue
        
        # 計算百分比
        calc_asr = lambda x: (x / v['total']) * 100
        nodef_asr = calc_asr(v['nodef'])
        scale_m_asr = calc_asr(v['scale_mild'])
        scale_s_asr = calc_asr(v['scale_strong'])
        gauss_m_asr = calc_asr(v['gauss_mild'])
        gauss_s_asr = calc_asr(v['gauss_strong'])
        
        prefix = "🔴 " + key if key == 'Overall' else "   " + key
            
        print(f"{prefix:<12} | {v['total']:>5} | {nodef_asr:>9.2f}% | {scale_m_asr:>10.2f}% | {scale_s_asr:>10.2f}% | {gauss_m_asr:>10.2f}% | {gauss_s_asr:>10.2f}%")

    print("=" * line_len)
    
    # 自動分析結論
    print("\n💡 【系統自動判讀】")
    overall = stats['Overall']
    best_scale = calc_asr(overall['scale_strong'])
    best_gauss = calc_asr(overall['gauss_strong'])
    
    if best_scale < best_gauss:
        print(f"🏆 綜合來看，目前的實驗設定中【影像縮放 (Scale 0.6)】的防禦效果更好！(ASR: {best_scale:.1f}%)")
    else:
        print(f"🏆 綜合來看，目前的實驗設定中【高斯模糊 (Strong)】的防禦效果更好！(ASR: {best_gauss:.1f}%)")

if __name__ == "__main__":
    print_compare_table()