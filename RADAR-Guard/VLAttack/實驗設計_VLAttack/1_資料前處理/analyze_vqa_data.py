import json
import os

# ================= 設定區 =================
FILE_PATH = 'right_vqa_ans_table.txt' 
# ==========================================

def get_answer_type(ans):
    """
    根據 VQA v2 的定義分類答案類型
    """
    #轉小寫並去頭尾空白
    ans = str(ans).lower().strip()
    
    # 1. Yes/No 類型
    if ans in ['yes', 'no']:
        return 'Yes/No'
    
    # 2. Number 類型 (包含數字字元與常見英文數字單字)
    number_words = [
        'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
        'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen',
        'none' # VQA 中 none 有時被視為 0
    ]
    
    # 判斷是否為純數字 (例如 "2", "10") 或是在數字單字表中
    if ans.isdigit() or ans in number_words:
        return 'Number'
    
    # 3. Other 類型
    return 'Other'

def analyze_data():
    if not os.path.exists(FILE_PATH):
        print(f"❌ 錯誤: 找不到檔案 {FILE_PATH}")
        return

    print(f"📂 正在讀取 JSON 檔案: {FILE_PATH} ...")
    
    try:
        with open(FILE_PATH, 'r', encoding='utf-8') as f:
            # 關鍵修正：使用 json.load 讀取字典結構
            data_dict = json.load(f)
            
        # 我們只關心 "答案" (Value)，不關心 "ID" (Key)
        answers = list(data_dict.values())
        
    except json.JSONDecodeError:
        print("❌ 錯誤：檔案內容不是有效的 JSON 格式。")
        print("請確認檔案內容是否以 '{' 開頭並以 '}' 結尾。")
        return

    total_count = len(answers)
    print(f"✅ 讀取成功！總共取得 {total_count} 筆資料。\n")

    # 初始化計數器
    stats = {
        'Yes/No': 0,
        'Number': 0,
        'Other': 0
    }
    
    # 範例儲存 (每類存前5個)
    examples = {
        'Yes/No': [],
        'Number': [],
        'Other': []
    }

    # 開始分類
    for ans in answers:
        a_type = get_answer_type(ans)
        stats[a_type] += 1
        
        if len(examples[a_type]) < 5:
            examples[a_type].append(ans)

    # ================= 文字輸出統計結果 =================
    print("="*50)
    print(f"📊 VQA 資料集統計報告")
    print("="*50)
    
    for k, v in stats.items():
        if total_count > 0:
            percentage = (v / total_count) * 100
        else:
            percentage = 0
            
        print(f"🔹 {k} 類型")
        print(f"   數量: {v} 筆")
        print(f"   佔比: {percentage:.2f}%")
        print(f"   範例: {examples[k]}")
        print("-" * 50)

if __name__ == "__main__":
    analyze_data()