import json
import os
from collections import Counter

# 設定你的路徑 (請依實際情況修改)
ann_root = os.environ.get('VLATTACK_ANN_ROOT', r'C:\Users\tracy_aboikg6\OneDrive\Desktop\program\VLAttack_Data\annotation')
train_json_path = os.path.join(ann_root, 'vqa_train.json')
output_path = os.path.join(ann_root, 'answer_list.json')

print(f"正在讀取: {train_json_path} ...")

try:
    with open(train_json_path, 'r') as f:
        train_data = json.load(f)

    # 統計所有答案出現的次數
    answer_counts = Counter()
    for item in train_data:
        # 確保有 'answer' 欄位
        if 'answer' in item:
            # 有些資料結構 answer 是列表，有些是字串，這裡做個防呆
            if isinstance(item['answer'], list):
                for ans in item['answer']:
                    answer_counts[ans] += 1
            else:
                answer_counts[item['answer']] += 1

    # BLIP 模型通常使用前 3129 個最常見的答案
    # 但為了保險，我們取出所有出現次數大於 0 的答案，並依照頻率排序
    # 注意：標準 BLIP 的列表順序很重要，理想情況是用官方檔案。
    # 這裡我們做一個替代方案：取出前 3128 個答案 (加一個 unknown 若需要)
    
    print(f"總共找到 {len(answer_counts)} 個不重複答案")
    
    # 根據 BLIP 原論文，這通常是 Top N (約 3129)。
    # 這裡我們直接生成最常見的 3129 個
    top_answers = [ans for ans, count in answer_counts.most_common(3129)]
    
    print(f"生成的 answer_list 包含 {len(top_answers)} 個答案")

    with open(output_path, 'w') as f:
        json.dump(top_answers, f)

    print(f"成功建立檔案: {output_path}")

except FileNotFoundError:
    print("錯誤：找不到 vqa_train.json，請確認路徑是否正確。")
except Exception as e:
    print(f"發生錯誤: {e}")