from flask import Flask, render_template, request, jsonify
import os
import time

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# ===== 1. 首頁路由 (雙重設定) =====
@app.route('/')
@app.route('/index.html')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    # 1. 接收前端傳來的資料
    file = request.files['image']
    question = request.form.get('question')
    eps = request.form.get('eps')

    # 2. 儲存原始檔案
    img_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(img_path)

    # 3. 模擬攻擊運算 (暫時先用假的，之後這邊改為呼叫你的 PGD 模組)
    # 我們先假設攻擊後圖片檔名叫 adv_xxx.png
    result_filename = f"adv_{file.filename}"
    result_path = os.path.join(RESULT_FOLDER, result_filename)
    
    # 模擬運算耗時
    time.sleep(1) 
    # 這裡暫時直接把原圖複製過去當作「偽對抗樣本」
    import shutil
    shutil.copy(img_path, result_path)

    # 4. 回傳 JSON 給前端，告訴它去哪裡拿圖
    return jsonify({
        "status": "done",
        "original_url": f"/{img_path}",
        "adv_url": f"/static/results/{result_filename}",
        "prediction": "這裡顯示模型的回答 (範例)"
    })

# ===== 2. VLAttack 頁面路由 (雙重設定) =====
@app.route('/vlattack')
@app.route('/vlattack.html')
def vlattack():
    return render_template('vlattack.html')

# ===== 3. AttackVLM 頁面路由 (雙重設定) =====
@app.route('/attackvlm')
@app.route('/attackvlm.html')
def attackvlm():
    return render_template('attackvlm.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005, debug=True)