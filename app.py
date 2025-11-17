from flask import Flask, request, jsonify
import os
import cv2
import numpy as np

from api import apifun


app = Flask(__name__)

@app.route('/process', methods=['POST'])
def process_data():
    try:
        # 获取传递的整数
        paper = int(request.form['paper'])
        #获取返回结果
        score_select=int(request.form['result'])
        ttf_select=int(request.form['ttf'])
        # 获取上传的图片
        image_file = request.files['image']
         # 读取图片到内存
        img_bytes = image_file.read()  # 获取图像字节流
        # 打印确认获取到的参数
        print(f"Received paper: {paper}, result: {score_select}, ttf: {ttf_select}")
        print(f"Received image: {image_file.filename}")
        # 使用 OpenCV 从字节流读取图像
        img_np = np.frombuffer(img_bytes, np.uint8)  # 将字节流转换为 numpy 数组
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)  # 使用 cv2.imdecode 解码图像
        # 如果图像为空，则返回错误
        if img is None:
            return jsonify({'status': 'error', 'message': 'Failed to decode image'})
        result = apifun(paper, score_select, ttf_select,img)
        # 根据 score_select 返回不同的数据
        if score_select == 1:
            response = {
                'status': 'success',
                'mean_score': result
            }
        elif score_select == 2:
            response = {
                'status': 'success',
                'score_history': result
            }
        
        return jsonify(response)

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })
        
if __name__ == '__main__':
    app.run(debug=True,port=7008)
#curl -X POST -F "paper=1" -F "result=2" -F "ttf=1" -F "image=@/data/yly/text_data/multi/test5.jpg" http://127.0.0.1:7008/process 