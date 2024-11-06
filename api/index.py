from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import io
import base64
import gdown  # 추가된 부분
import os  # 추가된 부분

app = Flask(__name__)

# 클래스 이름
classes = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# 모델 파일을 구글 드라이브에서 다운로드
model_url = 'https://drive.google.com/file/d/1TL3fLEy_l79xfRyhFpcc1pZi8_lQ9eJe/view?usp=sharing'  # YOUR_FILE_ID를 실제 파일 ID로 변경
model_path = 'best_model_resnet50.keras'

if not os.path.exists(model_path):
    gdown.download(model_url, model_path, quiet=False)

# 모델 로드
model = load_model(model_path)

@app.route('/')
def home():
    # 홈 페이지 렌더링
    return render_template('index.html')

@app.route('/upload_file', methods=['POST'])
def upload_file():
    # 파일이 요청에 포함되어 있는지 확인
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    # 파일 이름이 비어 있는지 확인
    if file.filename == '':
        return redirect(request.url)
    if file:
        # 이미지 예측 수행
        prediction = predict_image(file)
        file.seek(0)  # 파일 포인터를 처음으로 되돌립니다.
        image_data = file.read()
        # 이미지를 base64로 인코딩
        image_url = f"data:image/jpeg;base64,{base64.b64encode(image_data).decode('utf-8')}"
        # 예측 결과와 이미지를 렌더링
        return render_template('index.html', uploaded_image=image_url, classification_result=prediction)
    return redirect(url_for('home'))

def predict_image(file):
    # 이미지를 로드하고 전처리
    img = image.load_img(io.BytesIO(file.read()), target_size=(32, 32))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # 예측 수행
    predictions = model.predict(img_array)
    
    # 예측된 클래스 인덱스 추출
    predicted_class = np.argmax(predictions, axis=1)[0]
    
    # 예측된 클래스 인덱스를 레이블로 변환
    predicted_label = classes[predicted_class]
    
    return predicted_label

if __name__ == '__main__':
    # 애플리케이션 실행
    app.run(debug=True)