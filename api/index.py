from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing import image
import io
import base64
import os

app = Flask(__name__)

# 클래스 이름
classes = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# EfficientNetV2-M 모델 로드
model_url = "https://tfhub.dev/google/efficientnetv2/m/feature-vector/2"
model = hub.KerasLayer(model_url, trainable=False)

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

def preprocess_image(file):
    img = image.load_img(io.BytesIO(file.read()), target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def predict_image(file):
    # 이미지를 로드하고 전처리
    img_array = preprocess_image(file)

    # 특징 벡터 추출
    features = model(img_array)
    
    # 간단한 분류기 사용 (여기서는 예시로 임의의 가중치를 사용)
    # 실제로는 별도의 학습된 분류기를 사용해야 합니다.
    predictions = np.dot(features, np.random.rand(features.shape[-1], len(classes)))
    
    # 예측된 클래스 인덱스 추출
    predicted_class = np.argmax(predictions, axis=1)[0]
    
    # 예측된 클래스 인덱스를 레이블로 변환
    predicted_label = classes[predicted_class]
    
    return predicted_label

if __name__ == '__main__':
    # 애플리케이션 실행
    app.run(debug=True)