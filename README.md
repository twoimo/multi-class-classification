# 이미지 분류 서비스

이 프로젝트는 이미지를 업로드하고 분류 결과를 제공하는 웹 애플리케이션입니다.

## 분류 가능한 카테고리

이 애플리케이션은 다음 10가지 카테고리로 이미지를 분류할 수 있습니다:

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

## 요구 사항

- Python
- Flask
- gdown
- numpy
- tensorflow

## 설치 방법

1. 이 저장소를 클론합니다:

```bash
git clone https://github.com/twoimo/multi-class-classification.git
cd image-classification-service
```

2. 필요한 패키지를 설치합니다:

```bash
pip install -r requirements.txt
```

## Flask 로컬 서버 실행

1. api 폴더에 있는 index.py 파일을 루트 디렉토리로 이동시키고 app.py로 이름을 변경합니다(Vercel 배포하기 위해서 디렉토리 구조 변경함):
   
```bash
mv api/index.py app.py
```

2. Flask 애플리케이션을 실행합니다:

```bash
flask run --debug
```

## 사용 방법

1. 메인 페이지에서 이미지를 업로드합니다.
2. 업로드된 이미지의 분류 결과를 확인합니다.
3. 샘플 이미지를 다운로드할 수도 있습니다.
