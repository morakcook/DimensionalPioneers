# DimensionalPioneers

## 주제
2D 이미지를 3D 객체 파일로 변환하는 프로젝트입니다.

### 설명
이 프로젝트는 단일 2D 이미지를 3D 파일(.glb)로 변환합니다. 주요 사용 모델은 ZERO123++와 DUSt3r입니다.

## 프로젝트 시나리오

### 1. 사진 준비하기
- **설명**: 따로 준비되어있지 않은 개인의 이미지를 준비합니다.(예: 핸드폰 카메라 사진, 인터넷 사진 다운로드)

### 2. ZERO123++을 이용한 이미지 증강
- **모델**: ZERO123++
- **학습 여부**: 증강하는 이미지의 세부적인 요소, 객체의 형태, 색체등의 정확도를 위한 학습이 필요합니다.
- **오픈소스 모델 링크**: [ZERO123++ GitHub](https://github.com/SUDO-AI-3D/zero123plus?tab=readme-ov-file)
- **설명**: ZERO123++는 고급 이미지 증강 기술을 사용하여 2D 이미지의 깊이와 텍스처 정보를 향상시켜, 3D 변환의 정확도를 높입니다.
- **이해**: zero123++모델을 사용하여 주어진 이미지의 색상정보, 공간정보(객체의 굴곡, 깊이)를 계산하여 학습시킨 6가지의 각도에서의 객체의 모습을 생성합니다.

### 3. 생성돤 이미지 전처리
- **사용**: python, cv2
- **설명**: 생성된 이미지들을 3d객체화를 할때 객체의 형태에 대한 정확도를 높이기 위한 이미지 전처리를 합니다.
    1. 이미지 동일 크기 및 비율 조정
    2. 이미지 샤프닝
    3. 이미지 노이즈 제거
  
### 4. DUSt3r 모델을 이용한 3D 객체화
- **모델**: DUSt3r
- **학습 여부**: 제공된 이미지들을 기반으로 3d객체를 생성함에 있어 정확도를 위한 학습이 필요합니다.
- **오픈소스 모델 링크**: [DUSt3r GitHub](https://github.com/naver/dust3r)
- **설명**: DUSt3r는 증강된 2D 이미지를 사용하여 실제 3D 객체 파일(.glb)로 변환합니다. 이 과정은 고도의 세부사항과 정밀도를 요구합니다.
- **이해**: DUSt3r모델은 제공된 이미지들이 보여지는 각도와 위치를 계산하고 depthmap을 계산하여 하나의 3d공간에 중첩하여 3d정보를 축척하여 3d객체 파일화(.glb)를 합니다.

### 5. 최종 결과 확인하기
- **사용lib**: three.js
- **설명**: 생성된 3d객체 파일을 웹이나 앱 화면에서 확인하는 로직을 사용합니다.
- **이해**: 3d파일(.glb)을 웹 화면상에서 시각적으로 확인할수있게 도와주는 javascript library입니다.

## 진행화면

<img width="80%" src="https://github.com/morakcook/DimensionalPioneers/blob/main/Reference/%EC%8B%A4%ED%96%89%ED%99%94%EB%A9%B4.gif"/>

## 사용하는 모델 및 라이센스 정의

### 1. ZERO123++
- **원본 출처**: [ZERO123++ GitHub](https://github.com/SUDO-AI-3D/zero123plus?tab=readme-ov-file)
- **라이센스**:
  - **코드 라이센스**: [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
  - **모델 가중치 라이센스**: [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/deed.ko)
- **사용 목적**: 1장의 2D 이미지로부터 6장의 다른 각도에서 본 이미지를 생성
- **수정 및 통합 사항**: 원본 코드를 기반으로 6장의 이미지를 자동으로 생성하는 기능을 추가함.

### 2. DUSt3r
- **원본 출처**: [DUSt3r GitHub](https://github.com/naver/dust3r)
- **라이센스**: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.ko)
- **사용 목적**: 생성된 6장의 이미지를 이용하여 3D 모델을 생성
- **수정 및 통합 사항**: 원본 코드를 수정하여 3D 모델을 자동으로 생성하는 기능을 추가함.
