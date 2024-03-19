# DimensionalPioneers
공간의 2D이미지를 3D이미지로 변경하는 프로젝트입니다.

## 프로젝트 시나리오

### 1. 사진 준비하기
- **모델 사용 안 함**: 이 단계에서는 모델을 사용하지 않습니다. 대신 기본적인 이미지 편집 소프트웨어를 사용할 수 있습니다.
- **학습 여부**: 학습 과정 없음.
- **오픈소스 도구 예시**: GIMP 또는 Paint.NET과 같은 오픈소스 이미지 편집 소프트웨어를 사용해 이미지를 전처리할 수 있습니다.

### 2. 사진에서 깊이 알아내기
- **모델**: **MiDaS** (Multi-scale Deep Stereo Matching System).
- **학습 여부**: 사전 학습됨.
- **오픈소스 모델 링크**: [MiDaS GitHub](https://github.com/intel-isl/MiDaS)
- **이유 및 설명**: MiDaS는 다양한 장면과 객체에 대해 뛰어난 깊이 추정 성능을 보여줍니다. GitHub에서 제공되는 사전 학습된 모델을 사용해 바로 깊이 맵을 생성할 수 있습니다.

### 3. 3D 이미지 만들기
- **모델**: **Neural Radiance Fields (NeRF)**.
- **학습 여부**: 사용자 데이터에 따라 추가 학습 필요.
- **오픈소스 모델 링크**: [tiny-nerf GitHub](https://github.com/bmild/nerf)
- **이유 및 설명**: NeRF는 복잡한 3D 장면을 재현할 수 있는 강력한 모델입니다. tiny-nerf는 NeRF의 간소화된 구현으로, 학습 과정과 결과의 이해에 도움이 됩니다.

### 4. 이미지를 더 사실적으로 만들기
- **모델**: **Pix2Pix**.
- **학습 여부**: 사전 학습됨 및 추가 학습 필요.
- **오픈소스 모델 링크**: [Pix2Pix GitHub](https://github.com/phillipi/pix2pix)
- **이유 및 설명**: Pix2Pix는 이미지-대-이미지 변환 작업에 널리 사용되는 GAN 모델입니다. 3D 모델에 사실적인 텍스처를 추가하거나 세부 사항을 개선하는 데 사용할 수 있습니다.

### 5. 최종 결과 확인하기
- **모델 사용 안 함**: 이 단계는 모델 학습과 관련 없이, 3D 뷰어나 시각화 도구를 사용하여 수행됩니다.
- **학습 여부**: 학습 과정 없음.
- **오픈소스 도구 예시**: Blender 또는 MeshLab과 같은 3D 모델링 및 시각화 소프트웨어를 사용해 최종 3D 모델을 확인할 수 있습니다.

## 사용하는 모델 소스 예제 사이트

### 1. **MiDaS** (Multi-scale Deep Stereo Matching System)
- https://pytorch.org/hub/intelisl_midas_v2/
### 2. **Neural Radiance Fields (NeRF)**
-
### 3. **Pix2Pix**
- 
