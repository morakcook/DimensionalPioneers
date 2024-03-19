import cv2
import torch
import open3d as o3d
import matplotlib.pyplot as plt

# 이미지 파일 경로를 직접 지정
filename = "path/to/your/image.jpg"  # 'path/to/your/image.jpg'를 실제 이미지 파일 경로로 바꿉니다.

# 모델 타입 선택
model_type = "DPT_Large"     # MiDaS v3 - Large     (높은 정확도, 낮은 추론 속도)
#model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (중간 정확도, 중간 추론 속도)
#model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (낮은 정확도, 높은 추론 속도)

# MiDaS 모델 로드
midas = torch.hub.load("intel-isl/MiDaS", model_type)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

# 변환(transform) 로드
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform    

# 이미지 읽기 및 변환
img = cv2.imread(filename)
plt.imshow(img)  # BGR 이미지를 보여줌; plt.imshow()에 RGB 이미지를 넣어야 올바른 색상이 표시됨을 주의하세요.
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR에서 RGB로 변환

# 이미지 변환 및 모델을 통해 깊이 맵 예측
input_batch = transform(img).to(device)
with torch.no_grad():
    prediction = midas(input_batch)
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

output = prediction.cpu().numpy()

# 깊이 맵 시각화
plt.imshow(output, cmap='inferno')  # 깊이 맵을 시각화하기 위해 colormap을 'inferno'로 설정
plt.show()  # 이 줄을 활성화하여 깊이 맵을 실제로 보여줍니다.



# MiDaS 모델로부터 얻은 깊이 맵 결과
depth_map = output  # 여기에 위에서 얻은 깊이 맵 데이터를 삽입합니다.

# 이미지의 높이와 너비
height, width = depth_map.shape

# 포인트 클라우드 데이터를 위한 리스트
points = []

# 깊이 맵을 반복하면서 3D 좌표를 계산
for v in range(height):
    for u in range(width):
        # 깊이 값
        depth = depth_map[v, u]

        # 여기서는 간단한 변환을 사용합니다. 실제 응용에서는 카메라의 내부 파라미터를 사용해야 합니다.
        # 3D 공간의 X, Y, Z 좌표 계산 (간단한 예제를 위한 가정)
        x = (u - width / 2) * depth / 1000.0
        y = (v - height / 2) * depth / 1000.0
        z = depth

        # 계산된 좌표를 포인트 리스트에 추가
        points.append([x, y, z])

# 포인트 클라우드 객체 생성 및 포인트 설정
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)

# 포인트 클라우드 시각화
o3d.visualization.draw_geometries([point_cloud])