import os
import subprocess
from typing import List
import cv2
import rembg
import torch
from cog import BasePredictor, Input, Path
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
from PIL import Image
import numpy as np
import cv2
import os
import torch
import trimesh
from scipy.spatial.transform import Rotation
from dust3r.inference import inference, load_model
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images
from dust3r.utils.device import to_numpy
from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

# Zero123++ 모델 생성자
class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        if not os.path.exists("weights"):
            os.mkdir("weights")


        print("Setting up pipeline...")

        self.pipeline = DiffusionPipeline.from_pretrained(
            f"{os.getcwd()}/weights/zero123plusplus",
            custom_pipeline=f"{os.getcwd()}/diffusers-support/",
            torch_dtype=torch.float16,
            local_files_only=True
        )
        self.pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.pipeline.scheduler.config, timestep_spacing='trailing'
        )
        self.pipeline.to('cuda:0')

    def predict(
        self,
        image: Path = Input(description="Input image. Aspect ratio should be 1:1. Recommended resolution is >= 320x320 pixels."),
        remove_background: bool = Input(description="Remove the background of the input image", default=False),
        return_intermediate_images: bool = Input(description="Return the intermediate images together with the output images", default=False),
        Interim_deliverables: str = Input(),
        ) -> List[Path]:
        """Run a single prediction on the model"""
        outputs = []

        cond = Image.open(str(image))
        ret, mask = cv2.threshold(np.array(cond.split()[-1]), 0, 255, cv2.THRESH_BINARY)
        x, y, w, h = cv2.boundingRect(mask)
        image_filename = "original" + image.suffix

        # optional background removal step
        if remove_background:
            rembg_session = rembg.new_session()
            cond = rembg.remove(cond, session=rembg_session)
            # image should be a png after background removal
            image_filename += ".png"

        if return_intermediate_images:
            temp_original = f"/tmp/{image_filename}"
            cond.save(temp_original)
            outputs.append(temp_original)

        all_results = self.pipeline(cond, num_inference_steps=100).images[0]
        side_len = all_results.width//2
        subimages = [all_results.crop((x, y, x + side_len, y+side_len)) for y in range(0, all_results.height, side_len) for x in range(0, all_results.width, side_len)]
        for i, output_img in enumerate(subimages):
            filename = f"{Interim_deliverables}/output{i+1}.png"
            output_img.save(filename)
            outputs.append(filename)

        return([Path(output) for output in outputs])

# 이미지 크기 및 비율 맞추기
def Image_preprocessing(image_paths):

  # 목표 이미지 크기
  target_width = 512
  target_height = 512

  # 각 이미지에 대해서 크기 및 비율 조정을 수행
  for i, image_path in enumerate(image_paths):
      # 이미지를 읽어옴
      image = cv2.imread(image_path)

      # 샤프닝을 위한 커널 정의
      kernel_sharpening = np.array([[-1, -1, -1],
                                    [-1,  9, -1],
                                    [-1, -1, -1]])

      # 이미지 크기 조정
      resized_image = cv2.resize(image, (target_width, target_height))

      # 이미지에 샤프닝 적용
      sharpened_image = cv2.filter2D(resized_image, -1, kernel_sharpening)

      # Gaussian Blur 적용
      # (5, 5)는 Gaussian Kernel의 크기, 0은 표준 편차를 자동으로 계산하게 함
      denoised_image = cv2.GaussianBlur(sharpened_image, (5, 5), 0)

      cv2.imwrite(image_path, denoised_image)
      # 결과 확인
      print(f'Image {i+1} saved to:', image_path)

# 3D 장면 출력을 GLB 파일 형식으로 변환하는 함수입니다.
def _convert_scene_output_to_glb(outdir, imgs, pts3d, mask, focals, cams2world, cam_size=0.05,
                                 cam_color=None, as_pointcloud=False,
                                 transparent_cams=False):
    assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)

    scene = trimesh.Scene()

    if as_pointcloud:
        pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
        pct = trimesh.PointCloud(pts.reshape(-1, 3), colors=col.reshape(-1, 3))
        scene.add_geometry(pct)
    else:
        meshes = []
        for i in range(len(imgs)):
            meshes.append(pts3d_to_trimesh(imgs[i], pts3d[i], mask[i]))
        mesh = trimesh.Trimesh(**cat_meshes(meshes))
        scene.add_geometry(mesh)

    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
    scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
    outfile = os.path.join(outdir, 'scene.glb')
    scene.export(file_obj=outfile)
    return outfile


# 이미지들로부터 3D 모델을 처리하고 생성하는 메인 함수입니다.
def process_images_to_3d_model(input_image_paths, output_file_path, weights_path, device='cuda'):
    # 모델을 로드합니다.
    model = load_model(weights_path, device, verbose=False)

    # 입력 이미지들을 로드합니다.
    imgs = load_images(input_image_paths, size=512, verbose=False)

    # 이미지 쌍을 생성합니다.
    pairs = make_pairs(imgs, scene_graph='complete', prefilter=None, symmetrize=True)
    # 추론을 수행합니다.
    output = inference(pairs, model, device, batch_size=batch_size, verbose=False)

    # 글로벌 정렬을 수행하여 3D 장면을 생성합니다.
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer, verbose=False)
    loss = scene.compute_global_alignment(init='mst', niter=500, schedule='linear', lr=0.01)

    # 필요한 후처리를 수행합니다.
    scene = scene.clean_pointcloud()
    scene = scene.mask_sky()

    rgbimg = scene.imgs
    focals = scene.get_focals().cpu()
    cams2world = scene.get_im_poses().cpu()
    pts3d = to_numpy(scene.get_pts3d())
    scene.min_conf_thr = float(scene.conf_trf(torch.tensor(3.0)))
    msk = to_numpy(scene.get_masks())

    # 최종적으로 GLB 파일을 생성합니다.
    _convert_scene_output_to_glb(os.path.dirname(output_file_path), rgbimg, pts3d, msk, focals, cams2world, as_pointcloud=False,
                                 transparent_cams=False, cam_size=0.05)

def run(input_path, Interim_deliverables, output_path, weights_path):
  # Predictor 인스턴스 생성
  predictor = Predictor()

  # 모델 설정
  predictor.setup()

  # 이미지 변환 실행
  result_paths = predictor.predict(image=Path(input_path), remove_background=True, return_intermediate_images=False,Interim_deliverables=Interim_deliverables)

  # 경로 str화
  Interim_deliverables_paths = []
  for path in result_paths:
    Interim_deliverables_paths.append(str(path))

  # 생성된 이미지 전처리
  Image_preprocessing(Interim_deliverables_paths)

  # 특정 GPU 설정을 위한 torch 설정입니다.
  torch.backends.cuda.matmul.allow_tf32 = True
  batch_size = 1  # 배치 사이즈를 1로 설정합니다.

  # 이미지들로부터 3D 모델을 처리하고 생성합니다.
  process_images_to_3d_model(Interim_deliverables_paths, output_path, weights_path)

weights_path = f"{os.getcwd()}/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
input = f"{os.getcwd()}/Data/Input/img006.jpg"
Interim_deliverables = f"{os.getcwd()}/Data/Interim_deliverables"
output_file_path = f"{os.getcwd()}/Data/Output/output_model.glb"
run(input, Interim_deliverables, output_file_path, weights_path)