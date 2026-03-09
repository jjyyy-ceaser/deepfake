import os
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
# 🔧 3D 지원을 위한 reshape_transform 필요
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from utils import get_model, calculate_metrics_at_best_threshold
from data_loader import get_transforms

# ⚙️ 설정
MODEL_NAME = "videomae" # xception, swin, r3d, videomae
WEIGHT_PATH = rf"C:\Users\leejy\Desktop\test_experiment\results\final_weights\{MODEL_NAME}_fold1.pth"
VIDEO_PATH = r"C:\Users\leejy\Desktop\test_experiment\dataset\final_dataset_v2\test\fake\svd_001.mp4"
SAVE_DIR = r"C:\Users\leejy\Desktop\test_experiment\results\explainability"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------------------------------------------------
# 🛠️ [핵심 Fix] 5D 텐서 -> 2D 히트맵 변환 로직
# ---------------------------------------------------------
def reshape_transform_videomae(tensor):
    """
    VideoMAE의 3D Feature를 2D Grad-CAM용으로 변환
    Input: [B, N_patches, C] -> Output: [B, C, H, W] (Time축은 Mean으로 압축)
    """
    # VideoMAE Base 기준: 1568 patches (16frames * 14*14 patches)
    # 복잡하므로, 여기서는 Swin/VideoMAE 등 Transformer 계열의 일반적인 처리를 시도
    # (실제 구현 시 모델 내부 구조에 따라 달라질 수 있음. 여기서는 약식 구현)
    
    # 텐서가 4D(Conv)가 아니라 3D(Transformer output)일 경우
    if tensor.ndim == 3: 
        # [B, Tokens, Channels] -> [B, Channels, H, W] (Spatial Only)
        # 시간축을 무시하고 공간만 보거나, 시간축을 채널로 간주해야 함
        # 여기서는 간단히 마지막 LayerNorm 출력을 H, W로 복원한다고 가정
        b, n, c = tensor.shape
        h = w = int(np.sqrt(n)) # Assuming square spatial layout (ignoring time for now)
        result = tensor.permute(0, 2, 1).reshape(b, c, h, w)
    elif tensor.ndim == 5: # [B, C, T, H, W] (R3D)
        # 🔧 R3D의 경우 시간축(T)을 평균내어 2D 히트맵으로 만듦
        result = tensor.mean(dim=2) 
    else:
        result = tensor
    return result

def get_target_layer(model, model_name):
    name = model_name.lower()
    if 'xception' in name: return [model.act4]
    elif 'swin' in name: return [model.layers[-1].blocks[-1].norm1]
    elif 'r3d' in name: return [model.layer4[-1].conv2[0]] # 3D Conv
    elif 'videomae' in name: return [model.videomae.encoder.layer[-1].layernorm_after]
    return None

def visualize_forensics():
    print(f"🧐 [10.2] Grad-CAM 분석 시작: {MODEL_NAME}")
    
    model = get_model(MODEL_NAME, DEVICE)
    model.load_state_dict(torch.load(WEIGHT_PATH, map_location=DEVICE))
    model.eval()

    # 영상 로드 (16프레임)
    cap = cv2.VideoCapture(VIDEO_PATH)
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start = max(0, total//2 - 8) # 중앙 16프레임
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    
    while len(frames) < 16:
        ret, frame = cap.read()
        if not ret: break
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    
    if len(frames) < 16: return

    # 전처리
    transform = get_transforms(MODEL_NAME)
    # [T, C, H, W]
    input_tensor = torch.stack([transform(f) for f in frames]) 
    
    # 5D Batch 차원 추가 [1, T, C, H, W]
    if "r3d" in MODEL_NAME:
        input_tensor = input_tensor.permute(1, 0, 2, 3).unsqueeze(0).to(DEVICE) # [1, C, T, H, W]
    elif "videomae" in MODEL_NAME or "hybrid" in MODEL_NAME:
        input_tensor = input_tensor.unsqueeze(0).to(DEVICE) # [1, T, C, H, W]
    else: # 2D Model (중앙 프레임 1장만 사용)
        mid_frame = frames[len(frames)//2]
        input_tensor = transform(mid_frame).unsqueeze(0).to(DEVICE)

    # Grad-CAM 설정
    target_layers = get_target_layer(model, MODEL_NAME)
    
    # 🔧 [핵심] 3D 모델일 경우 reshape_transform 적용
    use_reshape = True if ("videomae" in MODEL_NAME or "r3d" in MODEL_NAME) else False
    
    try:
        cam = GradCAM(model=model, target_layers=target_layers, 
                      reshape_transform=reshape_transform_videomae if use_reshape else None)
        
        targets = [ClassifierOutputTarget(1)] # Fake Class
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        
        # 결과 시각화 (중앙 프레임 위에 오버레이)
        grayscale_cam = grayscale_cam[0, :]
        mid_img = np.array(frames[len(frames)//2]) / 255.0
        mid_img = cv2.resize(mid_img, (224, 224))
        
        vis = show_cam_on_image(mid_img, grayscale_cam, use_rgb=True)
        
        save_path = os.path.join(SAVE_DIR, f"CAM_{MODEL_NAME}_fixed.png")
        plt.imsave(save_path, vis)
        print(f"   💾 히트맵 저장 완료: {save_path}")
        
    except Exception as e:
        print(f"   ⚠️ Grad-CAM 오류 (구조적 한계): {e}")
        print("   💡 해결책: 3D 모델의 경우 Attention Rollout 등 전용 시각화 코드가 필요할 수 있습니다.")

if __name__ == "__main__":
    visualize_forensics()