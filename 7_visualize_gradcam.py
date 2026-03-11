import os
import warnings
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from utils import get_model
from data_loader import get_transforms

# 🔧 경고 메시지 제거
warnings.filterwarnings("ignore")

# ======================================================
# ⚙️ [설정] 평가 모델 및 타겟 파일
# ======================================================
MODEL_NAME = "swin_tiny_patch4_window7_224" # 모델 구조 로드용 (변경하지 마세요)

# 🚨 [수정 필수] results\final_weights 폴더에 있는 "실제 파일명"을 적어주세요!
# (예: "swin_f1.pth", "swin_best.pth", "swin_tiny_patch4_window7_224_f1.pth" 등)
WEIGHT_FILE = "swin_f1.pth" 
WEIGHT_PATH = rf"C:\Users\leejy\Desktop\test_experiment\results\final_weights\{WEIGHT_FILE}"

BASE_DIR = r"C:\Users\leejy\Desktop\test_experiment\dataset\final_dataset_v2"
FILENAME = "svd_003.mp4" 

SAVE_DIR = r"C:\Users\leejy\Desktop\test_experiment\results\explainability"
os.makedirs(SAVE_DIR, exist_ok=True)

DOMAIN_PATHS = {
    "Raw":          os.path.join(BASE_DIR, "raw", "test", "fake", FILENAME),
    "YouTube":      os.path.join(BASE_DIR, "youtube", "fake", FILENAME),
    "Instagram":    os.path.join(BASE_DIR, "instagram", "fake", FILENAME),
    "Kakao_High":   os.path.join(BASE_DIR, "kakao_high", "fake", FILENAME),
    "Kakao_Normal": os.path.join(BASE_DIR, "kakao_normal", "fake", FILENAME)
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------
# 🛠️ [Fix 1] 모델 호환 래퍼
# ---------------------------------------------------------
class HybridGradCAMWrapper(nn.Module):
    def __init__(self, model, frames=25):
        super().__init__()
        self.model = model
        self.frames = frames
    def forward(self, x):
        x = x.unsqueeze(1).repeat(1, self.frames, 1, 1, 1)
        out = self.model(x)
        return out.logits if hasattr(out, 'logits') else out

class TemporalGradCAMWrapper(nn.Module):
    def __init__(self, model, model_name):
        super().__init__()
        self.model = model
        self.model_name = model_name.lower()
        
    def forward(self, x):
        b, tc, h, w = x.shape
        c = 3
        t = tc // c
        x = x.view(b, t, c, h, w)
        if "r3d" in self.model_name:
            x = x.permute(0, 2, 1, 3, 4) 
        out = self.model(x)
        if hasattr(out, 'logits'):
            return out.logits
        return out

# ---------------------------------------------------------
# 🛠️ [Fix 2] 타겟 레이어 및 3D/Swin Reshape
# ---------------------------------------------------------
def get_target_layer(model, name):
    name = name.lower()
    base_m = model.model if hasattr(model, 'model') else model
    
    if "xception" in name:
        return [base_m.conv4] 
    elif "swin" in name:
        return [base_m.layers[-1].blocks[-1].norm1]
    elif "r3d" in name:
        return [base_m.layer4[-1].conv2[0]]
    elif "videomae" in name:
        return [base_m.videomae.encoder.layer[-1].layernorm_before]
    elif "gru" in name or "hybrid" in name:
        return [base_m.backbone.stages[-1].blocks[-1].conv_dw]
    else:
        raise ValueError(f"지원하지 않는 모델: {name}")

def reshape_transform_3d(tensor, model_name):
    name = model_name.lower()
    
    if "swin" in name:
        # 🚨 [Swin 바둑판 해결 핵심] 텐서 모양 복원
        if tensor.dim() == 4:
            return tensor.permute(0, 3, 1, 2)
        elif tensor.dim() == 3: 
            b, l, c = tensor.shape
            h = int(np.sqrt(l))
            return tensor.reshape(b, h, h, c).permute(0, 3, 1, 2)
            
    elif "videomae" in name:
        b, num_tokens, c = tensor.shape
        t = num_tokens // (14 * 14) 
        result = tensor.reshape(b, t, 14, 14, c)
        return result.mean(dim=1).permute(0, 3, 1, 2) 
        
    elif "r3d" in name:
        return tensor.mean(dim=2) 
        
    return tensor

def main():
    print(f"🎬 Grad-CAM 분석 시작: {MODEL_NAME.upper()}")
    
    base_model = get_model(MODEL_NAME, device=DEVICE, num_classes=2)
    
    # 가중치 파일 존재 여부 먼저 확인
    if not os.path.exists(WEIGHT_PATH):
        print(f"❌ [에러] 가중치 파일을 찾을 수 없습니다: {WEIGHT_PATH}")
        print("   -> 23번째 줄의 WEIGHT_FILE 변수를 실제 파일명으로 변경해주세요!")
        return
        
    state_dict = torch.load(WEIGHT_PATH, map_location=DEVICE, weights_only=True)
    new_state_dict = {k.replace('module.', '').replace('backbone.', ''): v for k, v in state_dict.items()}
    base_model.load_state_dict(new_state_dict, strict=False)
    base_model.eval()

    if "hybrid" in MODEL_NAME.lower():
        model_for_cam = HybridGradCAMWrapper(base_model, frames=25).to(DEVICE)
    elif "r3d" in MODEL_NAME.lower() or "videomae" in MODEL_NAME.lower():
        model_for_cam = TemporalGradCAMWrapper(base_model, MODEL_NAME).to(DEVICE)
    else:
        model_for_cam = base_model

    target_layers = get_target_layer(model_for_cam, MODEL_NAME)
    
    if any(x in MODEL_NAME for x in ["videomae", "r3d", "swin"]):
        reshape_fn = lambda t: reshape_transform_3d(t, MODEL_NAME)
    else:
        reshape_fn = None
        
    cam = GradCAM(model=model_for_cam, target_layers=target_layers, reshape_transform=reshape_fn)
    targets = [ClassifierOutputTarget(1)]
    transform = get_transforms(MODEL_NAME)

    for domain, path in DOMAIN_PATHS.items():
        if not os.path.exists(path):
            continue
            
        cap = cv2.VideoCapture(path)
        frames = []
        while len(frames) < 25:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
        cap.release()

        if "r3d" in MODEL_NAME.lower():
            start_idx = max(0, (len(frames) - 12) // 2)
            sampled_frames = frames[start_idx : start_idx + 12]
            tensors = [transform(f) for f in sampled_frames] 
            input_tensor = torch.cat(tensors, dim=0).unsqueeze(0).to(DEVICE) 
            
        elif "videomae" in MODEL_NAME.lower():
            start_idx = max(0, (len(frames) - 16) // 2)
            sampled_frames = frames[start_idx : start_idx + 16]
            tensors = [transform(f) for f in sampled_frames]
            input_tensor = torch.cat(tensors, dim=0).unsqueeze(0).to(DEVICE) 
            
        else: 
            sampled_frames = frames[len(frames)//2 : len(frames)//2 + 1]
            input_tensor = transform(sampled_frames[0]).unsqueeze(0).to(DEVICE)

        try:
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
            
            mid_img = np.array(frames[len(frames)//2]).astype(np.float32) / 255.0
            mid_img = cv2.resize(mid_img, (224, 224) if "r3d" not in MODEL_NAME else (112, 112))
            
            vis = show_cam_on_image(mid_img, grayscale_cam, use_rgb=True)
            save_name = f"{MODEL_NAME.split('_')[0]}_gradcam_{domain.lower()}.png"
            cv2.imwrite(os.path.join(SAVE_DIR, save_name), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            
            print(f"   ✅ [{domain:<12}] 히트맵 추출 성공 ➔ {save_name}")
            
        except Exception as e:
            print(f"   ❌ [{domain:<12}] 연산 중 에러 발생: {e}")

    print("\n🎉 모든 도메인에 대한 Grad-CAM 추출이 완료되었습니다!")

if __name__ == "__main__":
    main()