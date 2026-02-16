import torch
import torch.nn as nn
from torch.utils.data import Dataset
import cv2
import numpy as np
from PIL import Image
import timm
import torchvision.models as models
from transformers import VideoMAEForVideoClassification
import gc # [핵심] 가비지 컬렉션 모듈 추가
import os
# OpenCV가 멀티프로세싱과 충돌하는 것을 방지
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

class DeepfakeDataset(Dataset):
    def __init__(self, file_paths, labels, model_type, transform=None, num_frames=16):
        self.file_paths = file_paths
        self.labels = labels
        self.model_type = model_type
        self.transform = transform
        self.num_frames = num_frames

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # 무한 루프로 감싸서 에러가 나도 대체 데이터를 찾을 때까지 시도
        current_idx = idx
        while True:
            path = self.file_paths[current_idx]
            label = self.labels[current_idx]
            cap = None # 변수 초기화
            
            try:
                # 확장자 체크 (대소문자 무시)
                is_video = path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))

                if self.model_type == 'spatial':
                    if is_video:
                        # 비디오면 중간 프레임 1개만 가져와서 이미지로 사용
                        img = self.extract_single_frame(path)
                    else:
                        img = self.load_raw_image(path)
                    return img, label
                
                elif self.model_type == 'temporal':
                    return self.load_video_frames(path), label
                    
            except Exception as e:
                # 에러 로그 출력 (너무 많이 출력되면 지저분하므로 간략하게)
                # print(f"⚠️ Skip error data ({current_idx}): {e}")
                
                # 랜덤하게 다른 인덱스로 교체하여 재시도
                current_idx = np.random.randint(0, len(self.file_paths))
            
            finally:
                # [핵심] 사용한 비디오 자원 강제 해제
                if cap is not None:
                    cap.release()
                # 가비지 컬렉션으로 메모리 찌꺼기 즉시 청소
                gc.collect()

    def load_raw_image(self, path):
        img = cv2.imread(path)
        if img is None: raise ValueError("Image Load Failed")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        if self.transform: return self.transform(img_pil)
        return img_pil

    def extract_single_frame(self, path):
        """비디오의 중간 프레임 하나를 이미지로 반환"""
        # CAP_FFMPEG 옵션으로 안정성 확보
        cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            cap.release()
            raise ValueError("Empty Video")
        
        # 중간 지점 프레임 읽기
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
        ret, frame = cap.read()
        cap.release() # [중요] 읽자마자 즉시 해제
        
        if not ret: raise ValueError("Frame Extraction Failed")
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame)
        if self.transform:
            return self.transform(img_pil)
        return img_pil

    def load_video_frames(self, path):
        cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            cap.release()
            raise ValueError("Empty Video")

        frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        frames = []
        
        # 성능 최적화를 위해 필요한 프레임만 탐색해서 읽기
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret: break
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 여기서 바로 transform 적용해서 메모리 줄이기 (PIL 변환 후)
            img_pil = Image.fromarray(frame)
            if self.transform:
                frame_tensor = self.transform(img_pil)
            else:
                frame_tensor = torch.from_numpy(np.array(img_pil)).permute(2,0,1).float()/255.0
            
            frames.append(frame_tensor)
            
        cap.release() # [중요] 즉시 해제

        if len(frames) == 0:
            raise ValueError("No frames extracted")
            
        # 부족한 프레임 채우기 (Padding)
        while len(frames) < self.num_frames:
            frames.append(frames[-1])
            
        return torch.stack(frames).permute(1, 0, 2, 3)

def get_model(model_name, device, num_classes=2):
    model_name = model_name.lower()
    try:
        if 'xception' in model_name:
            model = timm.create_model('xception', pretrained=True, num_classes=num_classes)
        elif 'convnext' in model_name:
            model = timm.create_model('convnext_tiny', pretrained=True, num_classes=num_classes)
        elif 'swin' in model_name:
            model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=num_classes)
        elif 'r3d' in model_name:
            model = models.video.r3d_18(weights=models.video.R3D_18_Weights.DEFAULT)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif 'r2plus1d' in model_name:
            model = models.video.r2plus1d_18(weights=models.video.R2Plus1D_18_Weights.DEFAULT)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif 'videomae' in model_name:
            model = VideoMAEForVideoClassification.from_pretrained(
                "MCG-NJU/videomae-base-finetuned-kinetics",
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")
    except Exception as e:
        print(f"❌ Model Load Error ({model_name}): {e}")
        raise e 

    return model.to(device)