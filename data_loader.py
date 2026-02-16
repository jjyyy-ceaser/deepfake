import torch
from torch.utils.data import Dataset
import cv2
import os
import numpy as np

# ==========================================
# ⚙️ 데이터 로더 설정 (검토 완료)
# ==========================================
IMG_SIZE = 224
SEQ_LEN = 16

class UnifiedDataset(Dataset):
    def __init__(self, samples, model_type, transform=None):
        self.samples = samples  # [(path, label), ...]
        self.model_type = model_type  # 'spatial' or 'temporal'
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        cap = cv2.VideoCapture(path)
        
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # [Type A] Spatial Models (1 Frame)
            if self.model_type == 'spatial':
                if total_frames > 0:
                    # 학습 시 랜덤, 평가 시 중앙 (이 로직은 호출부에서 제어 가능하지만 여기선 랜덤 기본)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, np.random.randint(0, total_frames))
                ret, frame = cap.read()
                
                if not ret: # 에러 방지: 검은 화면
                    frame = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                if self.transform:
                    frame = self.transform(frame)
                cap.release()
                return frame, label

            # [Type B] Temporal Models (16 Frames)
            else:
                frames = []
                # 16프레임 균등 추출 (Uniform Sampling)
                if total_frames >= SEQ_LEN:
                    indices = np.linspace(0, total_frames-1, SEQ_LEN, dtype=int)
                else:
                    # 영상이 짧으면 반복 패딩
                    indices = np.arange(SEQ_LEN) % (total_frames if total_frames > 0 else 1)

                for i in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ret, frame = cap.read()
                    if not ret:
                        frame = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
                    else:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    if self.transform:
                        frame = self.transform(frame)
                    frames.append(frame)
                
                cap.release()
                # (T, C, H, W) -> (C, T, H, W) : PyTorch Video 모델 표준
                return torch.stack(frames).permute(1, 0, 2, 3), label

        except Exception as e:
            cap.release()
            # 치명적 오류 시 0 텐서 반환하여 학습 중단 방지
            c, h, w = 3, IMG_SIZE, IMG_SIZE
            if self.model_type == 'spatial':
                return torch.zeros((c, h, w)), label
            else:
                return torch.zeros((c, SEQ_LEN, h, w)), label