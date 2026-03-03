import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

IMG_SIZE = 224
SEQ_LEN = 16

class UnifiedDataset(Dataset):
    def __init__(self, samples, model_type, transform=None):
        self.samples = samples
        self.model_type = model_type
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        cap = cv2.VideoCapture(path)
        
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # [Type A] Spatial Models (Center Frame)
            if self.model_type == 'spatial':
                # 🚀 [품질 개선] 첫 프레임 대신 영상의 정확한 중앙 지점으로 이동
                # 페이드 인(Fade-in)이나 블랙 스크린을 피하고 피사체가 있는 구간을 타겟팅합니다.
                middle_idx = total_frames // 2 if total_frames > 0 else 0
                cap.set(cv2.CAP_PROP_POS_FRAMES, middle_idx)
                
                ret, frame = cap.read()
                if not ret: 
                    # 중앙 프레임 읽기 실패 시 첫 프레임으로 재시도
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                
                if not ret:
                    frame = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                if self.transform:
                    frame = self.transform(frame)
                cap.release()
                return frame, label

            # [Type B] Temporal & VideoMAE (전체 구간 균등 추출)
            else:
                frames = []
                indices = np.linspace(0, total_frames - 1, SEQ_LEN, dtype=int) if total_frames > 0 else [0]*SEQ_LEN
                
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
                
                stacked_frames = torch.stack(frames)
                if self.model_type == 'videomae':
                    return stacked_frames, label
                else:
                    return stacked_frames.permute(1, 0, 2, 3), label

        except Exception:
            if cap: cap.release()
            return torch.zeros((3, IMG_SIZE, IMG_SIZE)) if self.model_type == 'spatial' else torch.zeros((3, SEQ_LEN, IMG_SIZE, IMG_SIZE)), label