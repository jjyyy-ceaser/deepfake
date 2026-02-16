import torch
import torch.nn as nn
from torch.utils.data import Dataset
import cv2
import os
import timm
from torchvision import models, transforms
from transformers import VideoMAEForVideoClassification

class DeepfakeDataset(Dataset):
    def __init__(self, file_paths, labels, model_type, transform=None):
        self.files = file_paths
        self.labels = labels
        self.model_type = model_type
        self.transform = transform

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        label = self.labels[idx]
        
        cap = cv2.VideoCapture(path)
        frames = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # 16프레임 균등 추출
        indices = torch.linspace(0, frame_count-1, 16).long()
        
        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret: break
            if i in indices:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.transform: frame = self.transform(frame)
                frames.append(frame)
        cap.release()
        
        if len(frames) == 0: return torch.zeros(3, 16, 224, 224), label # 예외처리
        
        # Stack: (T, C, H, W) -> (C, T, H, W)
        video = torch.stack(frames).permute(1, 0, 2, 3) 
        
        # Spatial 모델이면 (C, H, W) 평균 사용 or 첫 프레임 등 (여기선 Average Pooling)
        if self.model_type == 'spatial':
            video = video.mean(dim=1) # Temporal 차원 압축
            
        return video, label

def get_model(model_name, device):
    if "videomae" in model_name:
        model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base", num_labels=2, ignore_mismatched_sizes=True)
    elif "r3d" in model_name:
        model = models.video.r3d_18(weights='KINETICS400_V1')
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif "r2plus1d" in model_name:
        model = models.video.r2plus1d_18(weights='KINETICS400_V1')
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif "swin" in model_name:
        model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=2)
    elif "convnext" in model_name:
        model = timm.create_model('convnext_tiny', pretrained=True, num_classes=2)
    elif "xception" in model_name:
        model = timm.create_model('xception', pretrained=True, num_classes=2)
    else:
        raise ValueError("Unknown model")
    
    return model.to(device)