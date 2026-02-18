import torch
import torch.nn as nn
from torch.utils.data import Dataset
import cv2
import numpy as np
from PIL import Image
import timm
import torchvision.models as models
from transformers import VideoMAEForVideoClassification
import gc
import os

# OpenCV 윈도우 충돌 방지
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
        current_idx = idx
        retry_count = 0
        while retry_count < 5:
            path = self.file_paths[current_idx]
            label = self.labels[current_idx]
            try:
                is_video = path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
                if self.model_type == 'spatial':
                    img = self.extract_single_frame(path) if is_video else self.load_raw_image(path)
                    if self.transform: img = self.transform(img)
                    return img, label
                elif self.model_type == 'temporal':
                    frames = self.load_video_frames(path)
                    if self.transform:
                        # [핵심 수정] VideoMAE는 (Frames, C, H, W)를 기대하므로 permute를 제거하거나 조정
                        frames = torch.stack([self.transform(f) for f in frames]) 
                        # 일반 3D CNN(R3D)은 (C, T, H, W)를 원하므로, 2_train_system.py에서 모델별로 처리하는 게 안전합니다.
                        # 여기서는 일단 (T, C, H, W) 기본 형태로 보냅니다.
                    return frames, label
            except Exception as e:
                # 에러 로그를 남겨야 나중에 확인 가능합니다.
                print(f"⚠️ Error loading {path}: {e}")
                current_idx = np.random.randint(0, len(self.file_paths))
                retry_count += 1
            finally:
                gc.collect()
        raise RuntimeError(f"Data loading failed after 5 retries at index {idx}")

    def load_raw_image(self, path):
        img = cv2.imread(path)
        if img is None: raise ValueError(f"Image Load Failed: {path}")
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def extract_single_frame(self, path):
        cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0: cap.release(); raise ValueError(f"Empty Video: {path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
        ret, frame = cap.read()
        cap.release()
        if not ret: raise ValueError(f"Read Failed: {path}")
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def load_video_frames(self, path):
        cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0: cap.release(); raise ValueError(f"Empty Video: {path}")
        indices = np.linspace(0, total - 1, self.num_frames, dtype=int)
        frames = []
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i); ret, frame = cap.read()
            if ret: frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        cap.release()
        if not frames: raise ValueError(f"No frames extracted from {path}")
        while len(frames) < self.num_frames: frames.append(frames[-1])
        return frames

def get_model(model_name, device, num_classes=2):
    name = model_name.lower()
    if 'xception' in name: model = timm.create_model('xception', pretrained=True, num_classes=num_classes)
    elif 'convnext' in name: model = timm.create_model('convnext_tiny', pretrained=True, num_classes=num_classes)
    elif 'swin' in name: model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=num_classes)
    elif 'r3d' in name: 
        model = models.video.r3d_18(weights=models.video.R3D_18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif 'r2plus1d' in name:
        model = models.video.r2plus1d_18(weights=models.video.R2Plus1D_18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif 'videomae' in name:
        model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics", num_labels=num_classes, ignore_mismatched_sizes=True)
    return model.to(device)