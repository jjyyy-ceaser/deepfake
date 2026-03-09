import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class DeepfakeDataset(Dataset):
    # 🔧 [수정 1] __init__ 인자에 model_name 추가 (기본값 None)
    def __init__(self, file_paths=None, labels=None, root_dir=None, 
                 model_name='xception', mode='train', transform=None, window_size=16):
        
        # model_type 자동 판별 (spatial / temporal)
        self.model_name = model_name.lower()
        if any(x in self.model_name for x in ['xception', 'swin']):
            self.model_type = 'spatial'
        else:
            self.model_type = 'temporal'

        self.mode = mode
        self.transform = transform
        self.window_size = window_size
        
        # [Case A] Grid Search용
        if file_paths is not None and labels is not None:
            self.video_paths = file_paths
            self.labels = labels
        # [Case B] 일반 학습용
        elif root_dir is not None:
            real_folder = os.path.join(root_dir, 'real')
            fake_folder = os.path.join(root_dir, 'fake')
            real_paths = sorted([os.path.join(real_folder, f) for f in os.listdir(real_folder) if f.endswith('.mp4')])
            fake_paths = sorted([os.path.join(fake_folder, f) for f in os.listdir(fake_folder) if f.endswith('.mp4')])
            self.video_paths = real_paths + fake_paths
            self.labels = [0] * len(real_paths) + [1] * len(fake_paths)
        else:
            raise ValueError("❌ 오류: file_paths/labels 또는 root_dir 중 하나는 필수입니다.")

        self.samples_per_clip = 3 if mode == 'train' else 1

    def __len__(self):
        return len(self.video_paths) * self.samples_per_clip

    def __getitem__(self, index):
        video_idx = index // self.samples_per_clip
        sample_view_idx = index % self.samples_per_clip
        
        path = self.video_paths[video_idx]
        label = self.labels[video_idx]
        
        frames = self._load_frames(path)
        
        if self.model_type == 'spatial':
            data = self._sample_spatial(frames, sample_view_idx)
        else:
            data = self._sample_temporal(frames, sample_view_idx)
            
        return data, torch.tensor(label, dtype=torch.long)

    def _load_frames(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        cap.release()
        
        # 🔧 [설계안 6절 일치] Boomerang 패딩 적용 (마지막 프레임 반복보다 자연스러운 시계열 형성)
        if 0 < len(frames) < 25:
            pad_source = frames[-2::-1] if len(frames) > 1 else frames
            while len(frames) < 25:
                needed = 25 - len(frames)
                frames.extend(pad_source[:needed])
        elif len(frames) == 0:
            frames = [Image.new('RGB', (224, 224))] * 25
        return frames[:25]

    def _sample_spatial(self, frames, view_idx):
        if self.mode == 'train':
            indices = [0, 12, 24]
            frame = frames[indices[view_idx]]
        else:
            frame = frames[12]
        if self.transform:
            frame = self.transform(frame)
        return frame

    def _sample_temporal(self, frames, view_idx):
        total_frames = len(frames)
        if self.mode == 'train':
            start_indices = np.linspace(0, total_frames - self.window_size, 3).astype(int)
            start_idx = start_indices[view_idx]
        else:
            start_idx = (total_frames - self.window_size) // 2
            
        sequence = frames[start_idx : start_idx + self.window_size]
        
        if self.transform:
            # 기본: [T, C, H, W] (torch.stack 결과)
            sequence = torch.stack([self.transform(img) for img in sequence])
            
            # 🔧 [수정 2] 모델별 차원 순서 결정 (핵심 수정!)
            # R3D는 (C, T, H, W)를 원함
            if 'r3d' in self.model_name:
                sequence = sequence.permute(1, 0, 2, 3) 
            # VideoMAE와 Hybrid는 (T, C, H, W)를 원함 -> permute 안 함 (그대로 둠)
            
        return sequence