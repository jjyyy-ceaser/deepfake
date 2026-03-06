import os
import cv2
import numpy as np
import glob
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from decord import VideoReader, cpu

# [Rev.13] 최종 통합 로더 (검은 화면 방지 + ID 매칭 보정)
MODEL_SPECS = {
    "r3d": {"size": (112, 112), "mean": [0.432, 0.394, 0.376], "std": [0.228, 0.221, 0.216], "frames": 16},
    "default": {"size": (224, 224), "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "frames": 16}
}

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def get_face_centric_crop(frame_np, ratio=0.9, margin=0.3):
    h, w, _ = frame_np.shape
    gray = cv2.cvtColor(frame_np, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(100, 100))
    
    if len(faces) > 0:
        fx, fy, fw, fh = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)[0]
        m_w, m_h = int(fw * margin), int(fh * margin)
        x1, y1 = max(0, fx - m_w), max(0, fy - m_h)
        x2, y2 = min(w, fx + fw + m_w), min(h, fy + fh + m_h)
        side = min(x2 - x1, y2 - y1)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        return max(0, cx - side//2), max(0, cy - side//2), min(w, cx + side//2), min(h, cy + side//2)
    else:
        side = int(min(h, w) * ratio)
        x1, y1 = (w - side) // 2, (h - side) // 2
        return x1, y1, x1 + side, y1 + side

class DeepfakeDataset(Dataset):
    def __init__(self, file_paths, labels, model_name, sampling='uniform'):
        self.file_paths = file_paths
        self.labels = labels
        self.model_name = model_name.lower()
        self.sampling = sampling
        self.spec = MODEL_SPECS["r3d"] if "r3d" in self.model_name else MODEL_SPECS["default"]
        self.transform = transforms.Compose([
            transforms.Resize(self.spec['size']),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.spec['mean'], std=self.spec['std'])
        ])

    def __len__(self): return len(self.file_paths)

    def _read_frames_decord(self, path):
        try:
            vr = VideoReader(path, ctx=cpu(0))
            total = len(vr)
            num_req = self.spec['frames']
            
            # ⚡ [검은 화면 방지] 영상이 16프레임보다 짧으면 마지막 프레임 반복(Padding)
            if total < num_req:
                indices = list(range(total)) + [total - 1] * (num_req - total)
            else:
                indices = np.linspace(0, total - 1, num_req, dtype=int).tolist()
            
            return vr.get_batch(indices).asnumpy()
        except: return None

    def __getitem__(self, idx):
        path, label = self.file_paths[idx], self.labels[idx]
        frames = self._read_frames_decord(path)
        if frames is None:
            sz = self.spec['size']
            return (torch.zeros(3, *sz) if any(m in self.model_name for m in ["xception", "swin"]) 
                    else torch.zeros(self.spec['frames'], 3, *sz)), label

        ref_idx = len(frames)//2 if any(m in self.model_name for m in ["xception", "swin"]) else 0
        x1, y1, x2, y2 = get_face_centric_crop(cv2.cvtColor(frames[ref_idx], cv2.COLOR_RGB2BGR))
        
        if any(m in self.model_name for m in ["xception", "swin"]):
            img = Image.fromarray(frames[len(frames)//2][y1:y2, x1:x2])
            return self.transform(img), label
            
        processed = [self.transform(Image.fromarray(f[y1:y2, x1:x2])) for f in frames]
        return torch.stack(processed), label

def prepare_dataset(base_dir):
    real_paths = glob.glob(os.path.join(base_dir, "real", "*.mp4"))
    fake_paths = glob.glob(os.path.join(base_dir, "fake", "*.mp4"))
    
    all_data = []
    for p in real_paths:
        # '000.mp4' -> ID: '000'
        all_data.append({'path': p, 'label': 0, 'id': os.path.basename(p).split('.')[0]})
    for p in fake_paths:
        # 'svd_000.mp4' -> ID: '000' (매칭 성공!)
        fid = os.path.basename(p).replace('svd_', '').split('.')[0]
        all_data.append({'path': p, 'label': 1, 'id': fid})
        
    random.shuffle(all_data)
    return [d['path'] for d in all_data], [d['label'] for d in all_data], [d['id'] for d in all_data]