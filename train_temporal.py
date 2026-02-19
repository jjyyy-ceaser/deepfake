import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchvision import models, transforms
import cv2
import os
import gc
import numpy as np
from tqdm import tqdm
import argparse
from sklearn.model_selection import StratifiedKFold

# 📂 [경로 고정]
BASE_DIR = "C:/Users/leejy/Desktop/test_experiment/dataset/final_datasets"
SEQUENCE_LENGTH = 16
IMG_SIZE = 224
BATCH_SIZE = 4 

# 📌 [최적 파라미터] R3D 복귀 완료
BEST_PARAMS = {
    'r3d': 1e-4,      # ✅ 복귀 완료
    'r2plus1d': 5e-5 
}

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
])

def clean_memory():
    """✨ VRAM 누수 방지 ✨"""
    gc.collect()
    torch.cuda.empty_cache()

class VideoSequenceDataset(Dataset):
    def __init__(self, samples, sequence_length=16, transform=None):
        self.samples = samples
        self.seq_len = sequence_length
        self.transform = transform

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start_frame = np.random.randint(0, total_frames - self.seq_len) if total_frames > self.seq_len else 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames = []
        for _ in range(self.seq_len):
            ret, frame = cap.read()
            if not ret: frame = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            else: frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.transform: frame = self.transform(frame)
            frames.append(frame)
        cap.release()
        frames = torch.stack(frames).permute(1, 0, 2, 3) 
        return frames, label

def get_model(model_name, device):
    if model_name == "r3d": 
        model = models.video.r3d_18(weights=models.video.R3D_18_Weights.KINETICS400_V1)
    elif model_name == "r2plus1d": 
        model = models.video.r2plus1d_18(weights=models.video.R2Plus1D_18_Weights.KINETICS400_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model.to(device)

def train_model(model_type, dataset_name, epochs=5): # 📌 Epoch 5
    clean_memory()
    folder_map = {"pure": "dataset_A_pure", "mixed": "dataset_B_mixed", "worst": "dataset_C_worst"}
    data_path = os.path.join(BASE_DIR, folder_map[dataset_name])
    
    print(f"🔥 [Temporal 5-Fold] 모델: {model_type} | 데이터: {dataset_name}")
    
    all_samples = []
    real_dir, fake_dir = os.path.join(data_path, "real"), os.path.join(data_path, "fake")
    if os.path.exists(real_dir): all_samples += [(os.path.join(real_dir, f), 0) for f in os.listdir(real_dir) if f.endswith('.mp4')]
    if os.path.exists(fake_dir): all_samples += [(os.path.join(fake_dir, f), 1) for f in os.listdir(fake_dir) if f.endswith('.mp4')]
    
    if len(all_samples) == 0:
        print(f"❌ 데이터 없음: {data_path}")
        return

    # 📌 5-Fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y = [s[1] for s in all_samples]

    for fold, (train_idx, val_idx) in enumerate(skf.split(all_samples, y)):
        print(f"  🔄 Fold {fold+1}/5 Start...")
        train_samples = [all_samples[i] for i in train_idx]
        
        train_ds = VideoSequenceDataset(train_samples, SEQUENCE_LENGTH, transform)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=2, persistent_workers=True)
        
        model = get_model(model_type, torch.device("cuda"))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=BEST_PARAMS.get(model_type, 1e-4))
        scaler = GradScaler()
        
        model.train()
        for epoch in range(epochs):
            loop = tqdm(train_loader, desc=f"Fold {fold+1} Ep {epoch+1}", leave=False)
            for inputs, labels in loop:
                inputs, labels = inputs.cuda(), labels.cuda()
                optimizer.zero_grad()
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                loop.set_postfix(loss=loss.item())
        
        torch.save(model.state_dict(), f"model_temporal_{model_type}_{dataset_name}_fold{fold+1}.pth")
        print(f"    ✅ Fold {fold+1} 저장 완료")
        del model, optimizer, scaler, train_loader
        clean_memory()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 🚨 기본값을 'others'로 설정 (A, C 실행용)
    parser.add_argument("--dataset", type=str, default="others")
    args = parser.parse_args()
    
    # ✅ R3D와 R2Plus1D 모두 실행
    target_models = ["r3d", "r2plus1d"]
    
    # ✅ Pure(A)와 Worst(C)만 순차 실행
    if args.dataset == "others":
        target_datasets = ["pure", "worst"]
    else:
        target_datasets = [args.dataset]
        
    for m in target_models:
        for d in target_datasets:
            train_model(m, d, epochs=5)