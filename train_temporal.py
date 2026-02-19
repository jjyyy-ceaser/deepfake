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

BASE_DIR = "C:/Users/leejy/Desktop/test_experiment/dataset"
SEQUENCE_LENGTH = 16
IMG_SIZE = 224
BATCH_SIZE = 4 

# 📌 [최적 파라미터]
BEST_PARAMS = {
    'r3d': 1e-4,       # 이번엔 안 쓰지만 기록용
    'r2plus1d': 5e-5   # 기존 1e-4 -> 최적값 5e-5 변경
}

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
])

def clean_memory():
    """✨ 메모리 세탁 기능 ✨"""
    gc.collect()
    torch.cuda.empty_cache()

class VideoSequenceDataset(Dataset):
    def __init__(self, data_dir, sequence_length=16, transform=None):
        self.data_dir = data_dir
        self.seq_len = sequence_length
        self.transform = transform
        self.samples = []
        real_dir, fake_dir = os.path.join(data_dir, "real"), os.path.join(data_dir, "fake")
        if os.path.exists(real_dir):
            self.samples += [(os.path.join(real_dir, f), 0) for f in os.listdir(real_dir) if f.endswith('.mp4')]
        if os.path.exists(fake_dir):
            self.samples += [(os.path.join(fake_dir, f), 1) for f in os.listdir(fake_dir) if f.endswith('.mp4')]

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
    print(f"🏗️ 모델 빌드 중: {model_name.upper()}...")
    if model_name == "r3d":
        model = models.video.r3d_18(weights=models.video.R3D_18_Weights.KINETICS400_V1)
    elif model_name == "r2plus1d":
        model = models.video.r2plus1d_18(weights=models.video.R2Plus1D_18_Weights.KINETICS400_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model.to(device)

def train_model(model_type, dataset_name, epochs=5):
    clean_memory()
    folder_map = {"pure": os.path.join("2_exp_train_pure", "train"), "mixed": "2_train_mixed", "worst": "2_train_worst"}
    data_path = os.path.join(BASE_DIR, folder_map[dataset_name])
    
    dataset = VideoSequenceDataset(data_path, SEQUENCE_LENGTH, transform)
    if len(dataset) == 0: return
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                            num_workers=8, pin_memory=True, 
                            prefetch_factor=2, persistent_workers=True)
    
    model = get_model(model_type, torch.device("cuda"))
    criterion = nn.CrossEntropyLoss()
    
    # 📌 최적 LR 적용
    lr = BEST_PARAMS.get(model_type, 1e-4)
    print(f"⚙️ 적용된 Learning Rate: {lr}")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    scaler = GradScaler()
    
    model.train()
    for epoch in range(epochs):
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
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
        clean_memory()
            
    torch.save(model.state_dict(), f"model_temporal_{model_type}_{dataset_name}.pth")
    print(f"✅ 저장 완료: model_temporal_{model_type}_{dataset_name}.pth")
    del model, optimizer, scaler
    clean_memory()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all")
    # 🚨 [수정됨] 기본값 'mixed'
    parser.add_argument("--dataset", type=str, default="mixed")
    args = parser.parse_args()
    
    target_models = ["r2plus1d"] if args.model == "all" else [args.model]
    # 'all'이어도 mixed(Dataset B)만 돌리도록 강제하거나, 옵션으로 조정
    # 여기선 안전하게 mixed만 리스트에 넣음
    target_datasets = ["mixed"] if args.dataset == "mixed" else [args.dataset]
    
    for m in target_models:
        for d in target_datasets:
            train_model(m, d, epochs=5)