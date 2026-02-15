import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast  # AMP 추가
from torchvision import models, transforms
import cv2
import os
import numpy as np
from tqdm import tqdm
import argparse

# ==========================================
# ⚙️ 설정 (수정됨: 112 -> 224 통일)
# ==========================================
BASE_DIR = "C:/Users/leejy/Desktop/test_experiment/dataset"
SEQUENCE_LENGTH = 16
IMG_SIZE = 224  # 🚨 핵심 수정: 112 -> 224로 상향 (Spatial과 통일)
BATCH_SIZE = 4  # 해상도가 커졌으므로 배치 사이즈 조절 (VRAM 12GB~24GB 기준 안정값)

# R3D/R2+1D 입력용 정규화
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
])

class VideoSequenceDataset(Dataset):
    def __init__(self, data_dir, sequence_length=16, transform=None):
        self.data_dir = data_dir
        self.seq_len = sequence_length
        self.transform = transform
        self.samples = []
        
        real_dir = os.path.join(data_dir, "real")
        fake_dir = os.path.join(data_dir, "fake")
        
        if os.path.exists(real_dir):
            for f in os.listdir(real_dir):
                if f.lower().endswith('.mp4'): self.samples.append((os.path.join(real_dir, f), 0))
        if os.path.exists(fake_dir):
            for f in os.listdir(fake_dir):
                if f.lower().endswith('.mp4'): self.samples.append((os.path.join(fake_dir, f), 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        start_frame = 0
        if total_frames > self.seq_len:
            start_frame = np.random.randint(0, total_frames - self.seq_len)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames = []
        for _ in range(self.seq_len):
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
        cap.release()
        
        # (C, T, H, W) 형식
        frames = torch.stack(frames).permute(1, 0, 2, 3) 
        return frames, label

def get_model(model_name, device):
    print(f"🏗️ 모델 빌드 중: {model_name.upper()} (Input: 224x224)...")
    if model_name == "r3d":
        model = models.video.r3d_18(weights=models.video.R3D_18_Weights.KINETICS400_V1)
    elif model_name == "r2plus1d":
        model = models.video.r2plus1d_18(weights=models.video.R2Plus1D_18_Weights.KINETICS400_V1)
    else:
        raise ValueError("지원하지 않는 모델입니다.")
        
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model.to(device)

def train_model(model_type, dataset_name, epochs=5):
    folder_map = {
        "pure": os.path.join("2_exp_train_pure", "train"),
        "mixed": "2_train_mixed",
        "worst": "2_train_worst"
    }
    target_folder = folder_map[dataset_name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n==================================================")
    print(f"🔥 [Temporal 학습] 모델: {model_type.upper()} | 데이터: {dataset_name.upper()}")
    print(f"==================================================")
    
    data_path = os.path.join(BASE_DIR, target_folder)
    dataset = VideoSequenceDataset(data_path, SEQUENCE_LENGTH, transform)
    
    if len(dataset) == 0:
        print(f"❌ 데이터 없음: {data_path}")
        return
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    model = get_model(model_type, device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scaler = GradScaler() # AMP Scaler
    
    model.train()
    for epoch in range(epochs):
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # AMP 적용 (메모리 절약 및 속도 향상)
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            loop.set_postfix(loss=loss.item())
            
    save_name = f"model_temporal_{model_type}_{dataset_name}.pth"
    torch.save(model.state_dict(), save_name)
    print(f"✅ 저장 완료: {save_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all", help="r3d / r2plus1d / all")
    parser.add_argument("--dataset", type=str, default="all", help="pure / mixed / worst / all")
    args = parser.parse_args()
    
    target_models = ["r3d", "r2plus1d"] if args.model == "all" else [args.model]
    target_datasets = ["pure", "mixed", "worst"] if args.dataset == "all" else [args.dataset]
    
    for m in target_models:
        for d in target_datasets:
            train_model(m, d, epochs=5)