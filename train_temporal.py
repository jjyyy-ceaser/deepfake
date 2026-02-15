import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import cv2
import os
import numpy as np
from tqdm import tqdm
import argparse

# ==========================================
# ⚙️ 설정
# ==========================================
BASE_DIR = "C:/Users/leejy/Desktop/test_experiment/dataset"
SEQUENCE_LENGTH = 16
IMG_SIZE = 112  # R3D, R(2+1)D 모델의 표준 입력 사이즈

# R3D 및 R(2+1)D를 위한 정규화 값 (Kinetics-400 데이터셋 기준)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
])

# ==========================================
# 📂 데이터셋 클래스
# ==========================================
class VideoSequenceDataset(Dataset):
    def __init__(self, data_dir, sequence_length=16, transform=None):
        self.data_dir = data_dir
        self.seq_len = sequence_length
        self.transform = transform
        self.samples = []
        
        real_dir = os.path.join(data_dir, "real")
        fake_dir = os.path.join(data_dir, "fake")
        
        # mp4 파일만 골라내어 샘플 리스트 생성
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
        
        # 랜덤한 시점에서 시퀀스 추출
        start_frame = 0
        if total_frames > self.seq_len:
            start_frame = np.random.randint(0, total_frames - self.seq_len)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames = []
        for _ in range(self.seq_len):
            ret, frame = cap.read()
            if not ret:
                # 프레임 부족 시 검은 화면으로 패딩
                frame = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
        cap.release()
        
        # VideoMAE와 달리 (C, T, H, W) 형식이 필요함
        frames = torch.stack(frames).permute(1, 0, 2, 3) 
        return frames, label

# ==========================================
# 🏗️ 모델 빌드 함수
# ==========================================
def get_model(model_name, device):
    print(f"🏗️ 모델 빌드 중: {model_name.upper()}...")
    if model_name == "r3d":
        model = models.video.r3d_18(weights=models.video.R3D_18_Weights.KINETICS400_V1)
    elif model_name == "r2plus1d":
        model = models.video.r2plus1d_18(weights=models.video.R2Plus1D_18_Weights.KINETICS400_V1)
    else:
        raise ValueError("지원하지 않는 모델입니다.")
        
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model.to(device)

# ==========================================
# 🔥 학습 핵심 함수
# ==========================================
def train_model(model_type, dataset_name, epochs=5):
    # 폴더 구조 매칭 수정 (2번, 2_train_mixed, 2_train_worst 반영)
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
    
    # RTX 4070 SUPER 12GB 기준으로 배치 사이즈 8이 안정적입니다.
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
    model = get_model(model_type, device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    model.train()
    for epoch in range(epochs):
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
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