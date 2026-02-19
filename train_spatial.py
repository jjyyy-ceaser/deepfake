import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
import cv2
import os
import random
import gc
import numpy as np
from tqdm import tqdm
import argparse
import timm

BASE_DIR = "C:/Users/leejy/Desktop/test_experiment/dataset"
IMG_SIZE = 224

# 📌 [최적 파라미터]
BEST_PARAMS = {
    'xception': 5e-5,
    'convnext': 1e-4,
    'swin': 5e-5
}

def clean_memory():
    gc.collect()
    torch.cuda.empty_cache()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class VideoFrameDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
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
        frame = None
        if total_frames > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, random.randint(0, total_frames - 1))
            ret, frame = cap.read()
        cap.release()
        
        if frame is None: frame = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        else: frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if self.transform: frame = self.transform(frame)
        return frame, label

def get_model(model_name, device):
    print(f"🏗️ 모델 빌드 중: {model_name.upper()}...")
    if model_name == "xception": model = timm.create_model('xception', pretrained=True, num_classes=2)
    elif model_name == "convnext": model = timm.create_model('convnext_tiny', pretrained=True, num_classes=2)
    elif model_name == "swin": model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=2)
    return model.to(device)

def train_model(model_type, dataset_name, epochs=5):
    clean_memory()
    folder_map = {"pure": os.path.join("2_exp_train_pure", "train"), "mixed": "2_train_mixed", "worst": "2_train_worst"}
    data_path = os.path.join(BASE_DIR, folder_map[dataset_name])
    
    print(f"🔥 [Spatial 학습 시작] 모델: {model_type} | 데이터셋: {dataset_name} (Dataset B)")
    dataset = VideoFrameDataset(data_path, transform=transform)
    if len(dataset) == 0: return
        
    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=8, pin_memory=True, 
                            prefetch_factor=2, persistent_workers=True)
    
    model = get_model(model_type, torch.device("cuda"))
    criterion = nn.CrossEntropyLoss()
    lr = BEST_PARAMS.get(model_type, 1e-4)
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
            
    torch.save(model.state_dict(), f"model_spatial_{model_type}_{dataset_name}.pth")
    print(f"✅ 저장 완료: model_spatial_{model_type}_{dataset_name}.pth")
    del model, optimizer, scaler
    clean_memory()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all")
    # 🚨 [수정됨] 기본값을 'mixed' (Dataset B)로 변경하여 이것만 실행
    parser.add_argument("--dataset", type=str, default="mixed") 
    args = parser.parse_args()
    
    target_models = ["xception", "convnext", "swin"] if args.model == "all" else [args.model]
    target_datasets = ["mixed"] if args.dataset == "mixed" else [args.dataset] 
    
    for m in target_models:
        for d in target_datasets:
            train_model(m, d, epochs=5)