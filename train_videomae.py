import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
import cv2
import os
import gc
import numpy as np
from tqdm import tqdm
import argparse
from transformers import VideoMAEForVideoClassification
from sklearn.model_selection import StratifiedKFold

# 📂 [경로 고정]
BASE_DIR = "C:/Users/leejy/Desktop/test_experiment/dataset/final_datasets"
SEQUENCE_LENGTH = 16
IMG_SIZE = 224
BATCH_SIZE = 4 

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def clean_memory():
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

def train_videomae(dataset_name, epochs=5): # 📌 Epoch 5
    clean_memory()
    folder_map = {"pure": "dataset_A_pure", "mixed": "dataset_B_mixed", "worst": "dataset_C_worst"}
    data_path = os.path.join(BASE_DIR, folder_map[dataset_name])
    
    print(f"🔥 [VideoMAE 5-Fold] 데이터: {dataset_name}")
    
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
        
        device = torch.device("cuda")
        model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base", num_labels=2, ignore_mismatched_sizes=True).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=5e-5)
        scaler = GradScaler()
        
        model.train()
        for epoch in range(epochs):
            loop = tqdm(train_loader, desc=f"Fold {fold+1} Ep {epoch+1}", leave=False)
            for inputs, labels in loop:
                inputs, labels = inputs.to(device), labels.to(device)
                inputs = inputs.permute(0, 2, 1, 3, 4) 
                
                optimizer.zero_grad()
                with autocast():
                    outputs = model(pixel_values=inputs, labels=labels)
                    loss = outputs.loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                loop.set_postfix(loss=loss.item())
        
        torch.save(model.state_dict(), f"model_temporal_videomae_{dataset_name}_fold{fold+1}.pth")
        print(f"    ✅ Fold {fold+1} 저장 완료")
        del model, optimizer, scaler, train_loader
        clean_memory()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 🚨 기본값을 'others'로 설정 (A, C 실행용)
    parser.add_argument("--dataset", type=str, default="others")
    args = parser.parse_args()
    
    # ✅ Pure(A)와 Worst(C)만 순차 실행
    if args.dataset == "others":
        datasets = ["pure", "worst"]
    else:
        datasets = [args.dataset]
        
    for d in datasets: 
        train_videomae(d, epochs=5)