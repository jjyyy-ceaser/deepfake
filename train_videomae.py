import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
import torchvision.transforms.functional as TF
import cv2
import os
import gc
import random
import numpy as np
from tqdm import tqdm
from transformers import VideoMAEForVideoClassification
from sklearn.model_selection import StratifiedKFold

# 📂 [경로 고정 - Dataset A Train 135쌍]
TRAIN_DIR = r"C:\Users\leejy\Desktop\test_experiment\dataset\split_datasets\dataset_A\train"
SEQUENCE_LENGTH = 16
IMG_SIZE = 224
BATCH_SIZE = 4 

# 📌 [최적 파라미터]
BEST_LR = 1e-4

base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def clean_memory():
    gc.collect()
    torch.cuda.empty_cache()

class EarlyStopping:
    def __init__(self, patience=3, path='checkpoint.pth'):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.path = path

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)

class VideoSequenceDataset(Dataset):
    def __init__(self, samples, sequence_length=16, transform=None, is_train=True):
        self.samples = samples
        self.seq_len = sequence_length
        self.transform = transform
        self.is_train = is_train

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start_frame = np.random.randint(0, total_frames - self.seq_len) if total_frames > self.seq_len else 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frames = []
        
        if self.is_train:
            apply_hflip = random.random() > 0.5
            brightness_factor = random.uniform(0.8, 1.2)
            contrast_factor = random.uniform(0.8, 1.2)
        else:
            apply_hflip = False
            brightness_factor = 1.0
            contrast_factor = 1.0
        
        for _ in range(self.seq_len):
            ret, cap_frame = cap.read()
            if not ret: 
                cap_frame = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            else: 
                cap_frame = cv2.cvtColor(cap_frame, cv2.COLOR_BGR2RGB)
                
            pil_img = transforms.ToPILImage()(cap_frame)
            pil_img = transforms.Resize((IMG_SIZE, IMG_SIZE))(pil_img)
            
            if apply_hflip:
                pil_img = TF.hflip(pil_img)
            pil_img = TF.adjust_brightness(pil_img, brightness_factor)
            pil_img = TF.adjust_contrast(pil_img, contrast_factor)
            
            if self.transform: 
                frame_tensor = self.transform(pil_img)
            else:
                frame_tensor = transforms.ToTensor()(pil_img)
                
            frames.append(frame_tensor)
            
        cap.release()
        frames = torch.stack(frames).permute(1, 0, 2, 3) 
        return frames, label

def train_videomae(epochs=15):
    clean_memory()
    print(f"🔥 [VideoMAE 5-Fold] 데이터: Dataset A Train")
    
    all_samples = []
    real_dir, fake_dir = os.path.join(TRAIN_DIR, "real"), os.path.join(TRAIN_DIR, "fake")
    if os.path.exists(real_dir): all_samples += [(os.path.join(real_dir, f), 0) for f in os.listdir(real_dir) if f.endswith('.mp4')]
    if os.path.exists(fake_dir): all_samples += [(os.path.join(fake_dir, f), 1) for f in os.listdir(fake_dir) if f.endswith('.mp4')]
    
    if len(all_samples) == 0:
        print(f"❌ 데이터 없음: {TRAIN_DIR}")
        return

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y = [s[1] for s in all_samples]

    for fold, (train_idx, val_idx) in enumerate(skf.split(all_samples, y)):
        print(f"\n🔄 Fold {fold+1}/5 Start...")
        train_samples = [all_samples[i] for i in train_idx]
        val_samples = [all_samples[i] for i in val_idx]
        
        train_ds = VideoSequenceDataset(train_samples, SEQUENCE_LENGTH, base_transform, is_train=True)
        val_ds = VideoSequenceDataset(val_samples, SEQUENCE_LENGTH, base_transform, is_train=False)
        
        # 📌 고성능 데이터 로더 세팅 이식
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=2, persistent_workers=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True, prefetch_factor=2, persistent_workers=True)
        
        device = torch.device("cuda")
        model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base", num_labels=2, ignore_mismatched_sizes=True).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=BEST_LR)
        scaler = GradScaler()
        
        save_path = f"model_temporal_videomae_fold{fold+1}.pth"
        early_stopping = EarlyStopping(patience=3, path=save_path)
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            loop = tqdm(train_loader, desc=f"Fold {fold+1} Ep {epoch+1} Train", leave=False)
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
                train_loss += loss.item()
                loop.set_postfix(loss=loss.item())
                
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    inputs = inputs.permute(0, 2, 1, 3, 4) 
                    with autocast():
                        outputs = model(pixel_values=inputs, labels=labels)
                        loss = outputs.loss
                    val_loss += loss.item()
            
            t_loss = train_loss / len(train_loader)
            v_loss = val_loss / len(val_loader)
            print(f"   - Epoch {epoch+1}: Train Loss: {t_loss:.4f}, Val Loss: {v_loss:.4f}")
            
            early_stopping(v_loss, model)
            if early_stopping.early_stop:
                print("   ⏹️ Validation Loss 개선 없음. 조기 종료합니다.")
                break
        
        print(f"    ✅ Fold {fold+1} 완료 (Best Model 저장됨: {save_path})")
        del model, optimizer, scaler, train_loader, val_loader, train_ds, val_ds
        clean_memory()

if __name__ == "__main__":
    train_videomae(epochs=15)
    