import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import torchvision.transforms.functional as TF
import cv2, os, gc, random, argparse
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold

# 📂 [경로 고정] 불균형 데이터셋 경로
TRAIN_DIR = r"C:\Users\leejy\Desktop\test_experiment\dataset\split_datasets\dataset_A\train"   
SEQUENCE_LENGTH = 16
IMG_SIZE = 224

# 🚀 [VRAM 안전선 최대 튜닝] 3D CNN은 VRAM을 많이 먹으므로 16~32 선에서 타협
BATCH_SIZE = 16 

base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
])

def clean_memory(): gc.collect(); torch.cuda.empty_cache()

class EarlyStopping:
    def __init__(self, patience=6, path='checkpoint.pth'):
        self.patience, self.counter, self.best_loss, self.early_stop, self.path = patience, 0, None, False, path
    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss, self.counter = val_loss, 0
            torch.save(model.state_dict(), self.path)
        else:
            self.counter += 1
            if self.counter >= self.patience: self.early_stop = True

class VideoSequenceDataset(Dataset):
    def __init__(self, samples, transform=None, is_train=True):
        self.samples, self.transform, self.is_train = samples, transform, is_train
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 16프레임 시퀀스 무작위 시작점 추출
        start = random.randint(0, total - SEQUENCE_LENGTH) if total > SEQUENCE_LENGTH else 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames, apply_hflip = [], (random.random() > 0.5) if self.is_train else False
        
        for _ in range(SEQUENCE_LENGTH):
            ret, frame = cap.read()
            if not ret: frame = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            else: frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = transforms.ToPILImage()(frame)
            pil_img = transforms.Resize((IMG_SIZE, IMG_SIZE))(pil_img)
            
            if apply_hflip: pil_img = TF.hflip(pil_img)
            
            # 📌 [약한 증강] 시간적 모델은 과도한 증강 시 프레임 연속성이 깨짐
            if self.is_train:
                pil_img = transforms.ColorJitter(brightness=0.2, contrast=0.2)(pil_img)
                
            frames.append(self.transform(pil_img))
        cap.release()
        return torch.stack(frames).permute(1, 0, 2, 3), label

def train_model(model_type, epochs=30):
    clean_memory()
    all_samples = []
    for sub, lab in [("real", 0), ("fake", 1)]:
        d = os.path.join(TRAIN_DIR, sub)
        if os.path.exists(d): all_samples += [(os.path.join(d, f), lab) for f in os.listdir(d) if f.endswith('.mp4')]
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # ⚖️ [가중치 설정] Real(270) vs Fake(135) 불균형 해소 페널티
    class_weights = torch.tensor([1.0, 2.0]).cuda()
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    for fold, (train_idx, val_idx) in enumerate(skf.split(all_samples, [s[1] for s in all_samples])):
        print(f"\n🔄 [Training] Fold {fold+1}/5 Start - {model_type.upper()}")
        
        # 🚀 [VRAM 풀가동 로더 튜닝] CPU 공급 가속
        train_loader = DataLoader(
            VideoSequenceDataset([all_samples[i] for i in train_idx], base_transform, True), 
            batch_size=BATCH_SIZE, shuffle=True, num_workers=4, 
            pin_memory=True, prefetch_factor=4, persistent_workers=True
        )
        val_loader = DataLoader(
            VideoSequenceDataset([all_samples[i] for i in val_idx], base_transform, False), 
            batch_size=BATCH_SIZE, shuffle=False, num_workers=4, 
            pin_memory=True, prefetch_factor=4, persistent_workers=True
        )
        
        model = (models.video.r3d_18(weights='KINETICS400_V1') if model_type == "r3d" else models.video.r2plus1d_18(weights='KINETICS400_V1'))
        model.fc = nn.Linear(model.fc.in_features, 2)
        model = model.cuda()

        for param in model.parameters(): param.requires_grad = False
        for param in model.layer4.parameters(): param.requires_grad = True
        for param in model.fc.parameters(): param.requires_grad = True

        if model_type == 'r3d': best_lr = 1e-04
        elif model_type == 'r2plus1d': best_lr = 5e-05
        else: best_lr = 1e-04

        # 🚀 [가중치 감쇠 투여]
        optimizer = optim.AdamW([
            {'params': model.layer4.parameters(), 'lr': best_lr * 0.1},
            {'params': model.fc.parameters(), 'lr': best_lr}
        ], weight_decay=1e-2)

        scaler, early_stopping = torch.amp.GradScaler('cuda'), EarlyStopping(patience=6, path=f"model_temporal_{model_type}_fold{fold+1}.pth")

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for inputs, labels in tqdm(train_loader, desc=f"Fold {fold+1} Ep {epoch+1}", leave=False):
                inputs, labels = inputs.cuda(), labels.cuda()
                optimizer.zero_grad()
                with torch.amp.autocast('cuda'): loss = criterion(model(inputs), labels)
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                train_loss += loss.item()
            
            model.eval(); val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.cuda(), labels.cuda()
                    with torch.amp.autocast('cuda'): val_loss += criterion(model(inputs), labels).item()
            
            v_loss = val_loss / len(val_loader)
            print(f"   - Ep {epoch+1}: Train {train_loss/len(train_loader):.4f} | Val {v_loss:.4f}")
            early_stopping(v_loss, model)
            if early_stopping.early_stop: 
                print("   ⚠️ 조기 종료 발동")
                break
        del model; clean_memory()

if __name__ == "__main__":
    for m in ["r3d", "r2plus1d"]: train_model(m)