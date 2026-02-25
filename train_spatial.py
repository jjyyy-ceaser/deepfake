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
from sklearn.model_selection import StratifiedKFold

# 📂 [경로 고정 - Dataset A Train 135쌍]
TRAIN_DIR = r"C:\Users\leejy\Desktop\test_experiment\dataset\split_datasets\dataset_A\train"

# 📌 [최적 파라미터 - 그리드 서치 결과 반영]
BEST_PARAMS = {
    'xception': 5e-5,
    'convnext': 1e-4,
    'swin': 5e-5
}

def clean_memory():
    gc.collect()
    torch.cuda.empty_cache()

# 📌 [1] 얼리 스토핑 클래스 정의
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

# 📌 [2] 학습용(증강 O)과 검증용(증강 X) Transform 분리
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class VideoFrameDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

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
        if frame is None: frame = np.zeros((224, 224, 3), dtype=np.uint8)
        else: frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.transform: frame = self.transform(frame)
        return frame, label

def get_model(model_name, device):
    if model_name == "xception": model = timm.create_model('xception', pretrained=True, num_classes=2)
    elif model_name == "convnext": model = timm.create_model('convnext_tiny', pretrained=True, num_classes=2)
    elif model_name == "swin": model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=2)
    return model.to(device)

def train_model(model_type, epochs=15):
    clean_memory()
    print(f"🔥 [Spatial 5-Fold] 모델: {model_type} | 학습 데이터: Dataset A Train")

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
        
        train_ds = VideoFrameDataset(train_samples, transform=train_transform)
        val_ds = VideoFrameDataset(val_samples, transform=val_transform)
        
        # 📌 고성능 데이터 로더 세팅 이식
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=2, persistent_workers=True)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=8, pin_memory=True, prefetch_factor=2, persistent_workers=True)
        
        model = get_model(model_type, torch.device("cuda"))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=BEST_PARAMS.get(model_type, 1e-4))
        scaler = GradScaler()
        
        save_path = f"model_spatial_{model_type}_fold{fold+1}.pth"
        early_stopping = EarlyStopping(patience=3, path=save_path)
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            loop = tqdm(train_loader, desc=f"Fold {fold+1} Ep {epoch+1} Train", leave=False)
            for inputs, labels in loop:
                inputs, labels = inputs.cuda(), labels.cuda()
                optimizer.zero_grad()
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item()
                loop.set_postfix(loss=loss.item())
            
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.cuda(), labels.cuda()
                    with autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    val_loss += loss.item()
            
            t_loss = train_loss / len(train_loader)
            v_loss = val_loss / len(val_loader)
            print(f"   - Epoch {epoch+1}: Train Loss = {t_loss:.4f}, Val Loss = {v_loss:.4f}")
            
            early_stopping(v_loss, model)
            if early_stopping.early_stop:
                print("   ⏹️ Validation Loss 개선 없음. 조기 종료합니다.")
                break
                
        print(f"    ✅ Fold {fold+1} 완료 (Best Model 저장됨: {save_path})")
        del model, optimizer, scaler, train_loader, val_loader, train_ds, val_ds
        clean_memory()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all")
    args = parser.parse_args()
    target_models = ["xception", "convnext", "swin"] if args.model == "all" else [args.model]
    for m in target_models:
        train_model(m, epochs=15)