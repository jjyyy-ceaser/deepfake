import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2, os, gc, timm, random
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from pathlib import Path

# [설정]
BASE_DATA_DIR = Path(r"C:\Users\leejy\Desktop\test_experiment\dataset\final_dataset")
SAVE_MODEL_DIR = Path(r"C:\Users\leejy\Desktop\test_experiment\models")
TARGET_DOMAINS = ["raw", "youtube", "instagram", "kakao_high", "kakao_normal"]
BATCH_SIZE = 64 

def clean_memory():
    gc.collect()
    torch.cuda.empty_cache()

class EarlyStopping:
    def __init__(self, patience=5, path=None): 
        self.patience, self.counter, self.best_loss, self.early_stop, self.path = patience, 0, None, False, path
    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss, self.counter = val_loss, 0
            torch.save(model.state_dict(), self.path)
        else:
            self.counter += 1
            if self.counter >= self.patience: self.early_stop = True

train_transform = transforms.Compose([
    transforms.ToPILImage(), transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
val_transform = transforms.Compose([
    transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class VideoFrameDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples, self.transform = samples, transform
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        mid_idx = total // 2 if total > 0 else 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret: frame = np.zeros((224, 224, 3), dtype=np.uint8)
        else: frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.transform(frame), label

def train_domain_model(domain, model_type, epochs=10):
    clean_memory()
    data_path = BASE_DATA_DIR / domain / "train"
    all_samples = []
    for sub, lab in [("real", 0), ("fake", 1)]:
        d = data_path / sub
        if d.exists(): all_samples += [(p, lab) for p in d.glob("*.mp4")]

    if not all_samples: return

    save_dir = SAVE_MODEL_DIR / f"spatial_{domain}_{model_type}"
    save_dir.mkdir(parents=True, exist_ok=True)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    labels = [s[1] for s in all_samples]
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0]).cuda())

    for fold, (train_idx, val_idx) in enumerate(skf.split(all_samples, labels)):
        print(f"\n🔄 [{domain.upper()}] Fold {fold+1}/5 - {model_type}")
        clean_memory()
        
        # num_workers=4 적용
        train_loader = DataLoader(VideoFrameDataset([all_samples[i] for i in train_idx], train_transform), 
                                  batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(VideoFrameDataset([all_samples[i] for i in val_idx], val_transform), 
                                batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
        
        if model_type == "xception": model = timm.create_model('xception', pretrained=True, num_classes=2)
        elif model_type == "convnext": model = timm.create_model('convnext_tiny', pretrained=True, num_classes=2)
        else: model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=2)
        model = model.cuda()

        optimizer = optim.AdamW(model.parameters(), lr=1e-4 if model_type != 'swin' else 5e-05, weight_decay=1e-2)
        early_stopping = EarlyStopping(patience=5, path=str(save_dir / f"best_fold{fold+1}.pth"))
        
        for epoch in range(epochs):
            model.train()
            for inputs, targets in tqdm(train_loader, desc=f"Ep {epoch+1}", leave=False):
                inputs, targets = inputs.cuda(), targets.cuda()
                optimizer.zero_grad()
                loss = criterion(model(inputs), targets)
                loss.backward(); optimizer.step()
            
            model.eval(); val_loss = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.cuda(), targets.cuda()
                    val_loss += criterion(model(inputs), targets).item()
            
            avg_val = val_loss / len(val_loader)
            print(f"   📅 Ep {epoch+1}: Val Loss {avg_val:.4f}")
            early_stopping(avg_val, model)
            if early_stopping.early_stop: break
        
        del model; clean_memory()

if __name__ == "__main__":
    for domain in TARGET_DOMAINS:
        for m in ["xception", "convnext", "swin"]:
            train_domain_model(domain, m)