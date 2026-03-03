import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
import cv2, os, gc, random
import numpy as np
from tqdm import tqdm
from transformers import VideoMAEForVideoClassification
from sklearn.model_selection import StratifiedKFold

# 📂 [경로 고정] 불균형 데이터셋 경로
TRAIN_DIR = r"C:\Users\leejy\Desktop\test_experiment\dataset\split_datasets\dataset_A\train"
SEQUENCE_LENGTH = 16
IMG_SIZE = 224

# 🚀 [VRAM 안전선 최대 튜닝] 트랜스포머 기반이라 VRAM 소모가 극심함 (권장 8~16)
BATCH_SIZE = 8

base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
            
            if self.is_train:
                pil_img = transforms.ColorJitter(brightness=0.2, contrast=0.2)(pil_img)
                
            frames.append(self.transform(pil_img))
        cap.release()
        return torch.stack(frames), label # VideoMAE 규격에 맞게 permute 제외

def train_videomae(epochs=30):
    clean_memory()
    all_samples = []
    for sub, lab in [("real", 0), ("fake", 1)]:
        d = os.path.join(TRAIN_DIR, sub)
        if os.path.exists(d): all_samples += [(os.path.join(d, f), lab) for f in os.listdir(d) if f.endswith('.mp4')]
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # ⚖️ [가중치 설정]
    class_weights = torch.tensor([1.0, 2.0]).cuda()
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    for fold, (train_idx, val_idx) in enumerate(skf.split(all_samples, [s[1] for s in all_samples])):
        print(f"\n🔄 [Training] Fold {fold+1}/5 Start - VIDEOMAE")
        
        # 🚀 [VRAM 풀가동 로더 튜닝]
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
        
        model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base", num_labels=2, ignore_mismatched_sizes=True).cuda()

        for param in model.parameters(): param.requires_grad = False
        for param in model.videomae.encoder.layer[-1].parameters(): param.requires_grad = True
        for param in model.classifier.parameters(): param.requires_grad = True

        best_lr = 1e-04  

        # 🚀 [가중치 감쇠 투여]
        optimizer = optim.AdamW([
            {'params': model.videomae.encoder.layer[-1].parameters(), 'lr': best_lr * 0.1},
            {'params': model.classifier.parameters(), 'lr': best_lr}
        ], weight_decay=1e-2)

        scaler, early_stopping = torch.amp.GradScaler('cuda'), EarlyStopping(patience=6, path=f"model_temporal_videomae_fold{fold+1}.pth")

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for inputs, labels in tqdm(train_loader, desc=f"Fold {fold+1} Ep {epoch+1}", leave=False):
                inputs, labels = inputs.cuda(), labels.cuda()
                optimizer.zero_grad()
                with torch.amp.autocast('cuda'):
                    # 🚀 [수동 로스 계산] VideoMAE 내장 로스 대신 가중치 적용 로스로 우회
                    logits = model(pixel_values=inputs).logits
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                train_loss += loss.item()
            
            model.eval(); val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.cuda(), labels.cuda()
                    with torch.amp.autocast('cuda'): 
                        logits = model(pixel_values=inputs).logits
                        val_loss += criterion(logits, labels).item()
                        
            v_loss = val_loss / len(val_loader)
            print(f"   - Ep {epoch+1}: Train {train_loss/len(train_loader):.4f} | Val {v_loss:.4f}")
            early_stopping(v_loss, model)
            if early_stopping.early_stop: 
                print("   ⚠️ 조기 종료 발동")
                break
        del model; clean_memory()

if __name__ == "__main__": train_videomae()