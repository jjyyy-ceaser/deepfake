import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2, os, random, gc, argparse, timm
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold

# 📂 [경로 고정] - 불균형 데이터셋 경로
TRAIN_DIR = r"C:\Users\leejy\Desktop\test_experiment\dataset\processed_cases\train\case4_mixed"

def clean_memory():
    gc.collect()
    torch.cuda.empty_cache()

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

# 📌 [증강 기법 원복] 안정적인 기본 증강만 유지
train_transform = transforms.Compose([
    transforms.ToPILImage(), transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.ToPILImage(), transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class VideoFrameDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples, self.transform = samples, transform
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 🚀 [정보량 극대화] 영상의 중앙(50%) 프레임 타겟팅
        mid_idx = total // 2 if total > 0 else 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_idx)
        ret, frame = cap.read()
        
        if not ret: 
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            
        cap.release()
        if not ret: frame = np.zeros((224, 224, 3), dtype=np.uint8)
        else: frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.transform(frame), label

def train_model(model_type, epochs=30): 
    clean_memory()
    all_samples = []
    for sub, lab in [("real", 0), ("fake", 1)]:
        d = os.path.join(TRAIN_DIR, sub)
        if os.path.exists(d): all_samples += [(os.path.join(d, f), lab) for f in os.listdir(d) if f.endswith('.mp4')]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # ⚖️ [가중치 설정] Real(270) vs Fake(135) -> Fake에 2.0배 손실 페널티 부여
    class_weights = torch.tensor([1.0, 2.0]).cuda()
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    for fold, (train_idx, val_idx) in enumerate(skf.split(all_samples, [s[1] for s in all_samples])):
        print(f"\n🔄 [Training] Fold {fold+1}/5 Start - {model_type.upper()}")
        
        # 🚀 [VRAM 풀가동 세팅] Batch Size 128 증폭 및 CPU 데이터 공급 가속
        train_loader = DataLoader(
            VideoFrameDataset([all_samples[i] for i in train_idx], train_transform), 
            batch_size=128,              
            shuffle=True, 
            num_workers=4, 
            pin_memory=True,             
            prefetch_factor=4,           
            persistent_workers=True      
        )
        
        val_loader = DataLoader(
            VideoFrameDataset([all_samples[i] for i in val_idx], val_transform), 
            batch_size=128,              
            shuffle=False, 
            num_workers=4, 
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True
        )
        
        # 📌 [아키텍처 로드]
        if model_type == "xception": model = timm.create_model('xception', pretrained=True, num_classes=2)
        elif model_type == "convnext": model = timm.create_model('convnext_tiny', pretrained=True, num_classes=2)
        else: model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=2)
        model = model.cuda()

        # 📌 [동결 해제 (Unfreeze) 전략] 아키텍처별 맞춤형 깊은 층 해제
        for param in model.parameters(): param.requires_grad = False 
        
        if model_type == "xception":
            unfreeze_layers = [model.block10, model.block11, model.block12, model.conv4, model.bn4, model.fc]
        elif model_type == "convnext":
            unfreeze_layers = [model.stages[-2], model.stages[-1], model.head]
        elif model_type == "swin":
            unfreeze_layers = [model.layers[-2], model.layers[-1], model.head]
            
        for layer in unfreeze_layers:
            for param in layer.parameters(): param.requires_grad = True

        # 📌 [동적 학습률 매핑] 초기 LR 장전
        if model_type == 'xception': best_lr = 1e-04 
        elif model_type == 'convnext': best_lr = 1e-04
        elif model_type == 'swin': best_lr = 5e-05

        # 🚀 [가중치 감쇠(Weight Decay) 투여] 과적합을 막기 위한 L2 정규화 페널티 적용
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=best_lr, 
            weight_decay=1e-2  
        )
        
        scaler = torch.amp.GradScaler('cuda')
        early_stopping = EarlyStopping(patience=6, path=f"model_spatial_{model_type}_fold{fold+1}.pth")
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for inputs, labels in tqdm(train_loader, desc=f"Fold {fold+1} Ep {epoch+1}", leave=False):
                inputs, labels = inputs.cuda(), labels.cuda()
                optimizer.zero_grad()
                with torch.amp.autocast('cuda'):
                    loss = criterion(model(inputs), labels)
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
                print("   ⚠️ 조기 종료(Early Stopping) 발동")
                break
        del model; clean_memory()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all")
    args = parser.parse_args()
    
    target_models = ["xception", "convnext", "swin"] if args.model == "all" else [args.model]
    for m in target_models: 
        train_model(m)