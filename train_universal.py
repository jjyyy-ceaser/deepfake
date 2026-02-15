import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms, models
from sklearn.model_selection import KFold
import cv2
import os
import numpy as np
import argparse
import timm
from transformers import VideoMAEForVideoClassification
from sklearn.metrics import roc_auc_score, accuracy_score

# ==========================================
# ⚙️ 설정 및 데이터셋
# ==========================================
BASE_DIR = "C:/Users/leejy/Desktop/test_experiment/dataset"
IMG_SIZE = 224
SEQ_LEN = 16

class UnifiedDataset(Dataset):
    def __init__(self, samples, model_type, transform=None):
        self.samples = samples # [(path, label), ...]
        self.model_type = model_type
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        cap = cv2.VideoCapture(path)
        
        try:
            if self.model_type == 'spatial':
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, np.random.randint(0, total))
                ret, frame = cap.read()
                if not ret: frame = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
                else: frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                if self.transform: frame = self.transform(frame)
                cap.release()
                return frame, label
            
            else: # temporal
                frames = []
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total >= SEQ_LEN:
                    start = np.random.randint(0, total - SEQ_LEN)
                    indices = np.arange(start, start + SEQ_LEN)
                else:
                    indices = np.arange(SEQ_LEN) % (total if total > 0 else 1)
                
                for i in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ret, frame = cap.read()
                    if not ret: frame = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
                    else: frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if self.transform: frame = self.transform(frame)
                    frames.append(frame)
                
                cap.release()
                frames = torch.stack(frames).permute(1, 0, 2, 3) # (C, T, H, W)
                return frames, label
                
        except Exception:
            cap.release()
            # 에러 시 더미 데이터 반환
            shape = (3, IMG_SIZE, IMG_SIZE) if self.model_type == 'spatial' else (3, SEQ_LEN, IMG_SIZE, IMG_SIZE)
            return torch.zeros(shape), label

def get_data_samples(dataset_name):
    # dataset_name: pure / mixed / worst
    folder_map = {
        "pure": os.path.join("2_exp_train_pure", "train"),
        "mixed": "2_train_mixed",
        "worst": "2_train_worst"
    }
    target_dir = os.path.join(BASE_DIR, folder_map[dataset_name])
    samples = []
    
    for cls_name, label in [("real", 0), ("fake", 1)]:
        d_path = os.path.join(target_dir, cls_name)
        if os.path.exists(d_path):
            files = [os.path.join(d_path, f) for f in os.listdir(d_path) if f.lower().endswith('.mp4')]
            # 정렬하여 5-Fold의 일관성 보장
            files.sort()
            for f in files:
                samples.append((f, label))
    return samples

def build_model(model_name, device):
    if "videomae" in model_name:
        model = VideoMAEForVideoClassification.from_pretrained(
            "MCG-NJU/videomae-base", num_labels=2, ignore_mismatched_sizes=True
        )
    elif "r3d" in model_name:
        model = models.video.r3d_18(weights=models.video.R3D_18_Weights.KINETICS400_V1)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif "r2plus1d" in model_name:
        model = models.video.r2plus1d_18(weights=models.video.R2Plus1D_18_Weights.KINETICS400_V1)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif "swin" in model_name:
        model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=2)
    elif "convnext" in model_name:
        model = timm.create_model('convnext_tiny', pretrained=True, num_classes=2)
    elif "xception" in model_name:
        model = timm.create_model('xception', pretrained=True, num_classes=2)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model.to(device)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="mixed")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--k_folds", type=int, default=5)
    parser.add_argument("--save_model", type=str, default="False")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_type = 'temporal' if any(x in args.model for x in ['videomae', 'r3d', 'r2plus1d']) else 'spatial'

    # 1. 데이터 로드 및 K-Fold 분할
    all_samples = get_data_samples(args.dataset)
    if len(all_samples) == 0:
        print("FINAL_VAL_AUC: 0.0") # 에러 처리
        return

    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=42)
    splits = list(kf.split(all_samples))
    train_idx, val_idx = splits[args.fold]
    
    train_samples = [all_samples[i] for i in train_idx]
    val_samples = [all_samples[i] for i in val_idx]

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_ds = UnifiedDataset(train_samples, model_type, transform)
    val_ds = UnifiedDataset(val_samples, model_type, transform)

    # 4070 SUPER 최적화: num_workers 조절
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # 2. 모델 및 옵티마이저
    model = build_model(args.model, device)
    
    if args.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    # 3. 학습 루프
    best_auc = 0.0
    
    for epoch in range(args.epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            if "videomae" in args.model:
                inputs = inputs.permute(0, 2, 1, 3, 4) # (B, T, C, H, W)
                
            optimizer.zero_grad()
            with autocast():
                if "videomae" in args.model:
                    outputs = model(pixel_values=inputs, labels=labels).logits
                else:
                    outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Validation
        model.eval()
        y_true, y_probs = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                if "videomae" in args.model:
                    inputs = inputs.permute(0, 2, 1, 3, 4)
                    outputs = model(pixel_values=inputs).logits
                else:
                    outputs = model(inputs)
                
                probs = torch.softmax(outputs, dim=1)[:, 1]
                y_true.extend(labels.cpu().numpy())
                y_probs.extend(probs.cpu().numpy())
        
        try:
            val_auc = roc_auc_score(y_true, y_probs)
        except:
            val_auc = 0.5
        
        if val_auc > best_auc:
            best_auc = val_auc
            if args.save_model == "True":
                torch.save(model.state_dict(), f"best_{args.model}_{args.dataset}.pth")

    # 매니저에게 결과 전달
    acc = accuracy_score(y_true, np.array(y_probs) > 0.5)
    print(f"FINAL_VAL_AUC: {best_auc}")
    print(f"FINAL_VAL_ACC: {acc}")

if __name__ == "__main__":
    main()