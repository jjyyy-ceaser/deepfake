import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms, models
from sklearn.model_selection import KFold
import os
import argparse
import timm
from transformers import VideoMAEForVideoClassification
from sklearn.metrics import roc_auc_score
from data_loader import UnifiedDataset  # 위에서 만든 로더 임포트

# 경로 설정 (사용자 환경)
BASE_DIR = "C:/Users/leejy/Desktop/test_experiment/dataset"

def get_data_samples(dataset_type="mixed"):
    # 2_train_mixed 폴더 고정 사용 (강건성 학습용)
    target_dir = os.path.join(BASE_DIR, "2_train_mixed")
    samples = []
    for cls_name, label in [("real", 0), ("fake", 1)]:
        d_path = os.path.join(target_dir, cls_name)
        if os.path.exists(d_path):
            files = sorted([os.path.join(d_path, f) for f in os.listdir(d_path) if f.endswith('.mp4')])
            for f in files: samples.append((f, label))
    return samples

def build_model(model_name, device):
    # 6개 모델 정의 확인 완료
    if "videomae" in model_name:
        model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base", num_labels=2, ignore_mismatched_sizes=True)
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
        raise ValueError(f"Unknown Model: {model_name}")
    return model.to(device)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--k_folds", type=int, default=5)
    parser.add_argument("--save_model", type=str, default="False") # "True" or "False"
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_type = 'temporal' if any(x in args.model for x in ['videomae', 'r3d', 'r2plus1d']) else 'spatial'

    # K-Fold Split (random_state=42로 고정하여 Grid Search와 본학습의 데이터 일치 보장)
    all_samples = get_data_samples()
    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=42)
    splits = list(kf.split(all_samples))
    train_idx, val_idx = splits[args.fold]
    
    # Transform
    tf = transforms.Compose([
        transforms.ToPILImage(), transforms.Resize((224, 224)),
        transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_ds = UnifiedDataset([all_samples[i] for i in train_idx], model_type, transform=tf)
    val_ds = UnifiedDataset([all_samples[i] for i in val_idx], model_type, transform=tf)
    
    # DataLoader (Pin Memory 사용)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = build_model(args.model, device)
    
    # Optimizer Setting
    if args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()
    
    best_auc = 0.0

    # Training Loop
    for epoch in range(args.epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # VideoMAE Input Fix: (B, C, T, H, W) -> (B, T, C, H, W)
            if "videomae" in args.model:
                inputs = inputs.permute(0, 2, 1, 3, 4)

            optimizer.zero_grad()
            with autocast():
                if "videomae" in args.model:
                    outputs = model(pixel_values=inputs).logits
                else:
                    outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Validation Loop (Epoch마다 체크)
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
        
        try: val_auc = roc_auc_score(y_true, y_probs)
        except: val_auc = 0.5
        
        if val_auc > best_auc:
            best_auc = val_auc
            # Save if requested (Only save the best within this fold)
            if args.save_model == "True":
                # 임시 이름으로 저장 (Manager가 나중에 이름 바꿈)
                torch.save(model.state_dict(), f"temp_best_{args.model}.pth")

    # Manager가 읽을 수 있게 출력
    print(f"FINAL_VAL_AUC: {best_auc}")

if __name__ == "__main__":
    main()