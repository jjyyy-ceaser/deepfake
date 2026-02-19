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

# 📂 [경로 고정]
BASE_DIR = "C:/Users/leejy/Desktop/test_experiment/dataset/final_datasets"

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

def train_model(model_type, dataset_name, epochs=5): # 📌 Epoch 5로 상향
    clean_memory()
    folder_map = {
        "pure": "dataset_A_pure", 
        "mixed": "dataset_B_mixed", 
        "worst": "dataset_C_worst"
    }
    data_path = os.path.join(BASE_DIR, folder_map[dataset_name])
    print(f"🔥 [Spatial 5-Fold] 모델: {model_type} | 데이터: {dataset_name}")

    all_samples = []
    real_dir, fake_dir = os.path.join(data_path, "real"), os.path.join(data_path, "fake")
    if os.path.exists(real_dir): all_samples += [(os.path.join(real_dir, f), 0) for f in os.listdir(real_dir) if f.endswith('.mp4')]
    if os.path.exists(fake_dir): all_samples += [(os.path.join(fake_dir, f), 1) for f in os.listdir(fake_dir) if f.endswith('.mp4')]
    
    if len(all_samples) == 0:
        print(f"❌ 데이터 없음: {data_path}")
        return

    # 📌 5-Fold 적용 (n_splits=5)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y = [s[1] for s in all_samples]

    for fold, (train_idx, val_idx) in enumerate(skf.split(all_samples, y)):
        print(f"  🔄 Fold {fold+1}/5 Start...")
        
        train_samples = [all_samples[i] for i in train_idx]
        val_samples = [all_samples[i] for i in val_idx] # Validation용 (여기선 학습에 집중)
        
        train_ds = VideoFrameDataset(train_samples, transform=transform)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=2, persistent_workers=True)
        
        model = get_model(model_type, torch.device("cuda"))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=BEST_PARAMS.get(model_type, 1e-4))
        scaler = GradScaler()
        
        model.train()
        for epoch in range(epochs):
            loop = tqdm(train_loader, desc=f"Fold {fold+1} Ep {epoch+1}", leave=False)
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
        
        torch.save(model.state_dict(), f"model_spatial_{model_type}_{dataset_name}_fold{fold+1}.pth")
        print(f"    ✅ Fold {fold+1} 저장 완료")
        del model, optimizer, scaler, train_loader
        clean_memory()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all")
    parser.add_argument("--dataset", type=str, default="all") 
    args = parser.parse_args()
    target_models = ["xception", "convnext", "swin"] if args.model == "all" else [args.model]
    target_datasets = ["pure", "mixed", "worst"] if args.dataset == "all" else [args.dataset]
    for m in target_models:
        for d in target_datasets:
            train_model(m, d, epochs=5) # 📌 Epoch 5 호출