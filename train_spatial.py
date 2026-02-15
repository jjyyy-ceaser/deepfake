import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import os
import random
import numpy as np
from tqdm import tqdm
import argparse
import timm

# ==========================================
# ⚙️ 설정
# ==========================================
BASE_DIR = "C:/Users/leejy/Desktop/test_experiment/dataset"
IMG_SIZE = 224

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class VideoFrameDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        
        real_dir = os.path.join(data_dir, "real")
        fake_dir = os.path.join(data_dir, "fake")
        
        if os.path.exists(real_dir):
            for f in os.listdir(real_dir):
                if f.lower().endswith('.mp4'): self.samples.append((os.path.join(real_dir, f), 0))
        if os.path.exists(fake_dir):
            for f in os.listdir(fake_dir):
                if f.lower().endswith('.mp4'): self.samples.append((os.path.join(fake_dir, f), 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame = None
        if total_frames > 0:
            random_idx = random.randint(0, total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, random_idx)
            ret, frame = cap.read()
        cap.release()
        
        if frame is None:
            frame = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            frame = self.transform(frame)
        return frame, label

def get_model(model_name, device):
    print(f"🏗️ 모델 빌드 중: {model_name.upper()}...")
    if model_name == "xception":
        model = timm.create_model('xception', pretrained=True, num_classes=2)
    elif model_name == "convnext":
        model = timm.create_model('convnext_tiny', pretrained=True, num_classes=2)
    elif model_name == "swin":
        model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=2)
    return model.to(device)

def train_model(model_type, dataset_name, epochs=5):
    # 폴더명 매칭 수정
    folder_map = {
        "pure": os.path.join("2_exp_train_pure", "train"),
        "mixed": "2_train_mixed",
        "worst": "2_train_worst"
    }
    target_folder = folder_map[dataset_name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n🔥 [Spatial 학습] 모델: {model_type.upper()} | 데이터: {dataset_name.upper()}")
    
    data_path = os.path.join(BASE_DIR, target_folder)
    dataset = VideoFrameDataset(data_path, transform=transform)
    if len(dataset) == 0:
        print(f"❌ 데이터 없음: {data_path}")
        return
        
    batch_size = 16 if model_type == "swin" else 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    model = get_model(model_type, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    model.train()
    for epoch in range(epochs):
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())
            
    save_name = f"model_spatial_{model_type}_{dataset_name}.pth"
    torch.save(model.state_dict(), save_name)
    print(f"✅ 저장 완료: {save_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all")
    parser.add_argument("--dataset", type=str, default="all")
    args = parser.parse_args()
    
    target_models = ["xception", "convnext", "swin"] if args.model == "all" else [args.model]
    target_datasets = ["pure", "mixed", "worst"] if args.dataset == "all" else [args.dataset]
    
    for m in target_models:
        for d in target_datasets:
            train_model(m, d, epochs=5)