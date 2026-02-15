import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
import cv2
import os
import numpy as np
from tqdm import tqdm
import argparse
from transformers import VideoMAEForVideoClassification

BASE_DIR = "C:/Users/leejy/Desktop/test_experiment/dataset"
SEQUENCE_LENGTH = 16
IMG_SIZE = 224
BATCH_SIZE = 4 

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class VideoSequenceDataset(Dataset):
    def __init__(self, data_dir, sequence_length=16, transform=None):
        self.data_dir = data_dir
        self.seq_len = sequence_length
        self.transform = transform
        self.samples = []
        real_dir, fake_dir = os.path.join(data_dir, "real"), os.path.join(data_dir, "fake")
        if os.path.exists(real_dir): self.samples += [(os.path.join(real_dir, f), 0) for f in os.listdir(real_dir) if f.endswith('.mp4')]
        if os.path.exists(fake_dir): self.samples += [(os.path.join(fake_dir, f), 1) for f in os.listdir(fake_dir) if f.endswith('.mp4')]

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start_frame = np.random.randint(0, total_frames - self.seq_len) if total_frames > self.seq_len else 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames = []
        for _ in range(self.seq_len):
            ret, frame = cap.read()
            if not ret: frame = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            else: frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.transform: frame = self.transform(frame)
            frames.append(frame)
        cap.release()
        frames = torch.stack(frames).permute(1, 0, 2, 3) 
        return frames, label

def train_videomae(dataset_name, epochs=5):
    folder_map = {"pure": os.path.join("2_exp_train_pure", "train"), "mixed": "2_train_mixed", "worst": "2_train_worst"}
    data_path = os.path.join(BASE_DIR, folder_map[dataset_name])
    dataset = VideoSequenceDataset(data_path, SEQUENCE_LENGTH, transform)
    if len(dataset) == 0: return
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    device = torch.device("cuda")
    
    print(f"🔥 [VideoMAE 학습] 데이터셋: {dataset_name.upper()}")
    model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base", num_labels=2, ignore_mismatched_sizes=True).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    scaler = GradScaler()
    
    model.train()
    for epoch in range(epochs):
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
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
            
            logits = outputs.logits
            acc = (logits.argmax(1) == labels).float().mean().item() * 100
            loop.set_postfix(loss=loss.item(), acc=acc)
            
    torch.save(model.state_dict(), f"model_temporal_videomae_{dataset_name}.pth")
    print(f"✅ 저장 완료")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="all")
    args = parser.parse_args()
    datasets = ["pure", "mixed", "worst"] if args.dataset == "all" else [args.dataset]
    for d in datasets: train_videomae(d, epochs=5)