import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import os
import numpy as np
from tqdm import tqdm
import argparse
from transformers import VideoMAEForVideoClassification

# ==========================================
# ⚙️ 1. VideoMAE 전용 설정
# ==========================================
BASE_DIR = "C:/Users/leejy/Desktop/test_experiment/dataset"
SEQUENCE_LENGTH = 16
IMG_SIZE = 224 # VideoMAE 모델의 표준 입력 해상도

# VideoMAE 공식 정규화 값 적용
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 📂 2. 데이터셋 클래스
# ==========================================
class VideoSequenceDataset(Dataset):
    def __init__(self, data_dir, sequence_length=16, transform=None):
        self.data_dir = data_dir
        self.seq_len = sequence_length
        self.transform = transform
        self.samples = []
        
        real_dir = os.path.join(data_dir, "real")
        fake_dir = os.path.join(data_dir, "fake")
        
        # mp4 확장자 대소문자 무관하게 탐색
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
        
        start_frame = 0
        if total_frames > self.seq_len:
            start_frame = np.random.randint(0, total_frames - self.seq_len)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames = []
        for _ in range(self.seq_len):
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
        cap.release()
        
        # VideoMAE 입력 형식: (C, T, H, W)
        frames = torch.stack(frames).permute(1, 0, 2, 3) 
        return frames, label

# ==========================================
# 🔥 3. VideoMAE 학습 핵심 함수
# ==========================================
def train_videomae(dataset_name, epochs=5):
    # 실제 폴더 구조 매칭 수정
    folder_map = {
        "pure": os.path.join("2_exp_train_pure", "train"),
        "mixed": "2_train_mixed",
        "worst": "2_train_worst"
    }
    target_folder = folder_map[dataset_name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n==================================================")
    print(f"🔥 [VideoMAE 학습] 데이터셋: {dataset_name.upper()}")
    print(f"==================================================")
    
    data_path = os.path.join(BASE_DIR, target_folder)
    dataset = VideoSequenceDataset(data_path, SEQUENCE_LENGTH, transform)
    if len(dataset) == 0: 
        print(f"❌ 데이터가 없습니다: {data_path}")
        return
    
    # RTX 4070 SUPER 12GB 기준: 배치 사이즈 4가 권장되나 OOM 발생 시 2로 낮추세요.
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0) 
    
    print("📥 VideoMAE 공식 베이스 모델 로딩 중...")
    model = VideoMAEForVideoClassification.from_pretrained(
        "MCG-NJU/videomae-base", 
        num_labels=2,
        ignore_mismatched_sizes=True
    )
    model = model.to(device)
    
    # Transformer 모델에는 AdamW 옵티마이저가 효과적입니다.
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    
    model.train()
    for epoch in range(epochs):
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)
            # VideoMAE 입력 규격: (B, T, C, H, W)
            inputs = inputs.permute(0, 2, 1, 3, 4) 
            
            optimizer.zero_grad()
            outputs = model(pixel_values=inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            logits = outputs.logits
            _, predicted = torch.max(logits, 1)
            correct = (predicted == labels).sum().item()
            acc = 100 * correct / labels.size(0)

            loop.set_postfix(loss=loss.item(), acc=acc)
            
    save_name = f"model_temporal_videomae_{dataset_name}.pth"
    torch.save(model.state_dict(), save_name)
    print(f"✅ 학습 완료 및 저장: {save_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="all")
    args = parser.parse_args()
    
    datasets_list = ["pure", "mixed", "worst"] if args.dataset == "all" else [args.dataset]
    for d in datasets_list:
        train_videomae(d, epochs=5)