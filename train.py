import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import timm
import torchvision.models.video as video_models
import cv2
import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- [설정값] ---
BATCH_SIZE = 4        
EPOCHS = 5            
LEARNING_RATE = 1e-4  
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "dataset"  

print(f"🔧 학습 장치 설정: {DEVICE}")

# ==========================================
# 1. 데이터셋 클래스 (짧은 영상 패딩 기능 포함)
# ==========================================
class DeepfakeDataset(Dataset):
    def __init__(self, video_paths, labels, num_frames=16, transform=None):
        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames 

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        path = self.video_paths[idx]
        label = self.labels[idx]
        
        cap = cv2.VideoCapture(path)
        frames = []
        while len(frames) < self.num_frames:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        # 영상이 너무 짧거나 깨진 경우 (검은 화면으로 대체)
        if not frames:
            frames = [np.zeros((224, 224, 3), dtype=np.uint8)] * self.num_frames
        
        # ⭐ 핵심: 프레임이 모자라면 마지막 장면을 복사해서 채운다 (Padding)
        while len(frames) < self.num_frames:
            frames.append(frames[-1])

        frames_np = np.array(frames, dtype=np.float32) / 255.0 
        video_tensor = torch.tensor(frames_np).permute(3, 0, 1, 2) 

        return video_tensor, label

# ==========================================
# 2. 데이터 불러오기
# ==========================================
print("\n📂 데이터셋 스캔 중...")
real_videos = glob.glob(os.path.join(DATA_DIR, "real", "*.mp4"))
fake_videos = glob.glob(os.path.join(DATA_DIR, "fake", "*.mp4"))

# 데이터 확인
if not real_videos and not fake_videos:
    print("⚠️ [오류] 데이터가 없습니다! dataset 폴더 위치를 확인하세요.")
    import sys; sys.exit()

print(f"   - Real 영상: {len(real_videos)}개")
print(f"   - Fake 영상: {len(fake_videos)}개")

paths = real_videos + fake_videos
labels = [0] * len(real_videos) + [1] * len(fake_videos) 

# 데이터 분할
train_paths, test_paths, train_labels, test_labels = train_test_split(
    paths, labels, test_size=0.2, random_state=42, shuffle=True
)

train_dataset = DeepfakeDataset(train_paths, train_labels)
test_dataset = DeepfakeDataset(test_paths, test_labels)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==========================================
# 3. 모델 정의
# ==========================================
def get_xception():
    try:
        model = timm.create_model('xception', pretrained=True, num_classes=2)
    except:
        model = timm.create_model('legacy_xception', pretrained=True, num_classes=2)
    return model.to(DEVICE)

def get_r3d():
    model = video_models.r3d_18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model.to(DEVICE)

# ==========================================
# 4. 학습 함수
# ==========================================
def train_model(model, model_name):
    print(f"\n🚀 [{model_name}] 학습 시작...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        correct = 0
        total = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for videos, labels in loop:
            videos, labels = videos.to(DEVICE), labels.to(DEVICE)
            
            if model_name == "Xception":
                inputs = videos[:, :, 0, :, :] 
            else: 
                inputs = videos

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loop.set_postfix(acc=100*correct/total)

    print(f"✅ [{model_name}] 학습 완료! 정확도: {100*correct/total:.2f}%")
    return model

# ==========================================
# 5. 실행
# ==========================================
if __name__ == "__main__":
    spatial_model = get_xception()
    spatial_model = train_model(spatial_model, "Xception")
    torch.save(spatial_model.state_dict(), "xception_result.pth")

    temporal_model = get_r3d()
    temporal_model = train_model(temporal_model, "R3D-18")
    torch.save(temporal_model.state_dict(), "r3d_result.pth")

    print("\n🎉 모든 실험이 종료되었습니다!")