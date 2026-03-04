import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import torchvision.transforms.functional as TF
import cv2
import os
import gc
import random
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from pathlib import Path

# ======================================================
# [설정] 7대 체크리스트 및 하이퍼파라미터
# ======================================================
# 1. 데이터 및 저장 경로
BASE_DATA_DIR = Path(r"C:\Users\leejy\Desktop\test_experiment\dataset\final_dataset")
SAVE_MODEL_DIR = Path(r"C:\Users\leejy\Desktop\test_experiment\models")

# 2. 학습할 도메인 및 모델 순서
TARGET_DOMAINS = ["raw", "youtube", "instagram", "kakao_high", "kakao_normal"]
TARGET_MODELS = ["r3d", "r2plus1d"]  # 두 가지 Temporal 모델 모두 학습

# 3. 하이퍼파라미터
SEQUENCE_LENGTH = 16    # 3D CNN 입력 프레임 수
IMG_SIZE = 112          # 이미지 크기 (메모리 절약: 112, 고성능: 224)
BATCH_SIZE = 8          # VRAM에 따라 조절 (4~16 권장)
EPOCHS = 10             # Fold당 에폭 수
LEARNING_RATE = 1e-4
SEED = 42
PATIENCE = 5            # 조기 종료 조건

# ======================================================

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def clean_memory():
    """🔥 VRAM 메모리 누수 방지를 위한 강력한 청소부"""
    gc.collect()
    torch.cuda.empty_cache()

def check_7_points(domain, model_name):
    print(f"\n✅ [CHECK] {domain.upper()} - {model_name.upper()} 7대 점검 시작")
    
    # 1. Seed
    seed_everything(SEED)
    print(f"   1. Seed: {SEED} (Fixed)")
    
    # 2. Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   2. Device: {device}")
    
    # 3. Data Path
    data_path = BASE_DATA_DIR / domain / "train"
    if not data_path.exists():
        raise FileNotFoundError(f"❌ 데이터 경로 없음: {data_path}")
    print(f"   3. Data Path: {data_path}")
    
    # 4. Hparams
    print(f"   4. Params: Batch={BATCH_SIZE}, LR={LEARNING_RATE}, Seq={SEQUENCE_LENGTH}")
    
    # 5. Save Path
    save_dir = SAVE_MODEL_DIR / f"temporal_{domain}_{model_name}"
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"   5. Save Dir: {save_dir}")
    
    return device, data_path, save_dir

# ======================================================
# [Dataset] 영상 프레임 시퀀스 로더
# ======================================================
class VideoSequenceDataset(Dataset):
    def __init__(self, samples, transform=None, is_train=True):
        self.samples = samples
        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 랜덤 시작점 (Train) vs 중앙 시작점 (Val)
        if total > SEQUENCE_LENGTH:
            if self.is_train:
                start = random.randint(0, total - SEQUENCE_LENGTH)
            else:
                start = (total - SEQUENCE_LENGTH) // 2
        else:
            start = 0
            
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        
        frames = []
        # Train일 때만 좌우 반전 랜덤 적용
        apply_hflip = (random.random() > 0.5) if self.is_train else False
        
        for _ in range(SEQUENCE_LENGTH):
            ret, frame = cap.read()
            if not ret:
                # 프레임 부족 시 검은 화면 패딩
                frame = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            pil_img = transforms.ToPILImage()(frame)
            pil_img = transforms.Resize((IMG_SIZE, IMG_SIZE))(pil_img)
            
            if apply_hflip:
                pil_img = TF.hflip(pil_img)
            
            if self.is_train:
                # 약한 색상 증강
                pil_img = transforms.ColorJitter(brightness=0.1, contrast=0.1)(pil_img)
                
            frames.append(self.transform(pil_img))
            
        cap.release()
        
        # (T, C, H, W) -> (C, T, H, W) : 3D CNN 입력 형태
        return torch.stack(frames).permute(1, 0, 2, 3), label

# ======================================================
# [Training] 단일 도메인 & 단일 모델 학습 함수
# ======================================================
def train_one_session(domain, model_name):
    # 1. 체크리스트 및 설정 로드
    device, data_path, save_dir = check_7_points(domain, model_name)
    clean_memory()
    
    # 2. 데이터 로드 (Real/Fake)
    all_samples = []
    for sub, lab in [("real", 0), ("fake", 1)]:
        d = data_path / sub
        if d.exists():
            files = list(d.glob("*.mp4"))
            all_samples += [(p, lab) for p in files]
            
    if not all_samples:
        print(f"⚠️ {domain} 데이터가 비어있습니다. 건너뜁니다.")
        return

    print(f"   6. Data Count: Total {len(all_samples)} samples")

    # 3. Transform (Kinetics-400 Norm)
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
    ])

    # 4. 5-Fold Stratified Cross Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    labels_only = [s[1] for s in all_samples]
    
    print(f"   7. Fold: 5-Fold Stratified CV Ready")
    print("="*50)

    for fold, (train_idx, val_idx) in enumerate(skf.split(all_samples, labels_only)):
        print(f"\n🔄 [{domain.upper()}-{model_name.upper()}] Fold {fold+1}/5 Start")
        
        # Dataset & Loader
        train_ds = VideoSequenceDataset([all_samples[i] for i in train_idx], base_transform, True)
        val_ds = VideoSequenceDataset([all_samples[i] for i in val_idx], base_transform, False)
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
        
        # Model Selection
        if model_name == "r3d":
            model = models.video.r3d_18(weights='KINETICS400_V1')
        else: # r2plus1d
            model = models.video.r2plus1d_18(weights='KINETICS400_V1')
            
        # 이진 분류로 변경
        model.fc = nn.Linear(model.fc.in_features, 2)
        model = model.to(device)
        
        # Loss & Optimizer
        criterion = nn.CrossEntropyLoss() # 1:1 비율이므로 가중치 불필요
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
        scaler = torch.amp.GradScaler('cuda') # Mixed Precision
        
        best_acc = 0.0
        patience = 0
        
        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0.0
            
            # Train Loop
            pbar = tqdm(train_loader, desc=f"Ep {epoch+1}", leave=False)
            for inputs, targets in pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
            
            # Validation Loop
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    with torch.amp.autocast('cuda'):
                        outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
            
            val_acc = correct / total
            avg_loss = train_loss / len(train_loader)
            
            print(f"   📅 Ep {epoch+1}: Loss {avg_loss:.4f} | Val Acc {val_acc:.4f}")
            
            # Checkpoint
            if val_acc > best_acc:
                best_acc = val_acc
                patience = 0
                torch.save(model.state_dict(), save_dir / f"best_fold{fold+1}.pth")
                # print(f"      💾 Best Save (Acc: {best_acc:.4f})")
            else:
                patience += 1
                if patience >= PATIENCE:
                    print("      🛑 Early Stopping")
                    break
        
        # Fold 종료 후 반드시 메모리 정리
        del model, optimizer, train_loader, val_loader, scaler
        clean_memory()
        print(f"   🧹 Fold {fold+1} Finished & Memory Cleaned")

# ======================================================
# [Main] 마스터 루프
# ======================================================
def main():
    print("🚀 [Master Train] Temporal Model 통합 학습 시작")
    print(f"📋 Domains: {TARGET_DOMAINS}")
    print(f"📋 Models: {TARGET_MODELS}")
    
    for domain in TARGET_DOMAINS:
        for model_name in TARGET_MODELS:
            try:
                train_one_session(domain, model_name)
                print(f"\n🎉 {domain.upper()} - {model_name.upper()} 완료!\n")
            except Exception as e:
                print(f"\n❌ ERROR in {domain}-{model_name}: {e}")
                # 에러 나도 다음 도메인으로 계속 진행
                continue

if __name__ == "__main__":
    main()