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
from pathlib import Path

# ======================================================
# [설정]
# ======================================================
BASE_DATA_DIR = Path(r"C:\Users\leejy\Desktop\test_experiment\dataset\final_dataset")
SAVE_MODEL_DIR = Path(r"C:\Users\leejy\Desktop\test_experiment\models")
TARGET_DOMAINS = ["raw", "youtube", "instagram", "kakao_high", "kakao_normal"]

# VideoMAE는 모델이 하나지만, 형식 통일을 위해 리스트로 관리하거나 변수로 지정
MODEL_TYPE = "videomae-base" 

SEQUENCE_LENGTH = 16
IMG_SIZE = 224
BATCH_SIZE = 4 # VRAM 안전값
EPOCHS = 10
SEED = 42

def clean_memory():
    gc.collect()
    torch.cuda.empty_cache()

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class VideoSequenceDataset(Dataset):
    def __init__(self, samples, transform=None, is_train=True):
        self.samples, self.transform, self.is_train = samples, transform, is_train
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total > SEQUENCE_LENGTH:
            start = random.randint(0, total - SEQUENCE_LENGTH) if self.is_train else (total - SEQUENCE_LENGTH) // 2
        else: start = 0
            
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames = []
        apply_hflip = (random.random() > 0.5) if self.is_train else False
        
        for _ in range(SEQUENCE_LENGTH):
            ret, frame = cap.read()
            if not ret: frame = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            else: frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = transforms.ToPILImage()(frame)
            pil_img = transforms.Resize((IMG_SIZE, IMG_SIZE))(pil_img)
            if apply_hflip: pil_img = TF.hflip(pil_img)
            frames.append(self.transform(pil_img))
        cap.release()
        return torch.stack(frames), label

def train_videomae_session(domain):
    seed_everything(SEED)
    clean_memory()
    
    data_path = BASE_DATA_DIR / domain / "train"
    
    # [수정 완료] 형식을 spatial/temporal과 완벽하게 통일!
    # 예: models/videomae_raw_videomae-base
    save_dir = SAVE_MODEL_DIR / f"videomae_{domain}_{MODEL_TYPE}"
    save_dir.mkdir(parents=True, exist_ok=True)

    all_samples = []
    for sub, lab in [("real", 0), ("fake", 1)]:
        d = data_path / sub
        if d.exists(): all_samples += [(p, lab) for p in d.glob("*.mp4")]

    if not all_samples: return

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    labels = [s[1] for s in all_samples]
    
    # ImageNet Norm 사용
    base_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for fold, (train_idx, val_idx) in enumerate(skf.split(all_samples, labels)):
        print(f"\n🔄 [{domain.upper()}-{MODEL_TYPE.upper()}] Fold {fold+1}/5")
        clean_memory()

        train_loader = DataLoader(
            VideoSequenceDataset([all_samples[i] for i in train_idx], base_tf, True), 
            batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
        )
        val_loader = DataLoader(
            VideoSequenceDataset([all_samples[i] for i in val_idx], base_tf, False), 
            batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
        )
        
        # 진짜 VideoMAE 모델 로드
        model = VideoMAEForVideoClassification.from_pretrained(
            "MCG-NJU/videomae-base", 
            num_labels=2, 
            ignore_mismatched_sizes=True
        ).cuda()
        
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
        criterion = nn.CrossEntropyLoss().cuda()
        best_acc = 0.0

        for epoch in range(EPOCHS):
            model.train()
            for inputs, targets in tqdm(train_loader, desc=f"Ep {epoch+1}", leave=False):
                inputs, targets = inputs.cuda(), targets.cuda()
                optimizer.zero_grad()
                # VideoMAE는 pixel_values 인자를 사용
                outputs = model(pixel_values=inputs)
                loss = criterion(outputs.logits, targets)
                loss.backward(); optimizer.step()

            model.eval(); correct = 0; total = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.cuda(), targets.cuda()
                    outputs = model(pixel_values=inputs)
                    _, pred = torch.max(outputs.logits, 1)
                    total += targets.size(0); correct += (pred == targets).sum().item()
            
            acc = correct / total
            print(f"   📅 Ep {epoch+1}: Val Acc {acc:.4f}")
            
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), save_dir / f"best_fold{fold+1}.pth")
        
        del model; clean_memory()

if __name__ == "__main__":
    print("🚀 [Master Train] VideoMAE 통합 학습 시작")
    for domain in TARGET_DOMAINS:
        train_videomae_session(domain)