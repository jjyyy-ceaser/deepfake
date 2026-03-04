import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torchvision.transforms.functional as TF
import cv2
import os
import gc
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import timm
from transformers import VideoMAEForVideoClassification
from pathlib import Path
from datetime import datetime

# ======================================================
# [설정] 경로 및 타겟 지정
# ======================================================
BASE_DATA_DIR = Path(r"C:\Users\leejy\Desktop\test_experiment\dataset\final_dataset")
MODEL_SAVE_DIR = Path(r"C:\Users\leejy\Desktop\test_experiment\models")
REPORT_SAVE_DIR = Path(r"C:\Users\leejy\Desktop\test_experiment\results")
REPORT_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# [도메인 순서] Raw -> SNS
TARGET_DOMAINS = ["raw", "instagram", "kakao_high", "kakao_normal", "youtube"]

# [모델 리스트]
MODELS_CONFIG = {
    "Spatial": ["xception", "convnext", "swin"],
    "Temporal": ["r3d", "r2plus1d"],
    "VideoMAE": ["videomae-base"]
}

# [하이퍼파라미터]
IMG_SIZE_SPATIAL = 224
IMG_SIZE_TEMPORAL = 112
IMG_SIZE_VIDEOMAE = 224
SEQ_LENGTH = 16
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================================================
# [유틸] 메모리 정리
# ======================================================
def clean_memory():
    gc.collect()
    torch.cuda.empty_cache()

# ======================================================
# [Transform]
# ======================================================
def get_transforms(mode):
    if mode == 'Temporal':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.43216, 0.394666, 0.37645], [0.22803, 0.22145, 0.216989])
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

# ======================================================
# [데이터셋]
# ======================================================
class TestVideoDataset(Dataset):
    def __init__(self, samples, transform=None, mode='Spatial', seq_len=16):
        self.samples = samples
        self.transform = transform
        self.mode = mode
        self.seq_len = seq_len

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frames = []
        if self.mode == 'Spatial':
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
            ret, frame = cap.read()
            cap.release()
            if not ret: frame = np.zeros((IMG_SIZE_SPATIAL, IMG_SIZE_SPATIAL, 3), dtype=np.uint8)
            else: frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return self.transform(frame), label
        
        else:
            start = max(0, (total_frames - self.seq_len) // 2)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            target_size = IMG_SIZE_VIDEOMAE if self.mode == 'VideoMAE' else IMG_SIZE_TEMPORAL
            
            for _ in range(self.seq_len):
                ret, frame = cap.read()
                if not ret: frame = np.zeros((target_size, target_size, 3), dtype=np.uint8)
                else: frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = transforms.ToPILImage()(frame)
                pil_img = transforms.Resize((target_size, target_size))(pil_img)
                frames.append(self.transform(pil_img))
            cap.release()
            video_tensor = torch.stack(frames)
            if self.mode == 'Temporal': video_tensor = video_tensor.permute(1, 0, 2, 3)
            return video_tensor, label

# ======================================================
# [모델 구조]
# ======================================================
def get_model_structure(category, model_name):
    if category == 'Spatial':
        if model_name == 'xception': return timm.create_model('xception', pretrained=False, num_classes=2)
        elif model_name == 'convnext': return timm.create_model('convnext_tiny', pretrained=False, num_classes=2)
        elif model_name == 'swin': return timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=2)
    elif category == 'Temporal':
        if model_name == 'r3d': m = models.video.r3d_18(weights=None); m.fc = nn.Linear(m.fc.in_features, 2); return m
        elif model_name == 'r2plus1d': m = models.video.r2plus1d_18(weights=None); m.fc = nn.Linear(m.fc.in_features, 2); return m
    elif category == 'VideoMAE':
        return VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base", num_labels=2, ignore_mismatched_sizes=True)
    return None

# ======================================================
# [메인] 실행 루프
# ======================================================
def run_cross_domain_test():
    clean_memory()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORT_SAVE_DIR / f"Final_Report_{timestamp}.txt"
    excel_path = REPORT_SAVE_DIR / f"Final_Result_{timestamp}.xlsx"
    
    report_file = open(report_path, "w", encoding="utf-8")
    
    def write_log(text):
        print(text)
        report_file.write(text + "\n")

    write_log(f"🚀 [Final Check] Deepfake Cross-Domain Evaluation Started: {timestamp}")
    write_log(f"📌 Config: Batch={BATCH_SIZE}, Device={DEVICE}")
    
    final_results = []

    # 1. 학습 도메인 (Train Domain)
    for train_domain in TARGET_DOMAINS:
        write_log(f"\n=======================================================")
        write_log(f"🏋️ Train Domain: [{train_domain.upper()}] Evaluation Start")
        write_log(f"=======================================================")

        for category, models_list in MODELS_CONFIG.items():
            for model_name in models_list:
                write_log(f"\n  [Model: {model_name}] (Category: {category})")
                
                # -------------------------------------------------
                # 5-Fold 모델 로드
                # -------------------------------------------------
                ensemble_models = []
                cat_lower = category.lower() 
                if cat_lower == 'videomae': folder_name = f"videomae_{train_domain}_{model_name}"
                else: folder_name = f"{cat_lower}_{train_domain}_{model_name}"
                
                model_dir = MODEL_SAVE_DIR / folder_name
                
                loaded_cnt = 0
                for fold in range(1, 6):
                    pth = model_dir / f"best_fold{fold}.pth"
                    if pth.exists():
                        m = get_model_structure(category, model_name)
                        m.load_state_dict(torch.load(pth))
                        m.to(DEVICE); m.eval()
                        ensemble_models.append(m)
                        loaded_cnt += 1
                
                if loaded_cnt == 0:
                    write_log(f"  ❌ Skipping: No checkpoints found in {model_dir}")
                    continue
                
                write_log(f"  ✅ Loaded {loaded_cnt} Folds (Ensemble Ready)")

                # -------------------------------------------------
                # 2. 테스트 도메인 순회 (Test Domain)
                # -------------------------------------------------
                for test_domain in TARGET_DOMAINS:
                    test_dir = BASE_DATA_DIR / test_domain / "test"
                    samples = []
                    for sub, lab in [("real", 0), ("fake", 1)]:
                        d = test_dir / sub
                        if d.exists(): samples += [(p, lab) for p in d.glob("*.mp4")]
                    
                    if not samples: 
                        write_log(f"    ⚠️ No data for {test_domain}, skipping...")
                        continue

                    # 데이터 로더 준비
                    tf = get_transforms(category)
                    ds = TestVideoDataset(samples, tf, mode=category)
                    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

                    # 인퍼런스
                    all_probs, all_targets = [], []
                    with torch.no_grad():
                        for inputs, targets in tqdm(loader, desc=f"    Testing on {test_domain}", leave=False):
                            inputs = inputs.to(DEVICE)
                            batch_probs = torch.zeros(inputs.size(0), 2).to(DEVICE)
                            for m in ensemble_models:
                                if category == 'VideoMAE': outputs = m(pixel_values=inputs).logits
                                else: outputs = m(inputs)
                                batch_probs += torch.softmax(outputs, dim=1)
                            batch_probs /= loaded_cnt
                            all_probs.extend(batch_probs[:, 1].cpu().tolist())
                            all_targets.extend(targets.tolist())

                    # [수정됨] 5대 지표 계산
                    preds = [1 if p >= 0.5 else 0 for p in all_probs]
                    acc = accuracy_score(all_targets, preds)
                    auc = roc_auc_score(all_targets, all_probs)
                    pre = precision_score(all_targets, preds, zero_division=0)
                    rec = recall_score(all_targets, preds, zero_division=0)
                    f1 = f1_score(all_targets, preds, zero_division=0)

                    # [수정됨] 상세 로그 출력
                    write_log(f"    ▶ Test on [{test_domain.upper()}]")
                    write_log(f"      ---------------------------------------------------------------")
                    write_log(f"      🎯 Result: Acc: {acc:.4f} | AUC: {auc:.4f} | Pre: {pre:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")
                    write_log(f"      ---------------------------------------------------------------")
                    
                    final_results.append({
                        "Category": category,
                        "Model": model_name,
                        "Train_Domain": train_domain,
                        "Test_Domain": test_domain,
                        "ACC": acc,
                        "AUC": auc,
                        "Precision": pre,
                        "Recall": rec,
                        "F1-Score": f1
                    })
                
                del ensemble_models
                clean_memory()

    report_file.close()
    if final_results:
        df = pd.DataFrame(final_results)
        cols = ["Category", "Model", "Train_Domain", "Test_Domain", "ACC", "AUC", "Precision", "Recall", "F1-Score"]
        df = df[cols]
        df.to_excel(excel_path, index=False)
        print(f"\n🎉 [COMPLETE] Report & Excel Saved!")
        print(f"📄 Report: {report_path}")
        print(f"📊 Excel : {excel_path}")
    else:
        print("❌ 저장할 결과가 없습니다.")

if __name__ == "__main__":
    run_cross_domain_test()