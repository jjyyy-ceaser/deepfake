import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import os
import glob
import sys
import gc
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import DeepfakeDataset, get_model
from torchvision import transforms
from torch.cuda.amp import GradScaler, autocast # 최신 pytorch라면 torch.amp 사용 권장

# ==========================================
# ⚙️ 본 학습(Main Training) 설정
# ==========================================
TARGET_DATASETS = ["dataset_B_mixed"] # 필요시 ["dataset_A_pure", "dataset_C_worst"] 추가 가능
NUM_EPOCHS = 30       # 충분한 학습을 위해 30으로 설정 (Early Stopping 있음)
PATIENCE = 5          # 5번 연속 성능 향상 없으면 조기 종료
BATCH_SIZE_SPATIAL = 32
BATCH_SIZE_TEMPORAL = 16 # VRAM 안전값
NUM_WORKERS = 2       # 윈도우 환경 충돌 방지
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🏆 Grid Search로 찾은 최적의 LR (Learning Rates)
BEST_PARAMS = {
    "xception": 5e-5,   
    "convnext": 1e-4,   
    "swin": 5e-5,       
    "r3d": 1e-4,        # Grid Search 결과 반영 (0.96)
    "r2plus1d": 5e-5    # Grid Search 결과 반영 (0.93)
}

# 학습할 모델 목록
MODELS_TO_TRAIN = ["r3d", "r2plus1d", "xception", "convnext", "swin"]

def get_transforms(model_name):
    """모델 타입에 따른 전처리(Normalization) 분리"""
    if model_name in ["r3d", "r2plus1d", "videomae"]:
        # Temporal 모델 전용 (Kinetics-400 통계값 등)
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
        ])
    else:
        # Spatial 모델 전용 (ImageNet 통계값)
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def train_one_fold(fold_idx, train_files, train_labels, val_files, val_labels, model_name, dataset_name):
    print(f"\n🚀 [Fold {fold_idx+1}] Model: {model_name} | Dataset: {dataset_name}")
    
    # 설정 가져오기
    lr = BEST_PARAMS.get(model_name, 1e-4) # 없으면 기본값 1e-4
    is_temporal = model_name in ["r3d", "r2plus1d", "videomae"]
    batch_size = BATCH_SIZE_TEMPORAL if is_temporal else BATCH_SIZE_SPATIAL
    tf = get_transforms(model_name)
    
    # 데이터셋 & 로더
    ds_tr = DeepfakeDataset(train_files, train_labels, 'temporal' if is_temporal else 'spatial', tf)
    ds_val = DeepfakeDataset(val_files, val_labels, 'temporal' if is_temporal else 'spatial', tf)
    
    l_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    l_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    
    # 모델 & 최적화
    model = get_model(model_name, DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    scaler = GradScaler()
    
    # 저장 경로
    save_dir = os.path.join("checkpoints", dataset_name, model_name)
    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, f"best_fold{fold_idx+1}.pth")
    
    best_auc = 0.0
    patience_counter = 0
    
    # === Epoch Loop ===
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0
        loop = tqdm(l_tr, desc=f"  Ep {epoch+1}/{NUM_EPOCHS}", leave=False, ncols=80)
        
        for x, y in loop:
            x, y = x.to(DEVICE), y.to(DEVICE, dtype=torch.long)
            optimizer.zero_grad()
            
            with autocast():
                # 차원 보정 (3D 모델은 permute 필요)
                out = model(x.permute(0, 2, 1, 3, 4)) if is_temporal else model(x)
                loss = criterion(out, y)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            
        # 검증 (Validation)
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for vx, vy in l_val:
                vx = vx.to(DEVICE)
                vout = model(vx.permute(0, 2, 1, 3, 4)) if is_temporal else model(vx)
                preds.extend(torch.softmax(vout, 1)[:, 1].cpu().tolist())
                trues.extend(vy.tolist())
        
        try:
            val_auc = roc_auc_score(trues, preds)
        except:
            val_auc = 0.5
            
        print(f"    ✅ Ep {epoch+1} | Loss: {train_loss/len(l_tr):.4f} | Val AUC: {val_auc:.4f}")
        
        # Best Model 저장 & Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
            print(f"      💾 Best Saved! ({best_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"      🛑 Early Stopping (No improve for {PATIENCE} epochs)")
                break
                
    # 메모리 정리
    del model, optimizer, scaler   
    torch.cuda.empty_cache()
    gc.collect()
    
    return best_auc

# === Main Execution ===
if __name__ == "__main__":
    # 멀티프로세싱 안전장치
    import torch.multiprocessing as mp
    try: mp.set_start_method('spawn', force=True)
    except: pass

    results_log = []

    for dataset_name in TARGET_DATASETS:
        print(f"\n\n{'='*40}\n🎯 Target Dataset: {dataset_name}\n{'='*40}")
        
        # 데이터 파일 로드
        base_path = os.path.join("dataset", "final_datasets", dataset_name)
        real_files = glob.glob(os.path.join(base_path, "real", "*"))
        fake_files = glob.glob(os.path.join(base_path, "fake", "*"))
        
        if not real_files:
            print(f"❌ 데이터 없음: {base_path}")
            continue
            
        all_files = real_files + fake_files
        all_labels = [0]*len(real_files) + [1]*len(fake_files)
        
        # 5-Fold Setting
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for model_name in MODELS_TO_TRAIN:
            fold_scores = []
            
            for fold, (t_idx, v_idx) in enumerate(skf.split(all_files, all_labels)):
                # 각 폴드별 학습 실행
                tr_files = [all_files[i] for i in t_idx]
                tr_labels = [all_labels[i] for i in t_idx]
                val_files = [all_files[i] for i in v_idx]
                val_labels = [all_labels[i] for i in v_idx]
                
                score = train_one_fold(fold, tr_files, tr_labels, val_files, val_labels, model_name, dataset_name)
                fold_scores.append(score)
            
            # 최종 결과 기록
            avg_score = np.mean(fold_scores)
            print(f"\n🏆 {model_name} on {dataset_name} | Avg AUC: {avg_score:.4f} {fold_scores}")
            results_log.append({"Dataset": dataset_name, "Model": model_name, "Avg_AUC": avg_score, "Folds": fold_scores})
            
            # 모델 간 메모리 정리
            gc.collect()

    # 최종 리포트 저장
    pd.DataFrame(results_log).to_csv("final_training_results.csv", index=False)
    print("\n🎉 모든 본 학습이 완료되었습니다! 'final_training_results.csv'를 확인하세요.")