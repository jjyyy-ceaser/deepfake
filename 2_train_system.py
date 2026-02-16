import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import os
import json
import glob
import sys
from tqdm import tqdm
from utils import DeepfakeDataset, get_model
from torchvision import transforms

# =========================================================
# ⚙️ [16H Quality Mode] 실험 설정
# =========================================================
DATASETS = ["dataset_A_pure", "dataset_C_worst", "dataset_B_mixed"] 
MODELS = ["xception", "convnext", "swin", "r3d", "r2plus1d", "videomae_v2"]

# 품질을 위해 배치는 32로 고정, VRAM 보호를 위해 MAE는 8 유지
BATCH_LIST = [32] 
MAE_BATCH_LIST = [8]
LR_LIST = [1e-4, 5e-5, 1e-5]

# 시스템 프리징 방지를 위한 최적의 워커 수
NUM_WORKERS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# 🛠️ 함수 정의 (기존 로직 유지)
# =========================================================

def get_data(dataset_name):
    base = os.path.join("dataset", "final_datasets", dataset_name)
    real = glob.glob(os.path.join(base, "real", "*"))
    fake = glob.glob(os.path.join(base, "fake", "*"))
    
    if len(real) == 0 or len(fake) == 0:
        print(f"⚠️ 경고: {dataset_name} 폴더 비어있음/경로 오류.")
        return [], []

    files = real + fake
    labels = [0]*len(real) + [1]*len(fake)
    return files, labels

def train_epoch(model, loader, criterion, optimizer, is_mae, desc=""):
    model.train()
    running_loss = 0.0
    loop = tqdm(loader, desc=desc, leave=False, file=sys.stdout)
    
    for x, y in loop:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        
        if is_mae: 
            out = model(pixel_values=x.permute(0,2,1,3,4)).logits
        else: 
            out = model(x)
            
        loss = criterion(out, y)
        if torch.isnan(loss):
            optimizer.zero_grad()
            continue

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        loop.set_postfix(loss=f"{loss.item():.4f}")
    
    return running_loss / len(loader)

def evaluate(model, loader, is_mae):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            if is_mae: out = model(pixel_values=x.permute(0,2,1,3,4)).logits
            else: out = model(x)
            probs = torch.softmax(out, 1)[:, 1]
            preds.extend(probs.cpu().tolist())
            trues.extend(y.cpu().tolist())
    try:
        if len(set(trues)) < 2: return 0.5 
        return roc_auc_score(trues, preds)
    except: return 0.5

# =========================================================
# 🚀 메인 실행 루프 (시간 최적화 적용)
# =========================================================
def run_experiment():
    print("="*60)
    print(f"🚀 [16H Mode Started] Device: {DEVICE}")
    print(f"🔥 Target: {torch.cuda.get_device_name(0)}")
    print("="*60)

    tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    for ds_name in DATASETS:
        print(f"\n🌍 [[ Dataset: {ds_name} ]]")
        files, labels = get_data(ds_name)
        if not files: continue

        class_weights = torch.tensor([1.0, 300.0/135.0]).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for model_name in MODELS:
            print(f"\n  🔹 Model: {model_name}")
            save_dir = os.path.join("checkpoints", ds_name, model_name)
            os.makedirs(save_dir, exist_ok=True)
            
            is_mae = 'videomae' in model_name
            model_type = 'temporal' if any(x in model_name for x in ['r3d', 'r2', 'mae']) else 'spatial'
            bs = MAE_BATCH_LIST[0] if is_mae else BATCH_LIST[0]

            # [Step 1] Grid Search - 1 Epoch로 단축 (시간 절약의 핵심)
            print(f"    🔎 Grid Search (1 Epoch)...", flush=True)
            best_auc = -1
            best_params = {'lr': 1e-4, 'bs': bs} 
            
            train_idx, val_idx = next(skf.split(files, labels))
            ds_train = DeepfakeDataset([files[i] for i in train_idx], [labels[i] for i in train_idx], model_type, tf)
            ds_val = DeepfakeDataset([files[i] for i in val_idx], [labels[i] for i in val_idx], model_type, tf)
            
            for lr in LR_LIST:
                # 메모리 안정성을 위해 prefetch_factor 조절
                loader_tr = DataLoader(ds_train, batch_size=bs, shuffle=True, num_workers=NUM_WORKERS, 
                                       pin_memory=True, prefetch_factor=1, persistent_workers=False)
                loader_val = DataLoader(ds_val, batch_size=bs, num_workers=NUM_WORKERS, pin_memory=True)
                
                try:
                    model = get_model(model_name, DEVICE)
                    opt = optim.AdamW(model.parameters(), lr=lr)
                    # 1 에폭만 학습하여 경향성 파악
                    train_epoch(model, loader_tr, criterion, opt, is_mae, desc=f"GS LR={lr}")
                    
                    current_val_auc = evaluate(model, loader_val, is_mae)
                    print(f"      👉 LR={lr} -> Val AUC: {current_val_auc:.4f}")
                    
                    if current_val_auc > best_auc:
                        best_auc = current_val_auc
                        best_params = {'lr': lr, 'bs': bs}
                except Exception as e:
                    print(f"      ❌ GS Error: {e}")
            
            # [Step 2] Main 5-Fold Training - 7 Epoch (품질 수렴 지점)
            print(f"    🚀 Starting 5-Fold CV (7 Epochs)...", flush=True)
            final_lr = best_params['lr']
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(files, labels)):
                ds_train = DeepfakeDataset([files[i] for i in train_idx], [labels[i] for i in train_idx], model_type, tf)
                ds_val = DeepfakeDataset([files[i] for i in val_idx], [labels[i] for i in val_idx], model_type, tf)
                
                loader_tr = DataLoader(ds_train, batch_size=bs, shuffle=True, num_workers=NUM_WORKERS, 
                                       pin_memory=True, prefetch_factor=1, persistent_workers=False)
                loader_val = DataLoader(ds_val, batch_size=bs, num_workers=NUM_WORKERS, pin_memory=True)
                
                model = get_model(model_name, DEVICE)
                opt = optim.AdamW(model.parameters(), lr=final_lr)
                best_fold_auc = 0.0
                
                for ep in range(7): # 10에서 7로 조정하여 시간 사수
                    desc_text = f"Fold {fold+1}/5 | Ep {ep+1}/7"
                    try:
                        train_epoch(model, loader_tr, criterion, opt, is_mae, desc=desc_text)
                        val_auc = evaluate(model, loader_val, is_mae)
                        if val_auc > best_fold_auc:
                            best_fold_auc = val_auc
                            torch.save(model.state_dict(), os.path.join(save_dir, f"fold{fold}_best.pth"))
                    except Exception as e: continue

                print(f"      🏆 Fold {fold} Done. Best AUC: {best_fold_auc:.4f}")

if __name__ == "__main__":
    run_experiment()