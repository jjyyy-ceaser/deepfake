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
# ⚙️ [Clean UI Mode] 실험 설정
# =========================================================
DATASETS = ["dataset_A_pure", "dataset_C_worst", "dataset_B_mixed"] 
MODELS = ["xception", "convnext", "swin", "r3d", "r2plus1d", "videomae_v2"]

BATCH_LIST = [32] 
MAE_BATCH_LIST = [8]
LR_LIST = [1e-4, 5e-5, 1e-5]

NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_data(dataset_name):
    base = os.path.join("dataset", "final_datasets", dataset_name)
    real = glob.glob(os.path.join(base, "real", "*"))
    fake = glob.glob(os.path.join(base, "fake", "*"))
    if not real or not fake: return [], []
    return real + fake, [0]*len(real) + [1]*len(fake)

def train_epoch(model, loader, criterion, optimizer, is_mae, desc=""):
    model.train()
    running_loss = 0.0
    # [핵심 수정] leave=False: 완료되면 진행바를 지워서 줄내림 방지
    # ncols=80: 진행바 길이를 고정해서 깨짐 방지
    loop = tqdm(loader, desc=desc, leave=False, ncols=80, file=sys.stdout)
    
    for x, y in loop:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        if is_mae: out = model(pixel_values=x.permute(0,2,1,3,4)).logits
        else: out = model(x)
        
        loss = criterion(out, y)
        if torch.isnan(loss): continue
        
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
            preds.extend(torch.softmax(out, 1)[:, 1].cpu().tolist())
            trues.extend(y.cpu().tolist())
    try: return roc_auc_score(trues, preds) if len(set(trues)) > 1 else 0.5
    except: return 0.5

def run_experiment():
    print(f"🚀 [System Started] Device: {DEVICE}")
    tf = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224,224)), transforms.ToTensor()])

    for ds_name in DATASETS:
        print(f"\n🌍 [[ Dataset: {ds_name} ]]")
        files, labels = get_data(ds_name)
        if not files: continue

        class_weights = torch.tensor([1.0, 300.0/135.0]).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for model_name in MODELS:
            print(f"  🔹 Model: {model_name}") # 최소한의 로그만 출력
            save_dir = os.path.join("checkpoints", ds_name, model_name)
            os.makedirs(save_dir, exist_ok=True)
            
            is_mae = 'videomae' in model_name
            model_type = 'temporal' if any(x in model_name for x in ['r3d', 'r2', 'mae']) else 'spatial'
            bs = MAE_BATCH_LIST[0] if is_mae else BATCH_LIST[0]

            # [Step 1] Grid Search (1 Epoch)
            # print("    🔎 Grid Search...") -> 너무 잡다한 로그는 생략
            best_auc = -1
            best_params = {'lr': 1e-4, 'bs': bs}
            
            train_idx, val_idx = next(skf.split(files, labels))
            ds_tr = DeepfakeDataset([files[i] for i in train_idx], [labels[i] for i in train_idx], model_type, tf)
            ds_val = DeepfakeDataset([files[i] for i in val_idx], [labels[i] for i in val_idx], model_type, tf)
            
            for lr in LR_LIST:
                l_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=1, persistent_workers=False)
                l_val = DataLoader(ds_val, batch_size=bs, num_workers=NUM_WORKERS, pin_memory=True)
                
                try:
                    model = get_model(model_name, DEVICE)
                    opt = optim.AdamW(model.parameters(), lr=lr)
                    train_epoch(model, l_tr, criterion, opt, is_mae, desc=f"GS LR={lr}")
                    val_auc = evaluate(model, l_val, is_mae)
                    
                    if val_auc > best_auc:
                        best_auc = val_auc
                        best_params = {'lr': lr, 'bs': bs}
                except: continue

            with open(os.path.join(save_dir, "best_params.json"), "w") as f:
                json.dump(best_params, f)

            # [Step 2] Main Training
            print(f"    🚀 Training (LR={best_params['lr']})...")
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(files, labels)):
                ds_tr = DeepfakeDataset([files[i] for i in train_idx], [labels[i] for i in train_idx], model_type, tf)
                ds_val = DeepfakeDataset([files[i] for i in val_idx], [labels[i] for i in val_idx], model_type, tf)
                
                l_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=1, persistent_workers=False)
                l_val = DataLoader(ds_val, batch_size=bs, num_workers=NUM_WORKERS, pin_memory=True)
                
                model = get_model(model_name, DEVICE)
                opt = optim.AdamW(model.parameters(), lr=best_params['lr'])
                best_fold_auc = 0.0
                
                for ep in range(7):
                    try:
                        loss = train_epoch(model, l_tr, criterion, opt, is_mae, desc=f"F{fold+1} E{ep+1}")
                        auc = evaluate(model, l_val, is_mae)
                        
                        if auc > best_fold_auc:
                            best_fold_auc = auc
                            torch.save(model.state_dict(), os.path.join(save_dir, f"fold{fold}_best.pth"))
                        
                        # [자동 저장] 이 파일만 열어보면 모든 결과가 다 있습니다.
                        with open(os.path.join(save_dir, "summary.txt"), "a") as f:
                            f.write(f"Fold {fold+1} | Ep {ep+1} | Loss: {loss:.4f} | AUC: {auc:.4f}\n")
                            
                    except: continue

if __name__ == "__main__":
    run_experiment()