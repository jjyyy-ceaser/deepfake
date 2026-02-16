import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import os
import json
import glob
from utils import DeepfakeDataset, get_model
from torchvision import transforms

# === 설정 ===
DATASETS = ["dataset_A_pure", "dataset_C_worst", "dataset_B_mixed"] # 순서: A -> C -> B
MODELS = ["xception", "convnext", "swin", "r3d", "r2plus1d", "videomae_v2"]
LR_LIST = [1e-4, 5e-5]
BATCH_LIST = [4, 8] # VideoMAE는 내부에서 2, 4로 조절
DEVICE = torch.device("cuda")

def get_data(dataset_name):
    base = os.path.join("dataset", "final_datasets", dataset_name)
    real = glob.glob(os.path.join(base, "real", "*"))
    fake = glob.glob(os.path.join(base, "fake", "*"))
    files = real + fake
    labels = [0]*len(real) + [1]*len(fake)
    return files, labels

def train_epoch(model, loader, criterion, optimizer, is_mae):
    model.train()
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        if is_mae: 
            out = model(pixel_values=x.permute(0,2,1,3,4)).logits
        else: 
            out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

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
    try: return roc_auc_score(trues, preds)
    except: return 0.5

def run_experiment():
    tf = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224,224)), transforms.ToTensor()])

    for ds_name in DATASETS:
        print(f"\n🌍 Dataset: {ds_name}")
        files, labels = get_data(ds_name)
        
        # 불균형 가중치 계산 (Real 300 : Fake 165)
        # Weight for Fake = 300 / 165 ≈ 1.82
        weights = torch.tensor([1.0, 300.0/165.0]).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=weights)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for model_name in MODELS:
            print(f"  >>> Model: {model_name}")
            save_dir = os.path.join("checkpoints", ds_name, model_name)
            os.makedirs(save_dir, exist_ok=True)
            
            model_type = 'temporal' if any(x in model_name for x in ['r3d', 'r2', 'mae']) else 'spatial'
            is_mae = 'mae' in model_name
            
            # --- [Step 1] Grid Search (Fold 0 only) ---
            print("    🔎 Grid Search...", end="", flush=True)
            best_auc = -1
            best_params = {}
            train_idx, val_idx = next(skf.split(files, labels)) # Fold 0
            
            ds_train = DeepfakeDataset([files[i] for i in train_idx], [labels[i] for i in train_idx], model_type, tf)
            ds_val = DeepfakeDataset([files[i] for i in val_idx], [labels[i] for i in val_idx], model_type, tf)
            
            current_batches = [2, 4] if is_mae else BATCH_LIST
            
            for lr in LR_LIST:
                for bs in current_batches:
                    loader_tr = DataLoader(ds_train, batch_size=bs, shuffle=True)
                    loader_val = DataLoader(ds_val, batch_size=bs)
                    model = get_model(model_name, DEVICE)
                    opt = optim.AdamW(model.parameters(), lr=lr)
                    
                    # 3 Epoch Short Run
                    for _ in range(3): train_epoch(model, loader_tr, criterion, opt, is_mae)
                    auc = evaluate(model, loader_val, is_mae)
                    
                    if auc > best_auc:
                        best_auc = auc
                        best_params = {'lr': lr, 'bs': bs}
            
            print(f" Done. Best: {best_params}")
            with open(os.path.join(save_dir, "best_params.json"), "w") as f: json.dump(best_params, f)

            # --- [Step 2] Main 5-Fold Training ---
            print("    🚀 5-Fold Training...")
            lr, bs = best_params['lr'], best_params['bs']
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(files, labels)):
                print(f"      Fold {fold}...", end="")
                ds_train = DeepfakeDataset([files[i] for i in train_idx], [labels[i] for i in train_idx], model_type, tf)
                ds_val = DeepfakeDataset([files[i] for i in val_idx], [labels[i] for i in val_idx], model_type, tf)
                
                loader_tr = DataLoader(ds_train, batch_size=bs, shuffle=True)
                loader_val = DataLoader(ds_val, batch_size=bs)
                
                model = get_model(model_name, DEVICE)
                opt = optim.AdamW(model.parameters(), lr=lr)
                
                best_fold_auc = 0
                for ep in range(10): # 10 Epochs
                    train_epoch(model, loader_tr, criterion, opt, is_mae)
                    val_auc = evaluate(model, loader_val, is_mae)
                    if val_auc > best_fold_auc:
                        best_fold_auc = val_auc
                        torch.save(model.state_dict(), os.path.join(save_dir, f"fold{fold}_best.pth"))
                print("Done.")

if __name__ == "__main__":
    run_experiment()