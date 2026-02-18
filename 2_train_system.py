import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import os, json, glob, sys
from tqdm import tqdm
from utils import DeepfakeDataset, get_model
from torchvision import transforms

# RTX 4070 SUPER + Ryzen 7500F 최적화 설정
DATASETS = ["dataset_A_pure", "dataset_C_worst", "dataset_B_mixed"]
MODELS = ["xception", "convnext", "swin", "r3d", "r2plus1d", "videomae_v2"]
LR_LIST = [1e-4, 5e-5, 1e-5] 

# [수정] 다른 프로그램 종료 시 4명이 가장 효율적입니다.
NUM_WORKERS = 4 
DEVICE = torch.device("cuda")

def run_experiment():
    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    
    for ds_name in DATASETS:
        print(f"\n🌍 Dataset: {ds_name}")
        base = os.path.join("dataset", "final_datasets", ds_name)
        real = glob.glob(os.path.join(base, "real", "*"))
        fake = glob.glob(os.path.join(base, "fake", "*"))
        files, labels = real + fake, [0]*len(real) + [1]*len(fake)
        if not files: continue

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for model_name in MODELS:
            save_dir = os.path.join("checkpoints", ds_name, model_name)
            os.makedirs(save_dir, exist_ok=True)
            
            is_mae = 'mae' in model_name.lower()
            is_temp = any(x in model_name.lower() for x in ['r3d', 'r2', 'mae'])
            
            # OOM 방지를 위한 배치 사이즈 최적화
            bs = 8 if is_mae else (32 if is_temp else 64)
            print(f"  🔹 Model: {model_name} | Batch: {bs} | Workers: {NUM_WORKERS}")

            # [Step 1] Grid Search
            best_lr = 1e-4
            best_gs_auc = 0.0
            t_idx, v_idx = next(skf.split(files, labels))
            
            ds_gs_tr = DeepfakeDataset([files[i] for i in t_idx], [labels[i] for i in t_idx], 'temporal' if is_temp else 'spatial', tf)
            ds_gs_val = DeepfakeDataset([files[i] for i in v_idx], [labels[i] for i in v_idx], 'temporal' if is_temp else 'spatial', tf)
            
            for lr in LR_LIST:
                print(f"    🔍 GS Testing LR={lr}...")
                model = get_model(model_name, DEVICE)
                optimizer = optim.AdamW(model.parameters(), lr=lr)
                criterion = nn.CrossEntropyLoss().to(DEVICE)
                
                # 고속 로딩 설정
                l_gs = DataLoader(ds_gs_tr, batch_size=bs, shuffle=True, 
                                num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
                
                model.train()
                for x, y in tqdm(l_gs, desc=f"GS LR={lr}", leave=False, ncols=80):
                    try:
                        x, y = x.to(DEVICE), y.to(DEVICE, dtype=torch.long)
                        optimizer.zero_grad()
                        if is_mae: out = model(pixel_values=x).logits
                        elif is_temp: out = model(x.permute(0, 2, 1, 3, 4))
                        else: out = model(x)
                        criterion(out, y).backward(); optimizer.step()
                    except: continue
                
                # 검증 로직
                model.eval()
                preds, trues = [], []
                l_vs = DataLoader(ds_gs_val, batch_size=bs, num_workers=NUM_WORKERS)
                with torch.no_grad():
                    for vx, vy in l_vs:
                        try:
                            vx = vx.to(DEVICE)
                            if is_mae: vout = model(pixel_values=vx).logits
                            elif is_temp: vout = model(vx.permute(0, 2, 1, 3, 4))
                            else: vout = model(vx)
                            preds.extend(torch.softmax(vout, 1)[:, 1].cpu().tolist())
                            trues.extend(vy.tolist())
                        except: continue
                
                auc = roc_auc_score(trues, preds) if len(set(trues)) > 1 else 0.5
                if auc > best_gs_auc: best_gs_auc = auc; best_lr = lr
            
            print(f"    ✅ Best LR: {best_lr}")

            # [Step 2] Main Training
            for fold, (t_idx, v_idx) in enumerate(skf.split(files, labels)):
                model = get_model(model_name, DEVICE)
                optimizer = optim.AdamW(model.parameters(), lr=best_lr)
                ds_tr = DeepfakeDataset([files[i] for i in t_idx], [labels[i] for i in t_idx], 'temporal' if is_temp else 'spatial', tf)
                l_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True, 
                                num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
                
                for ep in range(7):
                    model.train()
                    loop = tqdm(l_tr, desc=f"F{fold+1} E{ep+1}", leave=False, ncols=80, file=sys.stdout)
                    for x, y in loop:
                        try:
                            x, y = x.to(DEVICE), y.to(DEVICE, dtype=torch.long)
                            optimizer.zero_grad()
                            if is_mae: out = model(pixel_values=x).logits
                            elif is_temp: out = model(x.permute(0, 2, 1, 3, 4))
                            else: out = model(x)
                            nn.CrossEntropyLoss().to(DEVICE)(out, y).backward(); optimizer.step()
                        except: continue
                
                torch.save(model.state_dict(), os.path.join(save_dir, f"fold{fold}_best.pth"))
                del model; torch.cuda.empty_cache()

if __name__ == "__main__":
    import torch.multiprocessing as mp
    try: mp.set_start_method('spawn', force=True)
    except: pass
    run_experiment()