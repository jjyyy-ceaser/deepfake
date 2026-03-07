import os
import multiprocessing
import itertools
import pandas as pd
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import GroupKFold
# -----------------------------------------------------------------------------
# 📊 [핵심 수정] 5대 평가지표 모두 import
# -----------------------------------------------------------------------------
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

def get_optimizer(model, model_name, lr, weight_decay, layer_decay=1.0):
    if "videomae" in model_name and layer_decay < 1.0:
        params = [{'params': p, 'lr': lr * layer_decay if "videomae" in n else lr} for n, p in model.named_parameters()]
        return optim.AdamW(params, weight_decay=weight_decay)
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

def main():
    multiprocessing.freeze_support()
    os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
    
    from utils import get_model
    from data_loader import get_dataloader, prepare_dataset 

    # ✅ 경로가 맞는지 마지막으로 확인하세요! (split_data.py 결과 경로)
    BASE_DIR = r"C:\Users\leejy\Desktop\test_experiment\dataset\final_dataset_v2\train"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    GRID_CONFIGS = {
        "xception": {
            "fixed": {"bs": 64}, 
            "search": {"lr": [1e-3, 5e-4, 1e-4], "dropout": [0.2, 0.35, 0.5]}
        },
        "swin": {
            "fixed": {"bs": 64}, 
            "search": {"lr": [1e-4, 5e-5, 1e-5], "weight_decay": [0.01, 0.03, 0.05]}
        },
        "r3d": {
            "fixed": {"bs": 16}, 
            "search": {"lr": [1e-3, 5e-4, 1e-4], "sampling": ["uniform", "dense"]}
        },
        "videomae": {
            "fixed": {"bs": 4, "accum": 4}, 
            "search": {"lr": [5e-4, 1e-4], "layer_decay": [0.65, 0.7, 0.75]}
        },
        "hybrid": {
            "fixed": {"bs": 32}, 
            "search": {"lr": [1e-3, 5e-4, 1e-4], "dropout": [0.3, 0.4, 0.5]}
        }
    }

    print(f"🚀 데이터 로드 중: {BASE_DIR}")
    # -------------------------------------------------------------------------
    # ⚠️ 만약 여기서 FileNotFoundError가 뜨면 BASE_DIR 경로를 다시 확인해주세요.
    # -------------------------------------------------------------------------
    files, labels, groups = prepare_dataset(BASE_DIR)
    gkf = GroupKFold(n_splits=5)
    
    fold0_idx, fold0_val_idx = next(gkf.split(files, labels, groups=groups))
    
    pre_results = []

    for model_name, cfg in GRID_CONFIGS.items():
        keys, values = zip(*cfg["search"].items())
        combos = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        print(f"\n{'='*30} 🤖 {model_name.upper()} Preliminary Search {'='*30}")
        
        for i, params in enumerate(combos):
            print(f"\n▶ Combo [{i+1}/{len(combos)}]: {params}")
            
            tr_f, tr_l = [files[k] for k in fold0_idx], [labels[k] for k in fold0_idx]
            val_f, val_l = [files[k] for k in fold0_val_idx], [labels[k] for k in fold0_val_idx]
            
            samp = params.get("sampling", "uniform")
            loader_tr = get_dataloader(tr_f, tr_l, model_name, cfg["fixed"]["bs"], sampling=samp)
            loader_val = get_dataloader(val_f, val_l, model_name, cfg["fixed"]["bs"], sampling=samp, shuffle=False)
            
            model = get_model(model_name, DEVICE, dropout_rate=params.get("dropout", 0.0))
            opt = get_optimizer(model, model_name, params["lr"], params.get("weight_decay", 0.01), params.get("layer_decay", 1.0))
            criterion = nn.CrossEntropyLoss()
            scaler = torch.amp.GradScaler('cuda')
            accum_steps = cfg["fixed"].get("accum", 1)

            # 3 Epoch 예비 학습
            for epoch in range(3):
                model.train()
                pbar = tqdm(enumerate(loader_tr), total=len(loader_tr), 
                            desc=f"   Epoch {epoch+1}/3", unit="batch", leave=False)
                
                for step, (bx, by) in pbar:
                    bx, by = bx.to(DEVICE), by.to(DEVICE)
                    if "r3d" in model_name: 
                        bx = bx.permute(0, 2, 1, 3, 4)
                    
                    with torch.amp.autocast('cuda'):
                        out = model(pixel_values=bx).logits if "videomae" in model_name else model(bx)
                        loss = criterion(out, by) / accum_steps
                    
                    scaler.scale(loss).backward()
                    
                    if (step + 1) % accum_steps == 0:
                        scaler.step(opt); scaler.update(); opt.zero_grad()
                    
                    if step % 5 == 0:
                        pbar.set_postfix(loss=f"{loss.item()*accum_steps:.4f}")

                torch.cuda.empty_cache()

            # 🔍 검증 및 5대 지표 계산
            model.eval()
            preds_prob = [] # 확률값 (AUC용)
            preds_bin = []  # 0/1 라벨 (Acc, F1용)
            trues = []
            
            val_pbar = tqdm(loader_val, desc="   🔍 Evaluating", unit="batch", leave=False)
            
            with torch.no_grad():
                for bx, by in val_pbar:
                    bx = bx.to(DEVICE)
                    if "r3d" in model_name: 
                        bx = bx.permute(0, 2, 1, 3, 4)
                        
                    with torch.amp.autocast('cuda'):
                        out = model(pixel_values=bx).logits if "videomae" in model_name else model(bx)
                    
                    # Softmax로 확률 변환
                    probs = torch.softmax(out, 1)[:, 1].cpu()
                    preds_prob.extend(probs.tolist())
                    
                    # 0.5 기준으로 0(Real)과 1(Fake) 결정
                    preds_bin.extend((probs >= 0.5).int().tolist())
                    trues.extend(by.tolist())
                    
                    del bx, out
            
            # 📊 지표 계산
            try:
                auc = roc_auc_score(trues, preds_prob) if len(set(trues)) > 1 else 0.5
                acc = accuracy_score(trues, preds_bin)
                f1  = f1_score(trues, preds_bin, zero_division=0)
                prec = precision_score(trues, preds_bin, zero_division=0)
                rec = recall_score(trues, preds_bin, zero_division=0)
            except Exception as e:
                print(f"   ⚠️ 지표 계산 중 오류 발생: {e}")
                auc, acc, f1, prec, rec = 0.5, 0, 0, 0, 0

            # 콘솔 출력 (가독성 좋게)
            print(f"   📊 Result: AUC={auc:.4f} | Acc={acc:.4f} | F1={f1:.4f} | Pre={prec:.4f} | Rec={rec:.4f}")
            
            # 결과 저장
            pre_results.append({
                **params, 
                "model": model_name, 
                "auc": auc,
                "acc": acc,
                "f1": f1,
                "precision": prec,
                "recall": rec
            })
            
            # 실시간으로 CSV 업데이트 (혹시 멈춰도 기록 남게)
            pd.DataFrame(pre_results).to_csv("preliminary_results_detailed.csv", index=False)
            
            del model, opt, loader_tr, loader_val, criterion, scaler
            gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()

    print(f"\n{'='*30} 🏆 Preliminary Search 완료! ('preliminary_results_detailed.csv' 확인) {'='*30}")

if __name__ == "__main__":
    main()