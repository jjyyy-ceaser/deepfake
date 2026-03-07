import os
import multiprocessing
import itertools
import pandas as pd
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm  # 실시간 진행 상황 확인을 위한 라이브러리

def get_optimizer(model, model_name, lr, weight_decay, layer_decay=1.0):
    if "videomae" in model_name and layer_decay < 1.0:
        # VideoMAE의 Layer-wise Learning Rate Decay 적용
        params = [{'params': p, 'lr': lr * layer_decay if "videomae" in n else lr} for n, p in model.named_parameters()]
        return optim.AdamW(params, weight_decay=weight_decay)
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

def main():
    multiprocessing.freeze_support()
    # 윈도우 환경에서 OpenCV와의 충돌 방지 및 경로 설정
    os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
    
    from utils import get_model
    from data_loader import get_dataloader, prepare_dataset 

    
    BASE_DIR = r"C:\Users\leejy\Desktop\test_experiment\dataset\final_dataset_v2\train"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # [Rev.10] 정밀 사양서 기반 Grid Search Space 설정
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
    files, labels, groups = prepare_dataset(BASE_DIR)
    gkf = GroupKFold(n_splits=5)
    
    # 예비 탐색을 위한 Fold 0 인덱스 추출
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

            # 3 Epoch 짧은 예비 학습 시작
            for epoch in range(3):
                model.train()
                pbar = tqdm(enumerate(loader_tr), total=len(loader_tr), 
                            desc=f"   Epoch {epoch+1}/3", unit="batch", leave=False)
                
                for step, (bx, by) in pbar:
                    bx, by = bx.to(DEVICE), by.to(DEVICE)
                    
                    # R3D 모델만 차원 순서를 (B, C, T, H, W)로 변환
                    if "r3d" in model_name: 
                        bx = bx.permute(0, 2, 1, 3, 4)
                    
                    with torch.amp.autocast('cuda'):
                        out = model(pixel_values=bx).logits if "videomae" in model_name else model(bx)
                        loss = criterion(out, by) / accum_steps
                    
                    scaler.scale(loss).backward()
                    
                    if (step + 1) % accum_steps == 0:
                        scaler.step(opt); scaler.update(); opt.zero_grad()
                    
                    # 진행 바에 현재 Loss 표시
                    if step % 5 == 0:
                        pbar.set_postfix(loss=f"{loss.item()*accum_steps:.4f}")

                torch.cuda.empty_cache()

            # 검증 단계
            model.eval(); preds, trues = [], []
            val_pbar = tqdm(loader_val, desc="   🔍 Evaluating", unit="batch", leave=False)
            
            with torch.no_grad():
                for bx, by in val_pbar:
                    bx = bx.to(DEVICE)
                    if "r3d" in model_name: 
                        bx = bx.permute(0, 2, 1, 3, 4)
                        
                    with torch.amp.autocast('cuda'):
                        out = model(pixel_values=bx).logits if "videomae" in model_name else model(bx)
                    
                    preds.extend(torch.softmax(out, 1)[:, 1].cpu().tolist())
                    trues.extend(by.tolist())
                    del bx, out
            
            auc = roc_auc_score(trues, preds) if len(set(trues)) > 1 else 0.5
            print(f"   📊 Combo {i+1} Fold 0 AUC: {auc:.4f}")
            
            pre_results.append({**params, "model": model_name, "fold0_auc": auc})
            pd.DataFrame(pre_results).to_csv("preliminary_results.csv", index=False)
            
            # 메모리 해제 및 캐시 정리
            del model, opt, loader_tr, loader_val, criterion, scaler
            gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()

    print(f"\n{'='*30} 🏆 Preliminary Search 완료! {'='*30}")

if __name__ == "__main__":
    main()