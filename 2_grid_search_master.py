import os

# 🔧 [Windows 경로 오류 해결] 캐시 경로를 짧은 곳으로 강제 변경
# 이 코드는 반드시 다른 torch/transformers import보다 위에 있어야 합니다.
os.environ['HF_HOME'] = r'C:\hf_cache'
os.environ['TORCH_HOME'] = r'C:\torch_cache'
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import itertools
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve
from tqdm import tqdm

from utils import get_model
from data_loader import get_dataloader, prepare_dataset

# ---------------------------------------------------------
# 🛠️ [7절/10.3절] 고급 데이터 증강 (Mixup & CutMix)
# ---------------------------------------------------------
def apply_aug(bx, by, alpha=0.2):
    """Mixup/CutMix 50:50 확률 적용"""
    if np.random.rand() > 0.5: # Mixup
        lam = np.random.beta(alpha, alpha)
        idx = torch.randperm(bx.size(0)).to(bx.device)
        return lam * bx + (1 - lam) * bx[idx], by, by[idx], lam
    else: # CutMix
        lam = np.random.beta(alpha, alpha)
        idx = torch.randperm(bx.size(0)).to(bx.device)
        W, H = bx.size(2), bx.size(3)
        cut_rat = np.sqrt(1. - lam)
        cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
        cx, cy = np.random.randint(W), np.random.randint(H)
        x1, y1 = np.clip(cx - cut_w // 2, 0, W), np.clip(cy - cut_h // 2, 0, H)
        x2, y2 = np.clip(cx + cut_w // 2, 0, W), np.clip(cy + cut_h // 2, 0, H)
        bx[:, :, x1:x2, y1:y2] = bx[idx, :, x1:x2, y1:y2]
        lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
        return bx, by, by[idx], lam

# ---------------------------------------------------------
# 🛠️ [9-2절] Layer-wise LR Decay Optimizer
# ---------------------------------------------------------
def get_optimizer(model, model_name, lr, layer_decay=1.0):
    if "videomae" in model_name and layer_decay < 1.0:
        params = []
        for n, p in model.named_parameters():
            # Encoder 레이어에 decay 적용 (약식)
            ld = layer_decay if "encoder.layer" in n else 1.0
            params.append({"params": p, "lr": lr * ld})
        return optim.AdamW(params, weight_decay=0.01)
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

def calculate_iso_metrics(trues, bins, probs):
    tn, fp, fn, tp = confusion_matrix(trues, bins).ravel()
    apcer = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    bpcer = fp / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr, tpr, _ = roc_curve(trues, probs)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]
    return apcer, bpcer, eer

# ---------------------------------------------------------
# 📊 [9절] 그리드 탐색 메인
# ---------------------------------------------------------
GRID_CONFIGS = {
    "xception": {"search": {"lr": [1e-3, 1e-4, 1e-5], "dropout": [0.2, 0.4, 0.6]}, "bs": 32},
    # "swin":     {"search": {"lr": [1e-4, 5e-5, 1e-5], "weight_decay": [0.01, 0.05, 0.1]}, "bs": 16},
    # "r3d":      {"search": {"lr": [1e-3, 5e-4, 1e-4], "window_size": [12, 16]}, "bs": 16},
    # "videomae": {"search": {"lr": [5e-4, 1e-4], "layer_decay": [0.65, 0.75, 0.85]}, "bs": 4, "accum": 4},
    # "hybrid":   {"search": {"lr": [1e-3, 5e-4], "seq_len": [16, 25], "dropout": [0.4, 0.7]}, "bs": 16}
}

TRAIN_ROOT = r"C:\Users\leejy\Desktop\test_experiment\dataset\final_dataset_v2\train"
SAVE_DIR = r"C:\Users\leejy\Desktop\test_experiment\results"
DEVICE = torch.device("cuda")

def run_grid_search():
    files, labels, groups = prepare_dataset(TRAIN_ROOT)
    gkf = GroupKFold(n_splits=5)
    fold0_idx, fold0_val_idx = next(gkf.split(files, labels, groups=groups))
    
    results = []

    for model_name, cfg in GRID_CONFIGS.items():
        keys, values = zip(*cfg["search"].items())
        combos = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        for params in combos:
            print(f"\n🔍 [Grid Search] {model_name.upper()} | Params: {params}")
            
            frames = int(params.get("window_size", params.get("seq_len", 16)))
            loader_tr = get_dataloader([files[i] for i in fold0_idx], [labels[i] for i in fold0_idx], 
                                       model_name, cfg["bs"], 'train', frames)
            loader_val = get_dataloader([files[i] for i in fold0_val_idx], [labels[i] for i in fold0_val_idx], 
                                        model_name, cfg["bs"], 'test', frames)

            model = get_model(model_name, DEVICE, **params)
            # 🔧 [9-2절] Decay 반영 Optimizer
            optimizer = get_optimizer(model, model_name, params["lr"], params.get("layer_decay", 1.0))
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
            scaler = torch.amp.GradScaler('cuda')
            accum = cfg.get("accum", 1)

            # 3 Epochs 단기 탐색
            for epoch in range(3):
                model.train()
                for step, (bx, by) in enumerate(tqdm(loader_tr, leave=False)):
                    bx, by = bx.to(DEVICE), by.to(DEVICE)
                    # 🔧 [7절] CutMix/Mixup 증강 적용
                    bx_aug, y_a, y_b, lam = apply_aug(bx, by)
                    
                    with torch.amp.autocast('cuda'):
                        out = model(pixel_values=bx_aug).logits if "videomae" in model_name else model(bx_aug)
                        loss = (lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)) / accum
                    scaler.scale(loss).backward()
                    if (step + 1) % accum == 0:
                        scaler.step(optimizer); scaler.update(); optimizer.zero_grad()

            model.eval(); probs, bins, trues = [], [], []
            with torch.no_grad():
                for bx, by in loader_val:
                    bx = bx.to(DEVICE)
                    with torch.amp.autocast('cuda'):
                        out = model(pixel_values=bx).logits if "videomae" in model_name else model(bx)
                    p = torch.softmax(out, 1)[:, 1].cpu().numpy()
                    probs.extend(p); bins.extend((p>=0.5).astype(int)); trues.extend(by.numpy())

            apcer, bpcer, eer = calculate_iso_metrics(trues, bins, probs)
            results.append({**params, "model": model_name, "auc": roc_auc_score(trues, probs), "eer": eer, "apcer": apcer})
            pd.DataFrame(results).to_csv(os.path.join(SAVE_DIR, "grid_search_results.csv"), index=False)
            del model; torch.cuda.empty_cache()

if __name__ == "__main__":
    run_grid_search()