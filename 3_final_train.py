import os
import warnings
import gc
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# 🔧 [설정] 경고 제거 및 캐시 경로
warnings.filterwarnings("ignore")
os.environ['HF_HOME'] = r'C:\hf_cache'
os.environ['TORCH_HOME'] = r'C:\torch_cache'

# 로컬 모듈
from utils import get_model, calculate_metrics_at_best_threshold
from data_loader import get_dataloader, prepare_dataset

# ======================================================
# ⚙️ [최종 확정] 최적 하이퍼파라미터
# ======================================================
BEST_PARAMS = {
    "xception": {"lr": 1e-3, "dropout": 0.4, "bs": 32},
    "swin":     {"lr": 1e-4, "weight_decay": 0.1, "bs": 16},
    "r3d":      {"lr": 1e-4, "window_size": 12, "bs": 16},
    "videomae": {"lr": 1e-4, "layer_decay": 0.65, "bs": 4, "accum": 4},
    # "hybrid":   {"lr": 1e-3, "seq_len": 25, "dropout": 0.7, "bs": 8, "accum": 2}
}

TRAIN_ROOT = r"C:\Users\leejy\Desktop\test_experiment\dataset\final_dataset_v2\train"
SAVE_DIR = r"C:\Users\leejy\Desktop\test_experiment\results\final_weights"
DEVICE = torch.device("cuda")
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------------------------------------------------
# 🧹 메모리 관리
# ---------------------------------------------------------
def clean_memory():
    gc.collect()
    torch.cuda.empty_cache()

# ---------------------------------------------------------
# 🛠️ [7절] 4D/5D 호환 Mixup & CutMix
# ---------------------------------------------------------
def rand_bbox(size, lam):
    # size: [B, C, H, W] or [B, T, C, H, W] -> 마지막 두 차원은 항상 H, W
    W, H = size[-1], size[-2]
    cut_rat = np.sqrt(1. - lam)
    cw, ch = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1, y1 = np.clip(cx-cw//2, 0, W), np.clip(cy-ch//2, 0, H)
    x2, y2 = np.clip(cx+cw//2, 0, W), np.clip(cy+ch//2, 0, H)
    return x1, y1, x2, y2

def apply_aug(bx, by, alpha=0.2):
    """
    [참고] 이 함수는 입력 텐서의 차원 순서(C가 먼저인지 T가 먼저인지)와 무관하게 동작합니다.
    bx[..., y1:y2, x1:x2] 문법이 마지막 H, W 차원만 건드리기 때문입니다.
    따라서 R3D의 (B, C, T, H, W) 형태도 문제없이 증강됩니다.
    """
    if np.random.rand() > 0.5: # Mixup
        lam = np.random.beta(alpha, alpha)
        idx = torch.randperm(bx.size(0)).to(DEVICE)
        return lam * bx + (1 - lam) * bx[idx], by, by[idx], lam
    else: # CutMix
        lam = np.random.beta(alpha, alpha)
        idx = torch.randperm(bx.size(0)).to(DEVICE)
        x1, y1, x2, y2 = rand_bbox(bx.size(), lam)
        bx_aug = bx.clone()
        bx_aug[..., y1:y2, x1:x2] = bx[idx, ..., y1:y2, x1:x2]
        lam = 1 - ((x2-x1)*(y2-y1) / (bx.size(-1)*bx.size(-2)))
        return bx_aug, by, by[idx], lam

# ---------------------------------------------------------
# 🛠️ Optimizer (Frozen Layer 안전 장치 추가)
# ---------------------------------------------------------
def get_optimizer(model, model_name, lr, layer_decay=1.0):
    if "videomae" in model_name and layer_decay < 1.0:
        params = []
        for n, p in model.named_parameters():
            if not p.requires_grad: continue
            scale = layer_decay if "encoder.layer" in n else 1.0
            params.append({"params": p, "lr": lr * scale})
        return optim.AdamW(params, weight_decay=0.05)
    
    # 🔧 [Fix] 동결된 파라미터(Frozen)는 Optimizer에 넘기지 않도록 필터링
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    return optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)

def run_final_training():
    print(f"🚀 [Rev.18 Final] 5-Fold 학습 시작 (Optimization & R3D Corrected)")
    files, labels, groups = prepare_dataset(TRAIN_ROOT)
    gkf = GroupKFold(n_splits=5)
    
    final_results = []

    for model_name, params in BEST_PARAMS.items():
        print(f"\n{'='*60}\n🔥 학습 모델: {model_name.upper()}\n{'='*60}")
        
        for fold, (tr_idx, val_idx) in enumerate(gkf.split(files, labels, groups=groups)):
            # DataLoader
            frames = int(params.get("window_size", params.get("seq_len", 16)))
            loader_tr = get_dataloader([files[i] for i in tr_idx], [labels[i] for i in tr_idx], 
                                       model_name, params["bs"], 'train', frames)
            loader_val = get_dataloader([files[i] for i in val_idx], [labels[i] for i in val_idx], 
                                        model_name, params["bs"], 'test', frames)

            model = get_model(model_name, DEVICE, **params)
            optimizer = get_optimizer(model, model_name, params["lr"], params.get("layer_decay", 1.0))
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
            scaler = torch.cuda.amp.GradScaler() 
            accum = params.get("accum", 1)

            best_auc = 0.0
            best_stats = {}

            # Epoch Loop
            for epoch in range(10):
                # [Train]
                model.train()
                for step, (bx, by) in enumerate(tqdm(loader_tr, desc=f"F{fold+1} Ep{epoch+1}", leave=False)):
                    bx, by = bx.to(DEVICE), by.to(DEVICE)
                    
                    # 1. Augmentation (bx는 Dataset에서 이미 모델에 맞는 차원으로 옴)
                    # R3D의 경우 이미 [B, C, T, H, W] 상태임
                    bx_aug, y_a, y_b, lam = apply_aug(bx, by)
                    
                    # 🚨 [삭제됨] 이중 Transpose 방지
                    # if "r3d" in model_name: 
                    #     bx_aug = bx_aug.transpose(1, 2)

                    # 🔧 [Modern Autocast]
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        if "videomae" in model_name:
                            out = model(pixel_values=bx_aug).logits
                        else:
                            out = model(bx_aug)
                        
                        loss = (lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)) / accum
                    
                    scaler.scale(loss).backward()
                    if (step + 1) % accum == 0:
                        scaler.step(optimizer); scaler.update(); optimizer.zero_grad()

                # [Validation]
                model.eval()
                probs, trues = [], []
                
                with torch.no_grad():
                    for bx, by in loader_val:
                        bx = bx.to(DEVICE)
                        
                        # 🚨 [삭제됨] 이중 Transpose 방지
                        # if "r3d" in model_name: 
                        #     bx = bx.transpose(1, 2)

                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            if "videomae" in model_name:
                                out = model(pixel_values=bx).logits
                            else:
                                out = model(bx)
                                
                        p = torch.softmax(out, 1)[:, 1].cpu().numpy()
                        probs.extend(p); trues.extend(by.numpy())

                # [Metric]
                try:
                    auc = roc_auc_score(trues, probs)
                    apcer, bpcer, eer, best_thresh = calculate_metrics_at_best_threshold(trues, probs)
                except:
                    auc, eer, apcer, best_thresh = 0.5, 0.5, 0.5, 0.5

                # Save Best
                if auc > best_auc:
                    best_auc = auc
                    best_stats = {"auc": auc, "eer": eer, "apcer": apcer, "thresh": best_thresh, "epoch": epoch+1}
                    torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"{model_name}_f{fold+1}.pth"))

            print(f"   ✅ F{fold+1} 완료 | Best AUC: {best_stats.get('auc', 0):.4f}")
            
            final_results.append({
                "model": model_name, "fold": fold+1, 
                "best_auc": best_stats.get('auc', 0), "best_epoch": best_stats.get('epoch', 0),
                "eer": best_stats.get('eer', 0.5), "apcer": best_stats.get('apcer', 0.0), 
                "best_thresh": best_stats.get('thresh', 0.5)
            })
            
            clean_memory()

    pd.DataFrame(final_results).to_csv(os.path.join(SAVE_DIR, "final_training_summary.csv"), index=False)
    print(f"\n🏆 실험 종료! 결과 리포트: {SAVE_DIR}")

if __name__ == "__main__":
    run_final_training()