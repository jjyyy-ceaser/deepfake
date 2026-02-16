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
from tqdm import tqdm  # 진행바 라이브러리
from utils import DeepfakeDataset, get_model
from torchvision import transforms

# =========================================================
# ⚙️ 실험 설정 (Configuration)
# =========================================================
# 순서: Pure(기준) -> Worst(극한) -> Mixed(범용)
DATASETS = ["dataset_A_pure", "dataset_C_worst", "dataset_B_mixed"] 

# 사용할 모델 리스트
MODELS = ["xception", "convnext", "swin", "r3d", "r2plus1d", "videomae_v2"]

# 하이퍼파라미터 탐색 범위
LR_LIST = [1e-4, 5e-5]
BATCH_LIST = [4, 8] # VideoMAE는 메모리 이슈로 내부에서 [2, 4]로 자동 조정됨

# 장치 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# 🛠️ 핵심 함수 정의
# =========================================================

def get_data(dataset_name):
    """데이터 파일 경로와 라벨을 로드합니다."""
    base = os.path.join("dataset", "final_datasets", dataset_name)
    real = glob.glob(os.path.join(base, "real", "*"))
    fake = glob.glob(os.path.join(base, "fake", "*"))
    
    # 데이터 확인
    if len(real) == 0 or len(fake) == 0:
        print(f"⚠️ 경고: {dataset_name} 폴더가 비어있거나 경로가 잘못되었습니다.")
        return [], []

    files = real + fake
    labels = [0]*len(real) + [1]*len(fake)
    return files, labels

def train_epoch(model, loader, criterion, optimizer, is_mae, desc=""):
    """한 Epoch 동안 학습을 수행하며 진행바를 표시합니다."""
    model.train()
    running_loss = 0.0
    
    # tqdm으로 래핑하여 진행바 생성
    loop = tqdm(loader, desc=desc, leave=False, file=sys.stdout)
    
    for x, y in loop:
        x, y = x.to(DEVICE), y.to(DEVICE)
        
        optimizer.zero_grad()
        
        # 모델 타입에 따른 순전파(Forward)
        if is_mae: 
            # VideoMAE는 입력 구조가 다름 (Batch, Time, Channel, Height, Width)
            out = model(pixel_values=x.permute(0,2,1,3,4)).logits
        else: 
            out = model(x)
            
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # 진행바 옆에 실시간 Loss 표시
        loop.set_postfix(loss=f"{loss.item():.4f}")
    
    return running_loss / len(loader)

def evaluate(model, loader, is_mae):
    """모델 성능(AUC)을 평가합니다."""
    model.eval()
    preds, trues = [], []
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            if is_mae: 
                out = model(pixel_values=x.permute(0,2,1,3,4)).logits
            else: 
                out = model(x)
            
            # Softmax 확률 계산 (Fake일 확률)
            probs = torch.softmax(out, 1)[:, 1]
            preds.extend(probs.cpu().tolist())
            trues.extend(y.cpu().tolist())
            
    # AUC 계산 (에러 방지 처리)
    try:
        if len(set(trues)) < 2: return 0.5 # 라벨이 하나만 있는 경우
        return roc_auc_score(trues, preds)
    except:
        return 0.5

# =========================================================
# 🚀 메인 실행 루프 (Main Pipeline)
# =========================================================
def run_experiment():
    print(f"\n🚀 [System Started] Device: {DEVICE}")
    print(f"📦 Datasets: {DATASETS}")
    print(f"🤖 Models: {MODELS}")
    print("="*60)

    # 이미지 변환기 (Resize & Tensor)
    tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    for ds_name in DATASETS:
        print(f"\n🌍 [[ Processing Dataset: {ds_name} ]]")
        files, labels = get_data(ds_name)
        if not files: continue

        # ⚖️ 클래스 불균형 해결 (Weighted Loss)
        # Real: 300개, Fake: 135개 (Test로 30개 이동함)
        # Weight = 300 / 135 ≈ 2.22
        class_weights = torch.tensor([1.0, 300.0/135.0]).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # 5-Fold 세팅
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for model_name in MODELS:
            print(f"\n  🔹 Model: {model_name}")
            save_dir = os.path.join("checkpoints", ds_name, model_name)
            os.makedirs(save_dir, exist_ok=True)
            
            # 모델 타입 판별
            is_mae = 'videomae' in model_name
            model_type = 'temporal' if any(x in model_name for x in ['r3d', 'r2', 'mae']) else 'spatial'
            
            # 배치 사이즈 조정 (VideoMAE는 무거워서 줄임)
            current_batches = [2, 4] if is_mae else BATCH_LIST

            # -----------------------------------------------------
            # [Step 1] Grid Search (최적 파라미터 찾기) - Fold 0만 사용
            # -----------------------------------------------------
            print(f"    🔎 Grid Search (Hyperparameter Tuning)...", flush=True)
            
            best_auc = -1
            best_params = {'lr': 1e-4, 'bs': 4} # 기본값
            
            # 첫 번째 Fold만 추출
            train_idx, val_idx = next(skf.split(files, labels))
            
            ds_train = DeepfakeDataset([files[i] for i in train_idx], [labels[i] for i in train_idx], model_type, tf)
            ds_val = DeepfakeDataset([files[i] for i in val_idx], [labels[i] for i in val_idx], model_type, tf)
            
            for lr in LR_LIST:
                for bs in current_batches:
                    # 짧게 3 Epoch만 학습해봄
                    loader_tr = DataLoader(ds_train, batch_size=bs, shuffle=True)
                    loader_val = DataLoader(ds_val, batch_size=bs)
                    
                    model = get_model(model_name, DEVICE)
                    opt = optim.AdamW(model.parameters(), lr=lr)
                    
                    for ep in range(3):
                        train_epoch(model, loader_tr, criterion, opt, is_mae, desc=f"GS LR={lr} BS={bs}")
                    
                    val_auc = evaluate(model, loader_val, is_mae)
                    print(f"      👉 LR={lr}, BS={bs} -> Val AUC: {val_auc:.4f}")
                    
                    if val_auc > best_auc:
                        best_auc = val_auc
                        best_params = {'lr': lr, 'bs': bs}
            
            print(f"    ✅ Best Params Selected: {best_params} (AUC: {best_auc:.4f})")
            
            # 파라미터 저장
            with open(os.path.join(save_dir, "best_params.json"), "w") as f:
                json.dump(best_params, f)

            # -----------------------------------------------------
            # [Step 2] Main 5-Fold Training (본 학습)
            # -----------------------------------------------------
            print(f"    🚀 Starting 5-Fold Cross Validation...", flush=True)
            final_lr, final_bs = best_params['lr'], best_params['bs']
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(files, labels)):
                ds_train = DeepfakeDataset([files[i] for i in train_idx], [labels[i] for i in train_idx], model_type, tf)
                ds_val = DeepfakeDataset([files[i] for i in val_idx], [labels[i] for i in val_idx], model_type, tf)
                
                loader_tr = DataLoader(ds_train, batch_size=final_bs, shuffle=True)
                loader_val = DataLoader(ds_val, batch_size=final_bs)
                
                model = get_model(model_name, DEVICE)
                opt = optim.AdamW(model.parameters(), lr=final_lr)
                
                best_fold_auc = 0.0
                
                # 10 Epochs 본 학습
                for ep in range(10):
                    desc_text = f"Fold {fold+1}/5 | Ep {ep+1}/10"
                    train_loss = train_epoch(model, loader_tr, criterion, opt, is_mae, desc=desc_text)
                    val_auc = evaluate(model, loader_val, is_mae)
                    
                    # 최고 성능 모델 저장
                    if val_auc > best_fold_auc:
                        best_fold_auc = val_auc
                        save_path = os.path.join(save_dir, f"fold{fold}_best.pth")
                        torch.save(model.state_dict(), save_path)
                
                print(f"      🏆 Fold {fold} Done. Best AUC: {best_fold_auc:.4f}")
            
            print(f"    ✨ {model_name} 학습 완료.")

if __name__ == "__main__":
    run_experiment()