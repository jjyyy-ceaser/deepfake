import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import os, glob, sys, gc
import pandas as pd
from tqdm import tqdm
from utils import DeepfakeDataset, get_model
from torchvision import transforms
from torch.cuda.amp import GradScaler, autocast

# ==========================================
# ⚙️ Final Grid Search 설정 (VideoMAE 전용 모드)
# ==========================================
TARGET_DATASET = "dataset_B_mixed" 
LR_LIST = [1e-4, 5e-5, 1e-5]
NUM_WORKERS = 0 
DEVICE = torch.device("cuda")

# 1 Epoch은 너무 짧음. 3 Epoch으로 늘려 신뢰도 확보 (기존과 동일)
TEST_EPOCHS = 3 

# [수정 포인트 1] VideoMAE 모델만 탐색하도록 그룹 재설정
SPATIAL_MODELS = []
TEMPORAL_MODELS = ["videomae"]

def get_group_config(group_name):
    if group_name == 'TEMPORAL':
        tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
        ])
        return tf, 16
    else:
        tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return tf, 32

def run_grid_search_group(models, group_name):
    if not models:
        return []

    print(f"\n🚀 Starting Grid Search Group: {group_name} (Epochs: {TEST_EPOCHS})")
    tf, bs = get_group_config(group_name)
    
    base = os.path.join("dataset", "final_datasets", TARGET_DATASET)
    real = glob.glob(os.path.join(base, "real", "*"))
    fake = glob.glob(os.path.join(base, "fake", "*"))
    
    if len(real) == 0 or len(fake) == 0:
        print(f"❌ Error: 데이터를 찾을 수 없습니다. {base}")
        return []

    files, labels = real + fake, [0]*len(real) + [1]*len(fake)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    t_idx, v_idx = next(skf.split(files, labels)) 
    
    group_results = []
    
    for model_name in models:
        print(f"  🔹 Model: {model_name} (Batch: {bs})")
        is_temp = group_name == 'TEMPORAL'
        
        ds_tr = DeepfakeDataset([files[i] for i in t_idx], [labels[i] for i in t_idx], 'temporal' if is_temp else 'spatial', tf)
        ds_val = DeepfakeDataset([files[i] for i in v_idx], [labels[i] for i in v_idx], 'temporal' if is_temp else 'spatial', tf)
        
        for lr in LR_LIST:
            try:
                l_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True, num_workers=NUM_WORKERS)
                l_val = DataLoader(ds_val, batch_size=bs, shuffle=False, num_workers=NUM_WORKERS)
                
                model = get_model(model_name, DEVICE)
                optimizer = optim.AdamW(model.parameters(), lr=lr)
                criterion = nn.CrossEntropyLoss().to(DEVICE)
                scaler = GradScaler()
                
                best_epoch_auc = 0.0
                
                # 에포크 반복 루프 (기존과 동일)
                for ep in range(TEST_EPOCHS):
                    model.train()
                    for x, y in tqdm(l_tr, desc=f"    LR={lr} | Ep={ep+1}", leave=False, ncols=80):
                        x, y = x.to(DEVICE), y.to(DEVICE, dtype=torch.long)
                        optimizer.zero_grad()
                        with autocast():
                            out = model(x.permute(0, 2, 1, 3, 4)) if is_temp else model(x)
                            loss = criterion(out, y)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                    # 검증 (매 에포크마다 확인)
                    model.eval()
                    preds, trues = [], []
                    with torch.no_grad():
                        for vx, vy in l_val:
                            vx = vx.to(DEVICE)
                            vout = model(vx.permute(0, 2, 1, 3, 4)) if is_temp else model(vx)
                            preds.extend(torch.softmax(vout, 1)[:, 1].cpu().tolist())
                            trues.extend(vy.tolist())
                    
                    epoch_auc = roc_auc_score(trues, preds) if len(set(trues)) > 1 else 0.5
                    
                    # 3번 중 가장 잘 나온 점수를 기록
                    if epoch_auc > best_epoch_auc:
                        best_epoch_auc = epoch_auc
                
                print(f"    ✅ LR={lr} | Best AUC (over {TEST_EPOCHS} eps)={best_epoch_auc:.4f}")
                group_results.append({"Model": model_name, "LR": lr, "AUC": best_epoch_auc})
                
                del model, optimizer, scaler
                torch.cuda.empty_cache()
                gc.collect()

            except Exception as e:
                print(f"    ❌ Error at LR={lr}: {e}")
                torch.cuda.empty_cache()
                gc.collect()
                continue
                
    return group_results

if __name__ == "__main__":
    import torch.multiprocessing as mp
    try: mp.set_start_method('spawn', force=True)
    except: pass
    
    final_res = []
    
    # 공간적 모델 리스트는 비어있으므로 바로 통과됩니다.
    final_res.extend(run_grid_search_group(SPATIAL_MODELS, "SPATIAL"))
    
    torch.cuda.empty_cache()
    gc.collect()
    
    # 시간적 모델(VideoMAE) 탐색 수행
    final_res.extend(run_grid_search_group(TEMPORAL_MODELS, "TEMPORAL"))
    
    # [수정 포인트 2] 기존 결과를 덮어쓰지 않고 안전하게 '이어 쓰기(Append)'
    if final_res:
        csv_file = "grid_search_master_results.csv"
        new_df = pd.DataFrame(final_res)
        
        if os.path.exists(csv_file):
            # 1. 기존 CSV 파일 읽어오기
            existing_df = pd.read_csv(csv_file)
            # 2. 혹시 이전에 실행하다 중단된 videomae 기록이 있다면 삭제하여 중복 방지
            existing_df = existing_df[existing_df['Model'] != 'videomae']
            # 3. 기존 기록 밑에 새로운 videomae 기록 붙이기
            updated_df = pd.concat([existing_df, new_df], ignore_index=True)
            updated_df.to_csv(csv_file, index=False)
            print(f"\n💾 성공: 기존 '{csv_file}' 파일에 VideoMAE의 LR과 AUC 결과가 안전하게 이어 쓰기 되었습니다.")
        else:
            # 파일이 없을 경우 새로 생성
            new_df.to_csv(csv_file, index=False)
            print(f"\n💾 새로운 '{csv_file}' 파일이 생성되어 저장되었습니다.")
    else:
        print("\n⚠️ 기록할 결과가 없습니다.")