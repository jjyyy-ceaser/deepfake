import os
import glob
import gc
import torch
import cv2
import numpy as np
import pandas as pd
import timm
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from transformers import VideoMAEForVideoClassification
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm

# =====================================================================
# 1. [그리드 서치 결과 및 모델별 최적 세팅]
# =====================================================================
BASE_DIR = r"C:\Users\leejy\Desktop\test_experiment\dataset\test"
TRAIN_LIST_PATH = r"C:\Users\leejy\Desktop\test_experiment\dataset\train_list.txt"
MODEL_DIR = r"C:\Users\leejy\Desktop\test_experiment"

# [그리드 서치(grid_search_master_results.csv) 기반 최적 LR 및 모델별 설정]
# 학습 시 사용된 최적의 정규화(Mean/Std) 및 하이퍼파라미터 이식
MODEL_CONFIGS = {
    "xception": {
        "type": "spatial", "best_lr": 5e-05, 
        "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]
    },
    "convnext": {
        "type": "spatial", "best_lr": 1e-04, 
        "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]
    },
    "swin": {
        "type": "spatial", "best_lr": 5e-05, 
        "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]
    },
    "r3d": {
        "type": "temporal", "best_lr": 1e-04, 
        "mean": [0.43216, 0.394666, 0.37645], "std": [0.22803, 0.22145, 0.216989]
    },
    "r2plus1d": {
        "type": "temporal", "best_lr": 1e-04, 
        "mean": [0.43216, 0.394666, 0.37645], "std": [0.22803, 0.22145, 0.216989]
    },
    "videomae": {
        "type": "videomae", "best_lr": 5e-05, 
        "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]
    }
}

IMG_SIZE = 224
SEQ_LEN = 16 
NUM_FOLDS = 5

# =====================================================================
# 2. [자원 최적화 및 무결성 검증 시스템]
# =====================================================================
def clean_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def validate_leakage(test_files, train_list_path):
    """
    [데이터 오염 방지 로직]
    사용자 정의 오프셋 반영: Real(N) <-> Fake(N+1)
    """
    if not os.path.exists(train_list_path): return
    with open(train_list_path, 'r', encoding='utf-8') as f:
        train_set = set(os.path.splitext(line.strip())[0] for line in f if line.strip())
    
    if not train_set: return

    for p in test_files:
        fname = os.path.splitext(os.path.basename(p))[0]
        if fname in train_set:
            raise ValueError(f"🚨 [오염] '{fname}'은(는) 이미 학습된 데이터입니다.")
        
        # 오프셋 규칙 체크
        try:
            if fname.isdigit():
                p_fake = f"fake_svd_{int(fname)+1:03d}"
                if p_fake in train_set: raise ValueError(f"🚨 [오염] '{fname}'의 짝꿍 '{p_fake}'가 학습됨!")
            elif "fake_svd_" in fname:
                p_real = f"{int(fname.split('_')[-1])-1:05d}"
                if p_real in train_set: raise ValueError(f"🚨 [오염] '{fname}'의 짝꿍 '{p_real}'이(가) 학습됨!")
        except: continue
    print("✅ 무결성 확인: 학습 리스트와 겹치지 않는 순수 테스트 데이터셋입니다.")

# =====================================================================
# 3. [데이터 로더] 최적 로딩 속도 적용
# =====================================================================
class FinalTestDataset(Dataset):
    def __init__(self, file_paths, config):
        self.file_paths = file_paths
        self.config = config
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config['mean'], std=config['std'])
        ])

    def __len__(self): return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        cap = cv2.VideoCapture(path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if self.config['type'] == "spatial":
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
            ret, frame = cap.read()
            cap.release()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ret else np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            return self.transform(frame), path
        else:
            indices = np.linspace(0, total_frames - 1, SEQ_LEN, dtype=int)
            frames = []
            for i in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ret else np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
                frames.append(self.transform(frame))
            cap.release()
            return torch.stack(frames).permute(1, 0, 2, 3), path 

# =====================================================================
# 4. [메인 평가 엔진] 최적 LR 가중치 기반 추론
# =====================================================================
def start_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final_report = []

    for m_name, config in MODEL_CONFIGS.items():
        print(f"\n🔥 [타겟 모델] {m_name.upper()} (Best LR: {config['best_lr']})")
        
        for mc in ["case1", "case4"]:
            for pf_label, pf_folder in {"Raw": "raw", "YouTube": "youtube", "Instagram": "instagram", "Kakao_Normal": "kakao_normal", "Kakao_High": "kakao_high"}.items():
                t_path = os.path.join(BASE_DIR, mc, pf_folder)
                r_files = sorted(glob.glob(os.path.join(t_path, "real", "*.mp4")))
                f_files = sorted(glob.glob(os.path.join(t_path, "fake", "*.mp4")))
                
                if not r_files or not f_files: continue
                test_files = r_files + f_files
                labels = [0]*len(r_files) + [1]*len(f_files)

                if pf_label == "Raw" and mc == "case1": validate_leakage(test_files, TRAIN_LIST_PATH)

                fold_res = []
                for f in range(1, NUM_FOLDS + 1):
                    # 가중치 파일 탐색 (Best LR 버전 로드)
                    w_file = f"model_{config['type']}_{m_name}_pure_fold{f}.pth"
                    if m_name == "videomae": w_file = f"model_temporal_videomae_pure_fold{f}.pth"
                    
                    w_path = os.path.join(MODEL_DIR, w_file)
                    if not os.path.exists(w_path): continue
                    
                    # 모델 로드
                    if config['type'] == "spatial":
                        model = timm.create_model(m_name if 'swin' not in m_name else 'swin_tiny_patch4_window7_224', pretrained=False, num_classes=2)
                        if m_name == "convnext": model = timm.create_model('convnext_tiny', pretrained=False, num_classes=2)
                    elif config['type'] == "temporal":
                        model = models.video.r3d_18() if m_name == "r3d" else models.video.r2plus1d_18()
                        model.fc = torch.nn.Linear(model.fc.in_features, 2)
                    else: # videomae
                        model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base", num_labels=2, ignore_mismatched_sizes=True)
                    
                    model.load_state_dict(torch.load(w_path, map_location=device))
                    model.to(device).eval()
                    
                    # DataLoader 설정 (num_workers=4, pin_memory=True 반영)
                    loader = DataLoader(FinalTestDataset(test_files, config), batch_size=4, num_workers=4, pin_memory=True)
                    
                    probs = []
                    with torch.no_grad():
                        for inputs, _ in tqdm(loader, desc=f"Fold {f}", leave=False):
                            inputs = inputs.to(device)
                            if config['type'] == "videomae":
                                outputs = model(pixel_values=inputs.permute(0, 2, 1, 3, 4)).logits
                            else: outputs = model(inputs)
                            probs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
                    
                    preds = [1 if p > 0.5 else 0 for p in probs]
                    fold_res.append({"acc": accuracy_score(labels, preds), "auc": roc_auc_score(labels, probs)})
                    del model; clean_memory() # 🧼 메모리 즉시 정리

                if fold_res:
                    final_report.append({
                        "Model": m_name, "Case": mc, "Platform": pf_label,
                        "Acc": np.mean([x['acc'] for x in fold_res]),
                        "AUC": np.mean([x['auc'] for x in fold_res])
                    })

    pd.DataFrame(final_report).to_csv("Final_Robustness_Analysis.csv", index=False)
    print("\n✅ 평가 완료. 'Final_Robustness_Analysis.csv'를 확인하십시오.")

if __name__ == "__main__":
    start_evaluation()