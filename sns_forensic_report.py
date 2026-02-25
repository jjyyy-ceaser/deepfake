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
# 1. 환경 변수 및 설정
# =====================================================================
BASE_DIR = r"C:\Users\leejy\Desktop\test_experiment\dataset\test"
TRAIN_LIST_PATH = r"C:\Users\leejy\Desktop\test_experiment\dataset\train_list.txt"
MODEL_DIR = r"C:\Users\leejy\Desktop\test_experiment"

# 단일 가중치 모드이므로 LR은 배제하고, 모델 타입과 정규화 수치만 유지합니다.
MODEL_CONFIGS = {
    "xception": {"type": "spatial", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    "convnext": {"type": "spatial", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    "swin":     {"type": "spatial", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    "r3d":      {"type": "temporal", "mean": [0.43216, 0.394666, 0.37645], "std": [0.22803, 0.22145, 0.216989]},
    "r2plus1d": {"type": "temporal", "mean": [0.43216, 0.394666, 0.37645], "std": [0.22803, 0.22145, 0.216989]},
    "videomae": {"type": "videomae", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
}

IMG_SIZE = 224
SEQ_LEN = 16 

# =====================================================================
# 2. 시스템 자원 관리 및 오염 방지
# =====================================================================
def clean_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def validate_leakage(test_files, train_list_path):
    if not os.path.exists(train_list_path): return
    with open(train_list_path, 'r', encoding='utf-8') as f:
        train_set = set(os.path.splitext(line.strip())[0] for line in f if line.strip())
    
    if not train_set: return

    for p in test_files:
        fname = os.path.splitext(os.path.basename(p))[0]
        if fname in train_set:
            raise ValueError(f"🚨 [직접 오염] '{fname}' 파일은 학습에 사용된 데이터입니다.")
        try:
            if fname.isdigit(): 
                p_fake = f"fake_svd_{int(fname)+1:03d}"
                if p_fake in train_set: raise ValueError(f"🚨 [쌍방향 오염] '{fname}'의 짝꿍 '{p_fake}'가 학습됨!")
            elif "fake_svd_" in fname:
                p_real = f"{int(fname.split('_')[-1])-1:05d}"
                if p_real in train_set: raise ValueError(f"🚨 [쌍방향 오염] '{fname}'의 짝꿍 '{p_real}'이 학습됨!")
        except: continue
    print("✅ 무결성 확인: 오염되지 않은 순수 테스트 데이터입니다.")

# =====================================================================
# 3. 데이터 로더
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
# 4. 단일 가중치 기반 전수 평가 루프
# =====================================================================
def start_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final_report = []

    for m_name, config in MODEL_CONFIGS.items():
        # ✅ 직관적인 단일 가중치 파일명 매핑
        w_file = f"{m_name}_pretrained.pth"
        w_path = os.path.join(MODEL_DIR, w_file)
        
        print(f"\n🔥 [타겟 모델] {m_name.upper()}")
        
        if not os.path.exists(w_path):
            print(f"  ⚠️ 가중치 없음 건너뜀: '{w_file}' 파일을 찾을 수 없습니다.")
            continue

        # 가중치가 존재하면 1회만 모델을 메모리에 로드
        print(f"  ✅ '{w_file}' 가중치 로드 성공. 평가를 시작합니다.")
        if config['type'] == "spatial":
            model = timm.create_model(m_name if 'swin' not in m_name else 'swin_tiny_patch4_window7_224', pretrained=False, num_classes=2)
            if m_name == "convnext": model = timm.create_model('convnext_tiny', pretrained=False, num_classes=2)
        elif config['type'] == "temporal":
            model = models.video.r3d_18() if m_name == "r3d" else models.video.r2plus1d_18()
            model.fc = torch.nn.Linear(model.fc.in_features, 2)
        else:
            model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base", num_labels=2, ignore_mismatched_sizes=True)
        
        # 외부 가중치 로드 시 텐서 사이즈 불일치를 대비한 strict=False 처리
        model.load_state_dict(torch.load(w_path, map_location=device), strict=False)
        model.to(device).eval()

        for mc in ["case1", "case4"]:
            for pf_label, pf_folder in {"Raw": "raw", "YouTube": "youtube", "Instagram": "instagram", "Kakao_Normal": "kakao_normal", "Kakao_High": "kakao_high"}.items():
                t_path = os.path.join(BASE_DIR, mc, pf_folder)
                r_files = sorted(glob.glob(os.path.join(t_path, "real", "*.mp4")))
                f_files = sorted(glob.glob(os.path.join(t_path, "fake", "*.mp4")))
                
                if not r_files or not f_files: continue
                test_files = r_files + f_files
                labels = [0] * len(r_files) + [1] * len(f_files)

                if pf_label == "Raw" and mc == "case1": validate_leakage(test_files, TRAIN_LIST_PATH)

                loader = DataLoader(FinalTestDataset(test_files, config), batch_size=4, num_workers=4, pin_memory=True)
                
                probs = []
                with torch.no_grad():
                    for inputs, _ in tqdm(loader, desc=f"[{mc.upper()}] {pf_label}", leave=False):
                        inputs = inputs.to(device)
                        if config['type'] == "videomae":
                            outputs = model(pixel_values=inputs.permute(0, 2, 1, 3, 4)).logits
                        else: 
                            outputs = model(inputs)
                        probs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
                
                preds = [1 if p > 0.5 else 0 for p in probs]
                
                # 단일 평가 결과 적재
                final_report.append({
                    "Model": m_name, "Case": mc, "Platform": pf_label,
                    "Acc": accuracy_score(labels, preds),
                    "AUC": roc_auc_score(labels, probs),
                    "F1": f1_score(labels, preds, zero_division=0),
                    "Precision": precision_score(labels, preds, zero_division=0),
                    "Recall": recall_score(labels, preds, zero_division=0)
                })
        
        # 모델 평가가 완전히 끝나면 메모리에서 파기
        del model
        clean_memory()

    if final_report:
        df_report = pd.DataFrame(final_report)
        output_excel_path = "Final_Robustness_Analysis.xlsx"
        df_report.to_excel(output_excel_path, index=False)
        print(f"\n✅ 모든 평가가 완료되었습니다. '{output_excel_path}' 파일에서 5대 지표 결과를 확인하십시오.")
    else:
        print("\n🚨 평가된 결과가 없습니다. 가중치 파일명이나 테스트 비디오 경로를 확인해 주세요.")

if __name__ == "__main__":
    start_evaluation()