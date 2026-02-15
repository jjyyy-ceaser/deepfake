import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import cv2
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import timm
from transformers import VideoMAEForVideoClassification
from sklearn.metrics import accuracy_score, roc_auc_score

# ==========================================
# ⚙️ 1. 장비 점검 및 경로 설정 (빨간 줄 해결)
# ==========================================
# 윈도우 경로 오류를 방지하기 위해 슬래시(/)를 사용합니다.
BASE_DIR = "C:/Users/leejy/Desktop/test_experiment/dataset"
MODEL_DIR = "C:/Users/leejy/Desktop/test_experiment"

# GPU 강제 할당: GPU가 없으면 여기서 바로 에러가 발생하여 멈춥니다.
if not torch.cuda.is_available():
    raise RuntimeError("❌ GPU(CUDA)를 찾을 수 없습니다! 가상환경 설정을 다시 확인하세요.")

DEVICE = torch.device("cuda")
print(f"✅ 사용 중인 장치: {torch.cuda.get_device_name(0)}")

DOMAINS = ["3_test_svd", "4_test_runway", "5_test_pika", "6_test_ffpp"]
CASES = ["case1", "case2", "case3", "case4"]
SEQ_LEN = 16

# ==========================================
# 📂 2. 평가용 데이터셋 클래스
# ==========================================
class RobustnessEvalDataset(Dataset):
    def __init__(self, data_dir, model_category, transform=None):
        self.samples = []
        self.transform = transform
        self.model_category = model_category
        
        for cls_name, label in [("real", 0), ("fake", 1)]:
            path = os.path.join(data_dir, cls_name)
            if os.path.exists(path):
                # 각 케이스별 33개 영상 전수 조사
                files = sorted([f for f in os.listdir(path) if f.lower().endswith('.mp4')])[:33]
                for f in files:
                    self.samples.append((os.path.join(path, f), label))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        v_path, label = self.samples[idx]
        cap = cv2.VideoCapture(v_path)
        frames = []
        
        try:
            if self.model_category == "spatial":
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
                ret, frame = cap.read()
                if not ret: frame = np.zeros((224, 224, 3), dtype=np.uint8)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.transform: frame = self.transform(frame)
                cap.release()
                return frame, label
            else:
                for _ in range(SEQ_LEN):
                    ret, frame = cap.read()
                    if not ret: frame = np.zeros((224, 224, 3), dtype=np.uint8)
                    else: frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if self.transform: frame = self.transform(frame)
                    frames.append(frame)
                cap.release()
                frames = torch.stack(frames).permute(1, 0, 2, 3) # (C, T, H, W)
                return frames, label
        except Exception as e:
            cap.release()
            # 에러 발생 시 빈 텐서 반환
            return torch.zeros((3, 224, 224)) if self.model_category == "spatial" else torch.zeros((3, 16, 224, 224)), label

# ==========================================
# 🏗️ 3. 모델 로드 및 최적화
# ==========================================
def load_model_safely(m_file):
    parts = m_file.replace('.pth', '').split('_')
    m_cat = parts[1] 
    m_name = parts[2]
    
    if m_cat == "spatial":
        if m_name == "xception": model = timm.create_model('xception', num_classes=2)
        elif m_name == "convnext": model = timm.create_model('convnext_tiny', num_classes=2)
        elif m_name == "swin": model = timm.create_model('swin_tiny_patch4_window7_224', num_classes=2)
    elif m_cat == "temporal":
        if m_name == "r3d": model = models.video.r3d_18(num_classes=2)
        elif m_name == "r2plus1d": model = models.video.r2plus1d_18(num_classes=2)
        elif m_name == "videomae":
            model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base", num_labels=2, ignore_mismatched_sizes=True)
            
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, m_file), map_location=DEVICE))
    return model.to(DEVICE).eval(), m_cat

# ==========================================
# 🚀 4. 288개 실험 전수 조사 루프
# ==========================================
def run():
    model_files = sorted([f for f in os.listdir(MODEL_DIR) if f.startswith('model_') and f.endswith('.pth')])
    print(f"🔎 총 {len(model_files)}개의 모델을 발견했습니다.")
    
    final_results = []

    for m_file in model_files:
        print(f"\n📊 평가 중: {m_file}")
        model, m_cat = load_model_safely(m_file)
        
        # 4070 SUPER의 성능을 활용하기 위해 배치 사이즈를 16으로 올렸습니다.
        batch_size = 16 
        size = 112 if any(x in m_file for x in ["r3d", "r2plus1d"]) else 224
        transform = transforms.Compose([
            transforms.ToPILImage(), transforms.Resize((size, size)),
            transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        for domain in DOMAINS:
            for case in CASES:
                test_path = os.path.join(BASE_DIR, domain, case)
                # 윈도우 환경이므로 num_workers는 0~2 사이가 안전합니다.
                dataset = RobustnessEvalDataset(test_path, m_cat, transform)
                if len(dataset) == 0: continue
                
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
                y_true, y_probs = [], []

                with torch.no_grad():
                    for inputs, labels in loader:
                        inputs = inputs.to(DEVICE)
                        if "videomae" in m_file:
                            inputs = inputs.permute(0, 2, 1, 3, 4)
                            outputs = model(pixel_values=inputs).logits
                        else:
                            outputs = model(inputs)
                        
                        probs = torch.softmax(outputs, dim=1)[:, 1]
                        y_true.extend(labels.numpy())
                        y_probs.extend(probs.cpu().numpy())

                acc = accuracy_score(y_true, np.array(y_probs) > 0.5)
                try: auc = roc_auc_score(y_true, y_probs)
                except: auc = 0.5
                
                final_results.append({
                    "Model": m_file, "Domain": domain, "Case": case,
                    "Accuracy": acc, "AUC": auc
                })
                print(f"   [{domain}/{case}] ACC: {acc:.2f} | AUC: {auc:.2f}")

    # 최종 엑셀 및 CSV 저장
    df = pd.DataFrame(final_results)
    df.to_excel("Final_Robustness_Analysis_288.xlsx", index=False)
    print("\n✨ 288개 실험 전수 조사 완료! 엑셀 파일을 확인하세요.")

if __name__ == "__main__":
    run()