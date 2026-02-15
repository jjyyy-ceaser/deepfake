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

BASE_DIR = "C:/Users/leejy/Desktop/test_experiment/dataset"
MODEL_DIR = "C:/Users/leejy/Desktop/test_experiment"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DOMAINS = ["3_test_svd", "4_test_runway", "5_test_pika", "6_test_ffpp"]
CASES = ["case1", "case2", "case3", "case4"]
SEQ_LEN = 16

class RobustnessEvalDataset(Dataset):
    def __init__(self, data_dir, model_category, transform=None):
        self.samples = []
        self.transform = transform
        self.model_category = model_category
        for cls_name, label in [("real", 0), ("fake", 1)]:
            path = os.path.join(data_dir, cls_name)
            if os.path.exists(path):
                files = sorted([f for f in os.listdir(path) if f.lower().endswith('.mp4')])
                for f in files: self.samples.append((os.path.join(path, f), label))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        v_path, label = self.samples[idx]
        cap = cv2.VideoCapture(v_path)
        try:
            if self.model_category == "spatial":
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
                ret, frame = cap.read()
                if not ret: frame = np.zeros((224, 224, 3), dtype=np.uint8)
                else: frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.transform: frame = self.transform(frame)
                cap.release()
                return frame, label
            else:
                frames = []
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                indices = np.linspace(0, total_frames-1, SEQ_LEN, dtype=int) if total_frames >= SEQ_LEN else np.arange(SEQ_LEN) % (total_frames if total_frames > 0 else 1)
                
                for i in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ret, frame = cap.read()
                    if not ret: frame = np.zeros((224, 224, 3), dtype=np.uint8)
                    else: frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if self.transform: frame = self.transform(frame)
                    frames.append(frame)
                cap.release()
                frames = torch.stack(frames).permute(1, 0, 2, 3)
                return frames, label
        except:
            cap.release()
            return torch.zeros((3, 224, 224)) if self.model_category=="spatial" else torch.zeros((3, 16, 224, 224)), label

def load_model_safely(m_file):
    parts = m_file.replace('.pth', '').split('_')
    m_cat, m_name = parts[1], parts[2]
    
    if m_cat == "spatial":
        if m_name == "xception": model = timm.create_model('xception', num_classes=2)
        elif m_name == "convnext": model = timm.create_model('convnext_tiny', num_classes=2)
        elif m_name == "swin": model = timm.create_model('swin_tiny_patch4_window7_224', num_classes=2)
    elif m_cat == "temporal":
        if m_name == "r3d": model = models.video.r3d_18(num_classes=2) # Default input is handled by AdaptivePool, so 224 works fine
        elif m_name == "r2plus1d": model = models.video.r2plus1d_18(num_classes=2)
        elif m_name == "videomae":
            model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base", num_labels=2, ignore_mismatched_sizes=True)
            
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, m_file), map_location=DEVICE))
    return model.to(DEVICE).eval(), m_cat

def run():
    model_files = sorted([f for f in os.listdir(MODEL_DIR) if f.startswith('model_') and f.endswith('.pth')])
    final_results = []

    # 🚨 핵심 수정: 모든 모델에 대해 해상도 224로 통일
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    for m_file in model_files:
        print(f"\n📊 평가 중: {m_file}")
        try:
            model, m_cat = load_model_safely(m_file)
        except Exception as e:
            print(f"⚠️ 모델 로드 실패 ({m_file}): {e}")
            continue
            
        for domain in DOMAINS:
            for case in CASES:
                test_path = os.path.join(BASE_DIR, domain, case)
                if not os.path.exists(test_path): continue
                
                dataset = RobustnessEvalDataset(test_path, m_cat, transform)
                if len(dataset) == 0: continue
                
                loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)
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
                
                final_results.append({"Model": m_file, "Domain": domain, "Case": case, "Accuracy": acc, "AUC": auc})
                print(f"   [{domain}/{case}] Acc: {acc:.2f}")

    pd.DataFrame(final_results).to_excel("Final_Robustness_Analysis_Fixed.xlsx", index=False)
    print("\n✨ 평가 완료!")

if __name__ == "__main__":
    run()