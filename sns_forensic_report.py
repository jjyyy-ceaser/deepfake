import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
import os, gc, timm
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from transformers import VideoMAEForVideoClassification
from pathlib import Path
from datetime import datetime
import warnings

# 🚨 [모듈 임포트]
from data_loader import UnifiedDataset

warnings.filterwarnings("ignore")

BASE_DATA_DIR = Path(r"C:\Users\leejy\Desktop\test_experiment\dataset\final_dataset")
MODEL_SAVE_DIR = Path(r"C:\Users\leejy\Desktop\test_experiment\models")
REPORT_SAVE_DIR = Path(r"C:\Users\leejy\Desktop\test_experiment\results")
REPORT_SAVE_DIR.mkdir(parents=True, exist_ok=True)
TARGET_DOMAINS = ["raw", "instagram", "kakao_high", "kakao_normal", "youtube"]
MODELS_CONFIG = {
    "Spatial": ["xception", "convnext", "swin"],
    "Temporal": ["r3d", "r2plus1d"],
    "VideoMAE": ["videomae-base"]
}
BATCH_SIZE = 128
NUM_WORKERS = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def clean_memory():
    gc.collect()
    torch.cuda.empty_cache()

# Transform 정의 (정규화만)
def get_transforms(mode):
    if mode == 'Temporal':
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize([0.43216, 0.394666, 0.37645], [0.22803, 0.22145, 0.216989])
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

def get_model_structure(category, model_name):
    if category == 'Spatial':
        if model_name == 'xception': return timm.create_model('xception', pretrained=False, num_classes=2)
        elif model_name == 'convnext': return timm.create_model('convnext_tiny', pretrained=False, num_classes=2)
        elif model_name == 'swin': return timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=2)
    elif category == 'Temporal':
        if model_name == 'r3d': m = models.video.r3d_18(weights=None); m.fc = nn.Linear(m.fc.in_features, 2); return m
        elif model_name == 'r2plus1d': m = models.video.r2plus1d_18(weights=None); m.fc = nn.Linear(m.fc.in_features, 2); return m
    elif category == 'VideoMAE':
        return VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base", num_labels=2, ignore_mismatched_sizes=True)
    return None

def run_cross_domain_test():
    clean_memory()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORT_SAVE_DIR / f"Final_Report_{timestamp}.txt"
    excel_path = REPORT_SAVE_DIR / f"Final_Result_{timestamp}.xlsx"
    report_file = open(report_path, "w", encoding="utf-8")
    def write_log(text): print(text); report_file.write(text + "\n")

    write_log(f"🚀 [Final Check] Deepfake Cross-Domain Evaluation Started: {timestamp}")
    final_results = []

    for train_domain in TARGET_DOMAINS:
        write_log(f"\n=======================================================")
        write_log(f"🏋️ Train Domain: [{train_domain.upper()}] Evaluation Start")
        write_log(f"=======================================================")

        for category, models_list in MODELS_CONFIG.items():
            for model_name in models_list:
                write_log(f"\n  [Model: {model_name}] (Category: {category})")
                ensemble_models = []
                cat_lower = category.lower() 
                if cat_lower == 'videomae': folder_name = f"videomae_{train_domain}_{model_name}"
                else: folder_name = f"{cat_lower}_{train_domain}_{model_name}"
                model_dir = MODEL_SAVE_DIR / folder_name
                
                loaded_cnt = 0
                for fold in range(1, 6):
                    pth = model_dir / f"best_fold{fold}.pth"
                    if pth.exists():
                        m = get_model_structure(category, model_name)
                        try: m.load_state_dict(torch.load(pth, map_location=DEVICE))
                        except: m.load_state_dict(torch.load(pth, map_location='cpu'))
                        m.to(DEVICE); m.eval()
                        ensemble_models.append(m)
                        loaded_cnt += 1
                
                if loaded_cnt == 0:
                    write_log(f"  ❌ Skipping: No checkpoints found in {model_dir}")
                    continue
                write_log(f"  ✅ Loaded {loaded_cnt} Folds")

                for test_domain in TARGET_DOMAINS:
                    test_dir = BASE_DATA_DIR / test_domain / "test"
                    samples = []
                    for sub, lab in [("real", 0), ("fake", 1)]:
                        d = test_dir / sub
                        if d.exists(): samples += [(p, lab) for p in d.glob("*.mp4")]
                    if not samples: continue

                    # 🚨 [Loader 사용]
                    tf = get_transforms(category)
                    model_type_key = 'videomae' if category == 'VideoMAE' else category.lower()
                    img_s = 112 if category == 'Temporal' else 224
                    
                    ds = UnifiedDataset(samples, model_type_key, tf, img_size=img_s)
                    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=2)

                    all_probs, all_targets = [], []
                    with torch.no_grad():
                        for inputs, targets in tqdm(loader, desc=f"    Testing on {test_domain}", leave=False):
                            inputs = inputs.to(DEVICE, non_blocking=True)
                            batch_probs = torch.zeros(inputs.size(0), 2).to(DEVICE)
                            for m in ensemble_models:
                                if category == 'VideoMAE': outputs = m(pixel_values=inputs).logits
                                else: outputs = m(inputs)
                                batch_probs += torch.softmax(outputs, dim=1)
                            batch_probs /= loaded_cnt
                            all_probs.extend(batch_probs[:, 1].cpu().tolist())
                            all_targets.extend(targets.tolist())

                    preds = [1 if p >= 0.5 else 0 for p in all_probs]
                    acc = accuracy_score(all_targets, preds)
                    try: auc = roc_auc_score(all_targets, all_probs)
                    except: auc = 0.5
                    pre = precision_score(all_targets, preds, zero_division=0)
                    rec = recall_score(all_targets, preds, zero_division=0)
                    f1 = f1_score(all_targets, preds, zero_division=0)

                    write_log(f"    ▶ Test on [{test_domain.upper()}] -> Acc: {acc:.4f} | AUC: {auc:.4f}")
                    final_results.append({
                        "Category": category, "Model": model_name,
                        "Train_Domain": train_domain, "Test_Domain": test_domain,
                        "ACC": acc, "AUC": auc, "Precision": pre, "Recall": rec, "F1-Score": f1
                    })
                del ensemble_models; clean_memory()

    report_file.close()
    if final_results:
        df = pd.DataFrame(final_results)
        df.to_excel(excel_path, index=False)
        print(f"\n🎉 [COMPLETE] Report Saved: {excel_path}")

if __name__ == "__main__":
    try:
        from multiprocessing import freeze_support
        freeze_support()
    except: pass
    run_cross_domain_test()