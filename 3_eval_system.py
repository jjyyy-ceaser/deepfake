import torch
import pandas as pd
import glob, os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from utils import DeepfakeDataset, get_model
from torch.utils.data import DataLoader
from torchvision import transforms

DOMAINS = ["svd", "pika", "runway", "ffpp"]
CASES = ["case1_original", "case2_lowres", "case3_compress", "case4_mixed"]
TRAIN_SETS = ["dataset_A_pure", "dataset_C_worst", "dataset_B_mixed"]
MODELS = ["xception", "convnext", "swin", "r3d", "r2plus1d", "videomae_v2"]
DEVICE = torch.device("cuda")

def run_evaluation():
    results = []
    # ToPILImage 제거 (utils.py와 충돌 방지)
    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    
    test_loaders = {}
    print("📂 Loading Test Data...")
    for dom in DOMAINS:
        for case in CASES:
            label_val = 0 if dom == 'ffpp' else 1 
            path = os.path.join("dataset", "processed_cases", "test", case, dom)
            files = glob.glob(os.path.join(path, "*"))
            test_loaders[f"{dom}_{case}"] = (files, [label_val]*len(files))

    for train_set in TRAIN_SETS:
        for model_name in MODELS:
            print(f"📊 Eval: {train_set} | {model_name}")
            model_type = 'temporal' if any(x in model_name for x in ['r3d', 'r2', 'mae']) else 'spatial'
            is_mae = 'mae' in model_name
            is_temp = any(x in model_name for x in ['r3d', 'r2']) # 3D CNN 여부

            models_fold = []
            for fold in range(5):
                pth = os.path.join("checkpoints", train_set, model_name, f"fold{fold}_best.pth")
                if not os.path.exists(pth): continue
                m = get_model(model_name, DEVICE)
                m.load_state_dict(torch.load(pth))
                m.eval(); models_fold.append(m)

            for dom in DOMAINS:
                for case in CASES:
                    files, labels = test_loaders[f"{dom}_{case}"]
                    if len(files) == 0: continue
                    ds = DeepfakeDataset(files, labels, model_type, tf)
                    loader = DataLoader(ds, batch_size=8, shuffle=False)
                    
                    all_preds, all_labels = [], []
                    with torch.no_grad():
                        for x, y in loader:
                            x = x.to(DEVICE)
                            fold_outputs = []
                            for m in models_fold:
                                # [핵심 수정] 학습 코드와 차원 로직 일치
                                if is_mae: out = m(pixel_values=x).logits
                                elif is_temp: out = m(x.permute(0, 2, 1, 3, 4))
                                else: out = m(x)
                                fold_outputs.append(torch.softmax(out, 1)[:, 1].cpu().numpy())
                            
                            avg_pred = np.mean(fold_outputs, axis=0)
                            all_preds.extend(avg_pred)
                            all_labels.extend(y.numpy())
                    
                    # 지표 계산 로직은 동일
                    preds_binary = [1 if p > 0.5 else 0 for p in all_preds]
                    results.append({
                        "Train_Set": train_set, "Model": model_name, "Test_Domain": dom, "Test_Case": case,
                        "AUC": roc_auc_score(all_labels, all_preds) if len(set(all_labels))>1 else 0,
                        "ACC": accuracy_score(all_labels, preds_binary),
                        "F1": f1_score(all_labels, preds_binary, zero_division=0)
                    })

    pd.DataFrame(results).to_excel("Final_Robustness_Report.xlsx", index=False)
    print("🎉 실험 종료! 리포트 생성 완료.")

if __name__ == "__main__":
    run_evaluation()