import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from data_loader import UnifiedDataset
from train_universal import build_model # 모델 빌드 함수 재사용

# ==========================================
# ⚙️ 설정
# ==========================================
BASE_DIR = "C:/Users/leejy/Desktop/test_experiment/dataset"
CHECKPOINT_DIR = "checkpoints"
TARGET_MODELS = ["xception", "convnextv2_tiny", "swinv2_tiny", "r3d_18", "r2plus1d_18", "videomae_v2"]
DOMAINS = ["3_test_svd", "4_test_runway", "5_test_pika", "6_test_ffpp"]
CASES = ["case1", "case2", "case3", "case4"]
DEVICE = torch.device("cuda")

def run_eval():
    final_report = []
    
    # 평가용 전처리 (Resize 224)
    tf = transforms.Compose([
        transforms.ToPILImage(), transforms.Resize((224, 224)),
        transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print(f"🚀 [최종 평가] 5대 지표(ACC, AUC, F1, Pre, Rec) 산출 시작...")

    for model_name in TARGET_MODELS:
        print(f"\n📊 Evaluating {model_name}...")
        model_type = 'temporal' if any(x in model_name for x in ['videomae', 'r3d', 'r2plus1d']) else 'spatial'
        
        # 1. 5개 Fold 모델 로드
        models = []
        for fold in range(5):
            pth = os.path.join(CHECKPOINT_DIR, model_name, f"best_fold{fold}.pth")
            if os.path.exists(pth):
                m = build_model(model_name, DEVICE)
                m.load_state_dict(torch.load(pth))
                m.eval()
                models.append(m)
        
        if not models:
            print(f"⚠️ No models found for {model_name}. Skipping...")
            continue

        # 2. 도메인/케이스별 순회
        for domain in DOMAINS:
            for case in CASES:
                test_path = os.path.join(BASE_DIR, domain, case)
                if not os.path.exists(test_path): continue
                
                # 데이터 로드
                samples = []
                for cls, lbl in [("real", 0), ("fake", 1)]:
                    d = os.path.join(test_path, cls)
                    if os.path.exists(d):
                        files = [os.path.join(d, f) for f in os.listdir(d) if f.endswith('.mp4')]
                        for f in files: samples.append((f, lbl))
                
                if not samples: continue

                ds = UnifiedDataset(samples, model_type, transform=tf)
                loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2)
                
                # 지표별 점수 저장소 (5 Fold 분량)
                metrics = {
                    "ACC": [], "AUC": [], "F1": [], "Prec": [], "Rec": []
                }
                
                # 3. 5개 모델 각각 추론
                for m in models:
                    y_true = []
                    y_probs = []
                    y_preds = []
                    
                    with torch.no_grad():
                        for inputs, labels in loader:
                            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                            if "videomae" in model_name: inputs = inputs.permute(0, 2, 1, 3, 4)
                            
                            outputs = m(pixel_values=inputs).logits if "videomae" in model_name else m(inputs)
                            probs = torch.softmax(outputs, dim=1)[:, 1] # Fake 확률
                            preds = outputs.argmax(1) # 0 or 1 예측값
                            
                            y_true.extend(labels.cpu().numpy())
                            y_probs.extend(probs.cpu().numpy())
                            y_preds.extend(preds.cpu().numpy())
                    
                    # 지표 계산
                    try:
                        metrics["ACC"].append(accuracy_score(y_true, y_preds) * 100)
                        metrics["AUC"].append(roc_auc_score(y_true, y_probs) * 100)
                        metrics["F1"].append(f1_score(y_true, y_preds, zero_division=0) * 100)
                        metrics["Prec"].append(precision_score(y_true, y_preds, zero_division=0) * 100)
                        metrics["Rec"].append(recall_score(y_true, y_preds, zero_division=0) * 100)
                    except:
                        # 데이터가 한 클래스만 있는 경우 등 예외 처리
                        pass
                
                # 4. 통계 집계 (Mean ± SD)
                row = {
                    "Model": model_name, "Domain": domain, "Case": case
                }
                
                # 엑셀 출력을 위한 문자열 포맷팅
                console_msg = f"   [{domain}/{case}] "
                for k in metrics:
                    mean = np.mean(metrics[k])
                    std = np.std(metrics[k])
                    row[f"{k}_Mean"] = mean
                    row[f"{k}_SD"] = std
                    row[f"{k}_Str"] = f"{mean:.2f}±{std:.2f}"
                    console_msg += f"{k}:{mean:.1f} "
                
                print(console_msg)
                final_report.append(row)

    # 5. 최종 저장
    df = pd.DataFrame(final_report)
    output_file = "Final_Comprehensive_Report.xlsx"
    df.to_excel(output_file, index=False)
    print(f"\n✨ 모든 평가 완료! '{output_file}' 파일을 확인하세요.")

if __name__ == "__main__":
    run_eval()