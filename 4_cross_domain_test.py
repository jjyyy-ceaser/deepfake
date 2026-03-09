import os
import warnings
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

# 🔧 [설정] 터미널 클린업 및 캐시 경로 설정
warnings.filterwarnings("ignore")
os.environ['HF_HOME'] = r'C:\hf_cache'
os.environ['TORCH_HOME'] = r'C:\torch_cache'

# 로컬 모듈 임포트
from utils import get_model, calculate_metrics_at_best_threshold
from data_loader import get_dataloader, prepare_dataset

# ======================================================
# ⚙️ [설정] 테스트 환경 및 경로 (Rev.18 설계안 준수)
# ======================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_ROOT = r"C:\Users\leejy\Desktop\test_experiment\dataset\final_dataset_v2"
WEIGHT_DIR = r"C:\Users\leejy\Desktop\test_experiment\results\final_weights"
SAVE_DIR = r"C:\Users\leejy\Desktop\test_experiment\results"

# 📋 평가할 모델 라인업 (설계서 확정본)
MODELS = {
    "xception": "xception",
    "swin_tiny_patch4_window7_224": "swin", # 파일명이 swin_f1.pth 일 경우
    "r3d_18": "r3d",                        # 파일명이 r3d_f1.pth 일 경우
    "videomae_base": "videomae",            # 파일명이 videomae_f1.pth 일 경우
    "hybrid": "hybrid"                # 파일명이 hybrid_f1.pth 일 경우
}

# 🌍 도메인별 경로 매핑 (7번 스크립트 생성 구조와 1:1 매칭)
DOMAINS = {
    "Raw":           os.path.join(DATASET_ROOT, "raw", "test"),
    "Instagram":     os.path.join(DATASET_ROOT, "instagram"), 
    "YouTube":       os.path.join(DATASET_ROOT, "youtube"),   
    "Kakao_High":    os.path.join(DATASET_ROOT, "kakao_high"),
    "Kakao_Normal":  os.path.join(DATASET_ROOT, "kakao_normal") 
}

def main():
    print(f"🚀 [5-Fold Cross-Domain Test] 시작: 통계적 유의성 및 포렌식 지표 검증\n")
    final_results = []

    for model_name, weight_prefix in MODELS.items():
        print(f"🔍 분석 모델: {model_name}")
        
        # 🚨 [설계안 9절] 모델별 최적 윈도우 사이즈 설정
        if "r3d" in model_name.lower():
            frames = 12
        elif "videomae" in model_name.lower():
            frames = 16
        elif "gru" in model_name.lower() or "hybrid" in model_name.lower():
            frames = 25
        else:
            frames = 1  # Spatial Models

        for domain_name, domain_path in DOMAINS.items():
            if not os.path.exists(domain_path):
                print(f"   ⚠️  [Skip] 경로 없음: {domain_name}")
                continue

            test_files, test_labels, test_groups = prepare_dataset(domain_path)
            if not test_files: 
                print(f"   ⚠️  [Skip] 데이터 없음: {domain_name}")
                continue

            # 폴드별 지표 저장소 (ISO 30107-3 표준에 따라 BPCER 추가)
            fold_metrics = {"auc": [], "eer": [], "apcer": [], "bpcer": [], "acc": [], "pre": [], "rec": []}

            # 🛡️ [핵심] 5-Fold 가중치 순회 평가 (f1~f5 규칙 준수)
            for fold in range(1, 6):
                weight_file = f"{weight_prefix}_f{fold}.pth"
                weight_path = os.path.join(WEIGHT_DIR, weight_file)

                if not os.path.exists(weight_path):
                    continue

                test_loader = get_dataloader(
                    test_files, test_labels, 
                    model_name=model_name, 
                    batch_size=16,
                    mode='test', 
                    frames=frames
                )

                model = get_model(model_name, device=DEVICE, num_classes=2)
                try:
                    # 🔧 [Prefix 일관성] 훈련 시와 동일하게 접두어 제거 후 로드
                    state_dict = torch.load(weight_path, map_location=DEVICE)
                    new_state_dict = {}
                    for k, v in state_dict.items():
                        name = k.replace('module.', '').replace('backbone.', '')
                        new_state_dict[name] = v
                    
                    model.load_state_dict(new_state_dict, strict=False)
                    model.eval()
                except Exception as e:
                    print(f"      ❌ Fold {fold} 로드 실패: {e}")
                    continue

                trues, probs = [], []
                with torch.no_grad():
                    for bx, by in tqdm(test_loader, desc=f"      Fold {fold} | {domain_name}", leave=False):
                        bx = bx.to(DEVICE)
                        outputs = model(bx)
                        if hasattr(outputs, 'logits'): outputs = outputs.logits
                        
                        prob = torch.softmax(outputs, dim=1)[:, 1]
                        trues.extend(by.numpy())
                        probs.extend(prob.cpu().numpy())

                # 📊 지표 산출 (설계안 5절 준수: APCER/BPCER 분리)
                try:
                    auc = roc_auc_score(trues, probs)
                    # EER을 최소화하는 최적 임계값(best_thresh)에서 APCER/BPCER 산출
                    apcer, bpcer, eer, best_thresh = calculate_metrics_at_best_threshold(trues, probs)
                    
                    preds = (np.array(probs) >= best_thresh).astype(int)
                    acc = accuracy_score(trues, preds)
                    pre = precision_score(trues, preds, zero_division=0)
                    rec = recall_score(trues, preds, zero_division=0)

                    fold_metrics["auc"].append(auc)
                    fold_metrics["eer"].append(eer)
                    fold_metrics["apcer"].append(apcer)
                    fold_metrics["bpcer"].append(bpcer)
                    fold_metrics["acc"].append(acc)
                    fold_metrics["pre"].append(pre)
                    fold_metrics["rec"].append(rec)
                except Exception as e:
                    print(f"      ⚠️ 지표 계산 실패: {e}")

                del model; torch.cuda.empty_cache()

            # 도메인별 5-Fold 평균 및 표준편차 산출
            if fold_metrics["auc"]:
                res = {k: (np.mean(v), np.std(v)) for k, v in fold_metrics.items()}
                
                print(f"   🌍 {domain_name:<12} | Avg AUC: {res['auc'][0]:.4f} | Avg EER: {res['eer'][0]:.4f} | Avg ACC: {res['acc'][0]:.4f}")
                
                final_results.append({
                    "Model": model_name, "Domain": domain_name,
                    "AUC_Mean": res['auc'][0], "AUC_Std": res['auc'][1],
                    "EER_Mean": res['eer'][0], "EER_Std": res['eer'][1],
                    "APCER_Mean": res['apcer'][0], "BPCER_Mean": res['bpcer'][0],
                    "ACC_Mean": res['acc'][0], "Precision_Mean": res['pre'][0], "Recall_Mean": res['rec'][0]
                })

    # 최종 보고서 저장
    if final_results:
        df = pd.DataFrame(final_results)
        csv_path = os.path.join(SAVE_DIR, "5fold_full_metrics_report.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n✅ 5-Fold 테스트 완료! 전체 지표 저장됨: {csv_path}")
        
        # 요약 피벗 테이블 출력
        summary = df.pivot(index="Model", columns="Domain", values="AUC_Mean")
        print("\n[📊 도메인별 평균 AUC 요약]")
        print(summary.to_string())
    else:
        print("\n❌ 테스트 결과가 없습니다. 가중치 파일명과 경로를 확인하세요.")

if __name__ == "__main__":
    main()