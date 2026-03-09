import os
import warnings
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

# [Rev.18] 터미널 클린업 및 캐시 설정
warnings.filterwarnings("ignore")
os.environ['HF_HOME'] = r'C:\hf_cache'
os.environ['TORCH_HOME'] = r'C:\torch_cache'

# 로컬 모듈 연결
from utils import get_model, calculate_metrics_at_best_threshold
from data_loader import get_dataloader, prepare_dataset

# ======================================================
# ⚙️ [설정] 테스트 환경 및 경로
# ======================================================
DOMAIN_ROOT = r"C:\Users\leejy\Desktop\test_experiment\dataset\sns"
DOMAINS = {
    "Raw":           os.path.join(DOMAIN_ROOT, "raw"),
    "YouTube":       os.path.join(DOMAIN_ROOT, "youtube"),
    "Instagram":     os.path.join(DOMAIN_ROOT, "instagram"),
    "Kakao_High":    os.path.join(DOMAIN_ROOT, "kakao_high"),
    "Kakao_Normal":  os.path.join(DOMAIN_ROOT, "kakao_normal")
}

WEIGHT_DIR = r"C:\Users\leejy\Desktop\test_experiment\results\final_weights"
SAVE_DIR = r"C:\Users\leejy\Desktop\test_experiment\results\cross_domain_report"
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_CONFIGS = {
    "xception": {"dropout": 0.4, "bs": 32, "frames": 16},
    "swin":     {"bs": 16, "frames": 16},
    "r3d":      {"window_size": 12, "bs": 16, "frames": 12},
    "videomae": {"bs": 8, "frames": 16},
    "hybrid":   {"seq_len": 25, "dropout": 0.7, "bs": 8, "frames": 25}
}

def test_model_on_domain(model, loader, model_name):
    """단일 모델, 단일 도메인에 대한 추론 수행"""
    model.eval()
    probs, trues = [], []
    
    with torch.no_grad():
        for bx, by in loader:
            bx = bx.to(DEVICE)
            
            # 🚨 [주의] dataset.py에서 이미 permute(1,0,2,3)를 했다면 
            # 여기서 다시 transpose(1,2)를 하면 (B, T, C, H, W)가 되어 에러가 날 수 있습니다.
            # R3D는 (B, C, T, H, W)를 입력으로 받습니다.
            if "r3d" in model_name:
                # bx = bx.transpose(1, 2) # 차원 확인 후 필요시 해제
                pass

            # 🔧 [수정 2] 최신 Autocast 문법 적용 (Deprecation Warning 방지)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                if "videomae" in model_name:
                    out = model(pixel_values=bx).logits
                else:
                    out = model(bx)
            
            p = torch.softmax(out, 1)[:, 1].cpu().numpy()
            probs.extend(p)
            trues.extend(by.numpy())
            
    return trues, probs

def run_cross_domain_evaluation():
    print(f"🚀 [Rev.18] Cross-Domain 강건성 평가 시작")
    final_report = []

    for model_name, cfg in MODEL_CONFIGS.items():
        print(f"\n{'='*60}\n🔍 평가 모델: {model_name.upper()}\n{'='*60}")
        
        for domain_name, domain_path in DOMAINS.items():
            if not os.path.exists(domain_path):
                continue
                
            print(f"   🌍 Domain: {domain_name}")
            files, labels, _ = prepare_dataset(domain_path)
            if len(files) == 0: continue
                
            loader = get_dataloader(files, labels, model_name, cfg["bs"], 'test', cfg["frames"])
            fold_metrics = {'auc': [], 'eer': [], 'apcer': []}
            
            for fold in range(1, 6):
                weight_path = os.path.join(WEIGHT_DIR, f"{model_name}_f{fold}.pth")
                if not os.path.exists(weight_path): continue
                
                model = get_model(model_name, DEVICE, **cfg)
                model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
                
                trues, probs = test_model_on_domain(model, loader, model_name)
                
                # 🔧 [수정 1] 지표 산출 예외 처리 범위 확장 (Single-class 에러 방지)
                try:
                    # 데이터가 1개 클래스만 있을 경우 roc_auc_score는 ValueError를 던집니다.
                    auc = roc_auc_score(trues, probs)
                    apcer, _, eer, _ = calculate_metrics_at_best_threshold(trues, probs)
                except Exception as e:
                    print(f"      ⚠️ 지표 계산 실패 (Fold {fold}): {e}")
                    auc, apcer, eer = 0.5, 0.5, 0.5 
                
                fold_metrics['auc'].append(auc)
                fold_metrics['eer'].append(eer)
                fold_metrics['apcer'].append(apcer)
                
                del model; torch.cuda.empty_cache()

            if fold_metrics['auc']:
                avg_auc, avg_eer, avg_apcer = np.mean(fold_metrics['auc']), np.mean(fold_metrics['eer']), np.mean(fold_metrics['apcer'])
                print(f"      👉 결과: AUC {avg_auc:.4f} | EER {avg_eer:.4f} | APCER {avg_apcer:.4f}")
                
                final_report.append({
                    "Model": model_name, "Domain": domain_name,
                    "AUC": avg_auc, "EER": avg_eer, "APCER": avg_apcer
                })

    df = pd.DataFrame(final_report)
    save_path = os.path.join(SAVE_DIR, "final_cross_domain_report.csv")
    df.to_csv(save_path, index=False)
    print(f"\n🏆 테스트 완료! 리포트: {save_path}")

if __name__ == "__main__":
    run_cross_domain_evaluation()