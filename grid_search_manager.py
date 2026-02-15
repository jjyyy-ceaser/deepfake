import os
import subprocess
import pandas as pd
import itertools
import time
from datetime import datetime

# ==========================================
# ⚙️ 1. 탐색 공간 (Grid Search Space)
# ==========================================
# RTX 4070 SUPER (12GB) 맞춤형 설정
PARAM_GRID = {
    "learning_rate": [1e-4, 5e-5, 1e-5],    # 학습률 3종
    "batch_size": [4, 8],                   # 배치 사이즈 2종 (16은 OOM 위험)
    "optimizer": ["adamw", "adam"],         # 옵티마이저 2종
}

# 튜닝할 모델 목록
TARGET_MODELS = [
    "videomae_v2", 
    "r3d_18", 
    "swinv2_tiny", 
    "convnextv2_tiny"
]

# 사용할 데이터셋 (강건성 확보를 위해 Mixed 추천)
DATASET_TYPE = "mixed"  # pure / mixed / worst

# 교차 검증 설정
K_FOLDS = 5
EPOCHS_PER_RUN = 5  # 탐색용이므로 짧게 설정

def run_grid_search():
    results = []
    total_start_time = time.time()
    
    # 결과 저장 폴더
    os.makedirs("grid_results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"grid_results/grid_search_{DATASET_TYPE}_{timestamp}.csv"

    print(f"🚀 Grid Search 시작! (Target: {DATASET_TYPE})")
    print(f"   - Models: {TARGET_MODELS}")
    print(f"   - Grid: {PARAM_GRID}")
    
    # 1. 모델별 루프
    for model_name in TARGET_MODELS:
        # VideoMAE는 메모리를 많이 먹으므로 배치 사이즈 8 제외 (Safety Lock)
        current_grid = PARAM_GRID.copy()
        if "videomae" in model_name:
            current_grid["batch_size"] = [2, 4]
        
        keys, values = zip(*current_grid.items())
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        print(f"\n👉 [{model_name}] 총 {len(combinations)}개 조합 테스트 예정")

        # 2. 파라미터 조합별 루프
        for i, params in enumerate(combinations):
            lr = params['learning_rate']
            bs = params['batch_size']
            opt = params['optimizer']
            
            print(f"\n   Testing Combo {i+1}/{len(combinations)}: LR={lr}, BS={bs}, OPT={opt}")
            
            fold_scores = []
            
            # 3. 5-Fold Cross Validation 루프
            for fold_idx in range(K_FOLDS):
                print(f"      Running Fold {fold_idx+1}/{K_FOLDS}...", end=" ", flush=True)
                
                # subprocess로 train_universal.py 실행 (메모리 완전 초기화 효과)
                cmd = [
                    "python", "train_universal.py",
                    "--model", model_name,
                    "--dataset", DATASET_TYPE,
                    "--lr", str(lr),
                    "--batch_size", str(bs),
                    "--optimizer", opt,
                    "--epochs", str(EPOCHS_PER_RUN),
                    "--fold", str(fold_idx),
                    "--k_folds", str(K_FOLDS),
                    "--save_model", "False"  # 탐색 중엔 모델 저장 안 함 (용량 절약)
                ]
                
                try:
                    # 실행 및 출력 캡처
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    # 출력에서 Validation AUC 파싱 (train_universal.py가 출력해야 함)
                    output_lines = result.stdout.split('\n')
                    val_auc = 0.5
                    val_acc = 0.0
                    
                    for line in output_lines:
                        if "FINAL_VAL_AUC:" in line:
                            val_auc = float(line.split(":")[1].strip())
                        if "FINAL_VAL_ACC:" in line:
                            val_acc = float(line.split(":")[1].strip())
                            
                    if result.returncode != 0:
                        print(f"❌ Error in Fold {fold_idx}: {result.stderr}")
                        val_auc = 0.0 # 실패 처리
                    else:
                        print(f"✅ Done (AUC: {val_auc:.4f})")
                        
                    fold_scores.append(val_auc)
                    
                except Exception as e:
                    print(f"❌ Exception: {e}")
                    fold_scores.append(0.0)

            # 5-Fold 평균 계산
            avg_auc = sum(fold_scores) / K_FOLDS
            print(f"   👉 Average AUC: {avg_auc:.4f}")
            
            # 결과 기록
            record = {
                "Model": model_name,
                "Dataset": DATASET_TYPE,
                "LR": lr,
                "BatchSize": bs,
                "Optimizer": opt,
                "Avg_AUC": avg_auc,
                "Fold_Scores": str(fold_scores)
            }
            results.append(record)
            
            # 중간 저장
            pd.DataFrame(results).to_csv(csv_filename, index=False)

    total_time = (time.time() - total_start_time) / 60
    print(f"\n✨ 모든 탐색 완료! (소요시간: {total_time:.1f}분)")
    print(f"📄 결과 파일: {csv_filename}")
    
    # 최적 결과 출력
    df = pd.DataFrame(results)
    best_row = df.loc[df.groupby("Model")["Avg_AUC"].idxmax()]
    print("\n🏆 모델별 최적 설정 (Best Hyperparameters):")
    print(best_row[["Model", "LR", "BatchSize", "Optimizer", "Avg_AUC"]])

if __name__ == "__main__":
    run_grid_search()