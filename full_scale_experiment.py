import os
import subprocess
import pandas as pd
import time
import shutil
import argparse
import torch
import gc

# 🎯 6개 모델 전수 조사
TARGET_MODELS = [
    "xception", "convnextv2_tiny", "swinv2_tiny",  # Spatial
    "r3d_18", "r2plus1d_18", "videomae_v2"         # Temporal
]

# 탐색 범위
LR_LIST = [1e-4, 5e-5, 1e-5]
BATCH_LIST = [4, 8]
OPTIMIZER = "adamw"

# 단계별 설정
GS_EPOCHS = 5      # 탐색용 (5 Epoch)
FINAL_EPOCHS = 10  # 최종 학습용 (10 Epoch)

def run_manager(specific_model=None):
    os.makedirs("grid_results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    # 모델 선택 로직 (전체 vs 특정 모델)
    if specific_model:
        if specific_model not in TARGET_MODELS:
            print(f"❌ 목록에 없는 모델입니다: {specific_model}")
            return
        run_list = [specific_model]
    else:
        run_list = TARGET_MODELS

    print(f"📋 실행 계획: {run_list} (총 {len(run_list)}개 모델)")

    for model in run_list:
        print(f"\n{'='*50}")
        print(f"🔥 Processing Model: {model.upper()}")
        print(f"{'='*50}")

        # [STEP 1] Grid Search
        print(f"🔎 Step 1: Grid Search (Finding Best Params)...")
        best_auc = -1.0
        best_cfg = {"lr": 1e-4, "batch_size": 4} # 기본값
        
        current_batches = [2, 4] if "videomae" in model else BATCH_LIST

        for lr in LR_LIST:
            for bs in current_batches:
                print(f"   👉 Testing [LR={lr}, BS={bs}] ", end="")
                fold_auc_sum = 0.0
                valid_run = True
                
                # 5-Fold 검증
                for fold in range(5):
                    cmd = [
                        "python", "train_universal.py",
                        "--model", model,
                        "--lr", str(lr), "--batch_size", str(bs),
                        "--optimizer", OPTIMIZER,
                        "--epochs", str(GS_EPOCHS),
                        "--fold", str(fold),
                        "--save_model", "False"
                    ]
                    try:
                        res = subprocess.run(cmd, capture_output=True, text=True)
                        val_auc = 0.5
                        for line in res.stdout.split('\n'):
                            if "FINAL_VAL_AUC:" in line:
                                val_auc = float(line.split(":")[1].strip())
                        fold_auc_sum += val_auc
                        print(".", end="", flush=True)
                    except:
                        valid_run = False
                        print("X", end="", flush=True)

                avg_auc = fold_auc_sum / 5.0
                print(f" -> Avg AUC: {avg_auc:.4f}")

                if valid_run and avg_auc > best_auc:
                    best_auc = avg_auc
                    best_cfg = {"lr": lr, "batch_size": bs}

        print(f"🏆 Best Params Found: {best_cfg} (AUC: {best_auc:.4f})")
        
        # [STEP 2] Final Training (5-Fold, 10 Epochs)
        print(f"🚀 Step 2: Final Training (10 Epochs, Saving Models)...")
        model_save_dir = os.path.join("checkpoints", model)
        os.makedirs(model_save_dir, exist_ok=True)

        for fold in range(5):
            print(f"   📌 Training Fold {fold}/5 ... ", end="")
            cmd = [
                "python", "train_universal.py",
                "--model", model,
                "--lr", str(best_cfg['lr']), 
                "--batch_size", str(best_cfg['batch_size']),
                "--optimizer", OPTIMIZER,
                "--epochs", str(FINAL_EPOCHS),
                "--fold", str(fold),
                "--save_model", "True"
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL)
            
            src = f"temp_best_{model}.pth"
            dst = os.path.join(model_save_dir, f"best_fold{fold}.pth")
            if os.path.exists(src):
                if os.path.exists(dst): os.remove(dst)
                shutil.move(src, dst)
                print("Done.")
            else:
                print("Failed.")

        # [안전장치] 모델 하나 끝날 때마다 1분 쿨링
        print(f"\n❄️ GPU Cooling (60s)...")
        time.sleep(60)

    print(f"\n✨ All Selected Models Completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="실행할 특정 모델명 (예: xception)")
    args = parser.parse_args()
    
    run_manager(args.model)