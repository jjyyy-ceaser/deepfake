import os
import torch
import cv2
import numpy as np
from torch.utils.data import DataLoader
from data_loader import DeepfakeDataset, MODEL_SPECS
from torchvision.utils import save_image

# ==========================================
# ⚙️ 설정 (실제 학습 환경과 동일하게)
# ==========================================
# [중요] 아까 스플릿한 v2 폴더 경로로 설정하세요!
BASE_DIR = r"C:\Users\leejy\Desktop\test_experiment\dataset\final_dataset_v2\train"
SAVE_DIR = "debug_samples_v2"
BATCH_SIZE = 2 # 디버그니까 작게

# 학습 코드와 동일한 모델별 설정
GRID_CONFIGS = {
    "xception": {"fixed": {"bs": 16}, "type": "spatial"},
    "swin":     {"fixed": {"bs": 16}, "type": "spatial"},
    "r3d":      {"fixed": {"bs": 16}, "type": "temporal"},
    "videomae": {"fixed": {"bs": 4},  "type": "temporal"},
    "hybrid":   {"fixed": {"bs": 16}, "type": "temporal"}
}

def denormalize(tensor, mean, std):
    """ 정규화된 텐서를 다시 이미지로 복구 (시각화용) """
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def main():
    print(f"🚀 [Final Debug] 데이터 로더 및 입력 차원 정밀 점검 시작")
    print(f"📂 데이터 경로: {BASE_DIR}")
    
    if not os.path.exists(BASE_DIR):
        print(f"❌ 오류: 데이터 경로가 존재하지 않습니다. split_data.py를 먼저 실행했는지 확인하세요.")
        return

    os.makedirs(SAVE_DIR, exist_ok=True)

    # 더미 데이터 리스트 생성 (경로 확인용)
    import glob
    real_files = glob.glob(os.path.join(BASE_DIR, "real", "*.mp4"))
    fake_files = glob.glob(os.path.join(BASE_DIR, "fake", "*.mp4"))
    
    if not real_files or not fake_files:
        print(f"❌ 오류: real({len(real_files)}) 또는 fake({len(fake_files)}) 파일이 부족합니다.")
        return

    # 테스트용 파일 리스트 (각각 2개씩만)
    sample_files = real_files[:2] + fake_files[:2]
    sample_labels = [0, 0, 1, 1]

    # ====================================================
    # 🔍 모든 모델 타입 순회하며 차원 점검
    # ====================================================
    for model_name, cfg in GRID_CONFIGS.items():
        print(f"\n--------------------------------------------------")
        print(f"🔍 [Check] 모델: {model_name.upper()}")
        
        # 1. 데이터셋 & 로더 생성
        ds = DeepfakeDataset(sample_files, sample_labels, model_name=model_name, sampling='uniform')
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
        
        try:
            # 2. 배치 하나 뽑기
            inputs, labels = next(iter(loader))
            
            # 3. 2_grid_search_master.py와 똑같은 전처리 로직 적용 (핵심!)
            final_input = inputs
            if "r3d" in model_name:
                # R3D는 (B, T, C, H, W) -> (B, C, T, H, W)로 변환
                final_input = inputs.permute(0, 2, 1, 3, 4)
            
            # 4. 결과 출력
            shape = final_input.shape
            print(f"   ✅ Input Shape: {shape}")
            print(f"   ✅ Label Shape: {labels.shape}")
            
            # 5. 모델별 정상 규격 판별
            is_pass = False
            
            if model_name in ["xception", "swin"]:
                # (Batch, 3, 224, 224)
                if len(shape) == 4 and shape[2] == 224: is_pass = True
                
            elif model_name == "r3d":
                # (Batch, 3, 16, 112, 112) -> Permute 적용됨
                if len(shape) == 5 and shape[1] == 3 and shape[3] == 112: is_pass = True
                
            elif model_name in ["videomae", "hybrid"]:
                # (Batch, 16, 3, 224, 224) -> Permute 없음
                if len(shape) == 5 and shape[1] == 16 and shape[3] == 224: is_pass = True
            
            if is_pass:
                print(f"   🎉 [PASS] 규격 일치 확인완료")
            else:
                print(f"   ❌ [FAIL] 규격 불일치! (사양서 확인 필요)")

            # 6. 이미지 저장 (시각적 확인)
            # 첫 번째 배치의 첫 번째 샘플 저장
            spec = ds.spec
            if len(shape) == 5: # 비디오 (T, C, H, W) or (C, T, H, W)
                # 시각화를 위해 (T, C, H, W)로 통일
                viz_tensor = inputs[0] 
                viz_tensor = denormalize(viz_tensor, spec['mean'], spec['std'])
                save_path = os.path.join(SAVE_DIR, f"{model_name}_sample.jpg")
                save_image(viz_tensor, save_path, nrow=4) # 4x4 그리드로 저장
            else: # 이미지 (C, H, W)
                viz_tensor = inputs[0]
                viz_tensor = denormalize(viz_tensor, spec['mean'], spec['std'])
                save_path = os.path.join(SAVE_DIR, f"{model_name}_sample.jpg")
                save_image(viz_tensor, save_path)
            
            print(f"   💾 샘플 저장: {save_path}")

        except Exception as e:
            print(f"   ❌ [ERROR] 로딩 실패: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n--------------------------------------------------")
    print(f"✅ 모든 점검 완료. '{SAVE_DIR}' 폴더에서 이미지를 확인하세요.")

if __name__ == "__main__":
    main()