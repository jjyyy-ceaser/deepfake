import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

# 사용자 코드 import
from data_loader import prepare_dataset, get_transforms
from dataset import DeepfakeDataset  # 수정된 dataset 클래스

# ==========================================
# ⚙️ 설정 (Rev.18 실험 환경 동일)
# ==========================================
BASE_DIR = r"C:\Users\leejy\Desktop\test_experiment\dataset\final_dataset_v2\train"
SAVE_DIR = "debug_samples_v2"
BATCH_SIZE = 4 

# 모델별 설정
GRID_CONFIGS = {
    "xception": {"type": "spatial",  "input_size": 224},
    "swin":     {"type": "spatial",  "input_size": 224},
    "r3d":      {"type": "temporal", "input_size": 112},
    "videomae": {"type": "temporal", "input_size": 224},
    "hybrid":   {"type": "temporal", "input_size": 224}
}

def check_black_screen(tensor):
    """텐서의 표준편차(std)를 계산하여 단색 화면인지 감지"""
    std = torch.std(tensor)
    if std < 0.01: # 거의 단색
        return True, std.item()
    return False, std.item()

def denormalize(tensor, model_name):
    """모델별 정규화 역연산"""
    if "r3d" in model_name:
        mean = torch.tensor([0.432, 0.394, 0.376]).view(1, 3, 1, 1)
        std = torch.tensor([0.228, 0.221, 0.216]).view(1, 3, 1, 1)
    else:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return tensor * std + mean

def save_debug_batch(inputs, model_name, save_path):
    """배치 이미지 저장"""
    # 1. 차원 정리 (시각화를 위해 Image 형태로 변환)
    # VideoMAE/Hybrid: (B, T, C, H, W) -> T번째 프레임 추출
    # R3D: (B, C, T, H, W) -> T번째 프레임 추출
    
    if inputs.dim() == 5: 
        if inputs.shape[1] == 3: # (B, C, T, H, W) -> R3D
            img_tensor = inputs[:, :, 8, :, :] 
        else: # (B, T, C, H, W) -> Others
            img_tensor = inputs[:, 8, :, :, :] 
    else: # (B, C, H, W) -> Spatial
        img_tensor = inputs

    # 2. 역정규화
    img_tensor = denormalize(img_tensor.cpu(), model_name)
    img_tensor = torch.clamp(img_tensor, 0, 1)
    
    # 3. 저장
    grid = make_grid(img_tensor, nrow=2, padding=2)
    np_img = grid.permute(1, 2, 0).numpy()
    
    plt.figure(figsize=(10, 10))
    plt.imshow(np_img)
    plt.axis('off')
    plt.title(f"Model: {model_name} (Sample Frame)")
    plt.savefig(save_path)
    plt.close()

def main():
    print(f"🚀 [Debug] 데이터 로더 V2 점검 (호환성 패치 완료)")
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 1. 파일 리스트 준비
    if not os.path.exists(BASE_DIR):
        print(f"❌ 데이터 경로 없음: {BASE_DIR}")
        return
        
    files, labels, _ = prepare_dataset(BASE_DIR)
    print(f"📂 전체 파일 수: {len(files)} -> 디버깅용 8개 샘플링")
    
    # 샘플링
    sample_files = files[:4] + files[-4:]
    sample_labels = labels[:4] + labels[-4:]

    # 2. 모델별 순회 점검
    for model_name, cfg in GRID_CONFIGS.items():
        print(f"\n{'='*40}")
        print(f"🔍 검사 모델: {model_name.upper()}")
        print(f"{'='*40}")
        
        try:
            # 🔧 [수정된 부분] model_type 대신 model_name 전달
            ds = DeepfakeDataset(
                file_paths=sample_files,
                labels=sample_labels,
                model_name=model_name,  # 👈 여기가 핵심 수정사항!
                mode='train',
                transform=get_transforms(model_name),
                window_size=16
            )
            
            loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
            inputs, targets = next(iter(loader))
            
            # -------------------------------------------------
            # ✅ 체크포인트 1: 입력 차원 (Shape) 검증
            # -------------------------------------------------
            print(f"   📐 Input Shape: {inputs.shape}")
            
            is_valid = False
            # Spatial: (B, 3, H, W)
            if cfg['type'] == 'spatial' and inputs.dim() == 4:
                is_valid = True
            # Temporal (R3D): (B, C, T, H, W) -> dataset.py가 permute 해줌
            elif model_name == 'r3d' and inputs.shape[1] == 3 and inputs.dim() == 5:
                is_valid = True
            # Temporal (Others): (B, T, C, H, W) -> 원본 유지
            elif model_name in ['videomae', 'hybrid'] and inputs.shape[2] == 3 and inputs.dim() == 5:
                is_valid = True
                
            if is_valid:
                print(f"   ✅ [PASS] 차원 규격 정상")
            else:
                print(f"   ❌ [FAIL] 차원 규격 이상 발생! (dataset.py 확인 필요)")

            # -------------------------------------------------
            # ✅ 체크포인트 2: 검정 화면 감지
            # -------------------------------------------------
            is_black, std_val = check_black_screen(inputs.float())
            if is_black:
                print(f"   💀 [CRITICAL] 검정 화면 의심 (std: {std_val:.6f})")
            else:
                print(f"   ✅ [PASS] 이미지 정보량 정상 (std: {std_val:.4f})")

            # -------------------------------------------------
            # ✅ 체크포인트 3: 시각화 저장
            # -------------------------------------------------
            save_path = os.path.join(SAVE_DIR, f"debug_{model_name}.jpg")
            save_debug_batch(inputs, model_name, save_path)
            print(f"   💾 시각화 저장: {save_path}")
            
        except Exception as e:
            print(f"   ❌ [ERROR] {model_name} 로딩 중 치명적 오류: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n✅ 모든 디버깅 완료. '{SAVE_DIR}' 폴더 확인")

if __name__ == "__main__":
    main()