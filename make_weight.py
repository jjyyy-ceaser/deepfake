import os
import torch
import timm
import torchvision.models.video as video_models
from transformers import VideoMAEConfig, VideoMAEForVideoClassification

# [1] 경로 설정 (OneDrive 경로가 맞는지 확인하세요)
# 실험 폴더와 일치시키기 위해 바탕화면의 test_experiment 폴더로 잡는 것이 좋습니다.
WEIGHTS_DIR = r"C:\Users\leejy\Desktop\test_experiment\weights" 
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# [2] 환경 변수 설정 (다운로드 경로 강제)
os.environ['TORCH_HOME'] = WEIGHTS_DIR
os.environ['HUGGINGFACE_HUB_CACHE'] = os.path.join(WEIGHTS_DIR, "huggingface")

def download_all_weights():
    print(f"📦 가중치 다운로드 시작... 저장 경로: {WEIGHTS_DIR}")

    # 1. Swin Transformer (ImageNet-21K Pre-trained)
    # timm의 swin_tiny는 21K에서 학습하고 1K로 튜닝된 가중치를 가져옵니다. (설계안 부합)
    print("\n[1/4] Swin-T (ImageNet-21K knowledge) 확보 중...")
    timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)

    # 2. ConvNeXt (ImageNet-1K)
    print("[2/4] ConvNeXt-Tiny (ImageNet-1K) 확보 중...")
    timm.create_model('convnext_tiny', pretrained=True)

    # 3. R3D-18 (Kinetics-400)
    print("[3/4] R3D-18 (Kinetics-400) 확보 중...")
    # weights 매개변수를 사용하여 명시적으로 버전을 지정합니다.
    video_models.r3d_18(weights=video_models.R3D_18_Weights.KINETICS400_V1)

    # 4. VideoMAE (Kinetics-400, V1)
    # *설계안 변경 제안: V2(K-700) 대신 라이브러리 호환성이 좋은 V1(K-400)을 사용합니다.
    print("[4/4] VideoMAE V1 (Kinetics-400) 확보 중...")
    VideoMAEForVideoClassification.from_pretrained(
        "MCG-NJU/videomae-base-finetuned-kinetics",
        cache_dir=os.environ['HUGGINGFACE_HUB_CACHE']
    )

    print("\n✅ 모든 모델의 사전 학습 가중치 확보 완료!")
    print(f"👉 Xception 가중치({WEIGHTS_DIR}\\xception_ffpp.pth)는 수동으로 넣어두셨죠?")

if __name__ == "__main__":
    download_all_weights()