import os
import torch
import timm
from torchvision import models
from transformers import VideoMAEForVideoClassification

# ---------------------------------------------------------
# 1. 경로 설정 및 폴더 생성 (Rev.18 준수)
# ---------------------------------------------------------
CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print(f"🚀 [실험 Rev.18] 모델 가중치 다운로드 시작...")

# ---------------------------------------------------------
# 2. 모델별 가중치 획득 로직
# ---------------------------------------------------------

# ① Xception (FaceForensics++ - 수동 배치 필요)
# FF++ 가중치는 라이브러리 자동 지원이 아니므로 폴더만 준비합니다.
print("\n[1/5] Xception FF++ 가중치 폴더 준비...")
print(f"⚠️  주의: FF++ 가중치 파일(.pth)을 직접 {CHECKPOINT_DIR} 폴더에 넣어주세요.")

# ② Swin Transformer-Tiny (ImageNet-21K)
print("\n[2/5] Swin-T ImageNet-21K 가중치 다운로드 중...")
try:
    # ms_in22k는 ImageNet-21K(22K) 가중치를 호출하는 태그입니다.
    timm.create_model('swin_tiny_patch4_window7_224.ms_in22k', pretrained=True, num_classes=0)
    print("✅ Swin-T 가중치 로드 완료")
except Exception as e:
    print(f"❌ Swin-T 에러: {e}")

# ③ R3D-18 (Kinetics-400)
print("\n[3/5] R3D-18 Kinetics-400 가중치 다운로드 중...")
try:
    weights = models.video.R3D_18_Weights.KINETICS400_V1
    models.video.r3d_18(weights=weights)
    print("✅ R3D-18 가중치 로드 완료")
except Exception as e:
    print(f"❌ R3D-18 에러: {e}")

# ④ VideoMAE V2 (Kinetics-700)
print("\n[4/5] VideoMAE Kinetics-700 가중치 다운로드 중...")
try:
    VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics-700")
    print("✅ VideoMAE 가중치 로드 완료")
except Exception as e:
    print(f"❌ VideoMAE 에러: {e}")

# ⑤ ConvNeXt-Tiny (ImageNet-1K)
print("\n[5/5] ConvNeXt-Tiny ImageNet-1K 가중치 다운로드 중...")
try:
    timm.create_model('convnext_tiny', pretrained=True, num_classes=0)
    print("✅ ConvNeXt 가중치 로드 완료")
except Exception as e:
    print(f"❌ ConvNeXt 에러: {e}")

print(f"\n{'='*50}")
print("🎉 라이브러리 기반 가중치 다운로드 완료!")
print(f"📍 위치: C:\\Users\\leejy\\.cache\\torch\\hub\\checkpoints (자동 캐시)")
print(f"📍 수동 배치: {os.path.abspath(CHECKPOINT_DIR)} 폴더에 Xception FF++ 파일을 넣으세요.")
print(f"{'='*50}")