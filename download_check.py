# download_check.py
import torch
import timm
from torchvision import models
from transformers import VideoMAEForVideoClassification
import os

print("🚀 모델 다운로드 테스트 시작...")

# 1. Timm Models (Xception, ConvNext, Swin)
print("\n1. Xception 다운로드 중... (timm)")
try:
    m = timm.create_model('xception', pretrained=True, num_classes=2)
    print("✅ Xception 완료.")
except Exception as e: print(f"❌ 실패: {e}")

print("\n2. ConvNext 다운로드 중... (timm)")
try:
    m = timm.create_model('convnext_tiny', pretrained=True, num_classes=2)
    print("✅ ConvNext 완료.")
except Exception as e: print(f"❌ 실패: {e}")

print("\n3. Swin Transformer 다운로드 중... (timm)")
try:
    m = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=2)
    print("✅ Swin 완료.")
except Exception as e: print(f"❌ 실패: {e}")

# 2. Torchvision Models (R3D, R2Plus1D)
print("\n4. R3D_18 다운로드 중... (torchvision)")
try:
    m = models.video.r3d_18(weights='KINETICS400_V1')
    print("✅ R3D_18 완료.")
except Exception as e: print(f"❌ 실패: {e}")

print("\n5. R2Plus1D 다운로드 중... (torchvision)")
try:
    m = models.video.r2plus1d_18(weights='KINETICS400_V1')
    print("✅ R2Plus1D 완료.")
except Exception as e: print(f"❌ 실패: {e}")

# 3. Transformers (VideoMAE)
print("\n6. VideoMAE 다운로드 중... (HuggingFace)")
try:
    m = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base", num_labels=2, ignore_mismatched_sizes=True)
    print("✅ VideoMAE 완료.")
except Exception as e: print(f"❌ 실패: {e}")

print("\n✨ 모든 모델 준비 완료! 이제 학습 코드를 다시 실행하세요.")