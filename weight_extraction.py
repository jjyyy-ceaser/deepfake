from transformers import VideoMAEForVideoClassification
import torch

# VideoMAE-Base 모델 로드
model = VideoMAEForVideoClassification.from_pretrained(
    "MCG-NJU/videomae-base", 
    num_labels=2, 
    ignore_mismatched_sizes=True
)

torch.save(model.state_dict(), r'C:\Users\leejy\Desktop\test_experiment\videomae_pretrained.pth')
print("✅ VideoMAE 가중치 추출 완료!")