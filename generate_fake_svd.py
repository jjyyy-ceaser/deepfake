import torch
import os
import cv2
import numpy as np
import gc
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_video
from tqdm import tqdm
from PIL import Image

# ==========================================
# ⚙️ VRAM 12GB 최적화 설정
# ==========================================
torch.backends.cuda.matmul.allow_tf32 = True 

REAL_VIDEO_DIR = r"C:\Users\leejy\Desktop\test_experiment\dataset\real"
FAKE_VIDEO_DIR = r"C:\Users\leejy\Desktop\test_experiment\dataset\fake"
TARGET_COUNT = 300
MODEL_ID = "stabilityai/stable-video-diffusion-img2vid-xt"

def prepare_image_for_svd(frame, target_size=(1024, 576)):
    """ [비율 보정] 원본 1:1 유지하며 Center Crop """
    h, w, _ = frame.shape
    target_w, target_h = target_size
    target_aspect = target_w / target_h
    new_h = int(w / target_aspect) 
    start_y = (h - new_h) // 2
    cropped_frame = frame[max(0, start_y):min(h, start_y+new_h), :] 
    return cv2.resize(cropped_frame, (target_w, target_h))

print(f"💎 SVD 안정 생성 모드 (VRAM 스와핑 방지)")
os.makedirs(FAKE_VIDEO_DIR, exist_ok=True)

try:
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, variant="fp16"
    )
    # 🚨 핵심: 공유 메모리 사용을 막기 위해 VRAM 사용량 절감
    pipe.enable_model_cpu_offload() 
    pipe.enable_attention_slicing() 
    print("✅ 모델 준비 완료")
except Exception as e:
    print(f"❌ 오류: {e}"); exit()

real_videos = sorted([f for f in os.listdir(REAL_VIDEO_DIR) if f.endswith('.mp4')])
pbar = tqdm(total=TARGET_COUNT)
processed_count = 0

for video_name in real_videos:
    if processed_count >= TARGET_COUNT: break
    save_path = os.path.join(FAKE_VIDEO_DIR, f"svd_{os.path.splitext(video_name)[0]}.mp4")
    if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
        processed_count += 1; pbar.update(1); continue

    try:
        cap = cv2.VideoCapture(os.path.join(REAL_VIDEO_DIR, video_name))
        ret, frame = cap.read(); cap.release()
        if not ret: continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(prepare_image_for_svd(frame))

        # ⚡ decode_chunk_size=4: VRAM 12GB에서 스와핑 없이 가장 빠른 타협점
        generator = torch.manual_seed(42)
        result = pipe(
            image, decode_chunk_size=4, num_inference_steps=20, generator=generator
        )
        export_to_video(result.frames[0], save_path, fps=7)
        
        # 🧹 다음 영상 생성 전 VRAM 찌꺼기 완벽 제거
        del result, image
        gc.collect()                 
        torch.cuda.empty_cache()     
        torch.cuda.synchronize()     
        
        processed_count += 1
        pbar.update(1)
        pbar.set_description(f"Done: {os.path.basename(save_path)}")

    except Exception as e:
        print(f"\n❌ 오류: {e}"); continue
pbar.close()