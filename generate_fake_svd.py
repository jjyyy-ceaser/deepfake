import torch
import os
import cv2
import numpy as np
import gc
import time
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_video
from tqdm import tqdm
from PIL import Image

# ==========================================
# ⚙️ 경로 및 설정
# ==========================================
REAL_VIDEO_DIR = r"C:\Users\leejy\Desktop\test_experiment\dataset\real"
FAKE_VIDEO_DIR = r"C:\Users\leejy\Desktop\test_experiment\dataset\fake"
TARGET_COUNT = 300
MODEL_ID = "stabilityai/stable-video-diffusion-img2vid-xt"

def prepare_image_for_svd(frame, target_size=(1024, 576)):
    """
    512x512 원본에서 16:9 비율(512x288)을 중앙 크롭하여 왜곡 방지
    """
    h, w, _ = frame.shape
    target_w, target_h = target_size
    target_aspect = target_w / target_h
    
    # 1:1 -> 16:9로 만들기 위해 상하를 자름
    new_h = int(w / target_aspect) # 512 / 1.77 = 288
    start_y = (h - new_h) // 2
    cropped = frame[start_y:start_y+new_h, :] 
    
    return cv2.resize(cropped, (target_w, target_h))

print(f"💎 SVD 비율 보정 + 메모리 관리 모드 실행")
os.makedirs(FAKE_VIDEO_DIR, exist_ok=True)

try:
    pipe = StableVideoDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16, variant="fp16")
    pipe.enable_model_cpu_offload()
    pipe.enable_attention_slicing()
except Exception as e:
    print(f"❌ 로딩 실패: {e}"); exit()

real_videos = sorted([f for f in os.listdir(REAL_VIDEO_DIR) if f.endswith('.mp4')])
pbar = tqdm(total=TARGET_COUNT)
count = 0

for video_name in real_videos:
    if count >= TARGET_COUNT: break
    save_path = os.path.join(FAKE_VIDEO_DIR, f"svd_{os.path.splitext(video_name)[0]}.mp4")
    
    if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
        count += 1; pbar.update(1); continue

    try:
        cap = cv2.VideoCapture(os.path.join(REAL_VIDEO_DIR, video_name))
        ret, frame = cap.read(); cap.release()
        if not ret: continue

        # ⚡ 비율 보정 핵심 로직 적용
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(prepare_image_for_svd(frame))

        result = pipe(image, decode_chunk_size=2, num_inference_steps=25, generator=torch.manual_seed(42))
        export_to_video(result.frames[0], save_path, fps=7)
        
        # ⚡ 메모리 청소
        del result; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
        time.sleep(1)
        count += 1; pbar.update(1)
    except Exception as e:
        print(f" 에러: {e}"); continue

pbar.close()