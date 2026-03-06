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
# ⚙️ 경로 설정
# ==========================================
REAL_VIDEO_DIR = r"C:\Users\leejy\Desktop\test_experiment\dataset\real"
FAKE_VIDEO_DIR = r"C:\Users\leejy\Desktop\test_experiment\dataset\fake"
TARGET_COUNT = 300
MODEL_ID = "stabilityai/stable-video-diffusion-img2vid-xt"

def prepare_image_for_svd(frame, target_size=(1024, 576)):
    """
    [정석 해결법]
    원본을 억지로 늘리지 않고, 16:9 비율 영역만 중앙에서 잘라낸 뒤 확대합니다.
    인물의 얼굴 형태(기하학적 구조)가 100% 보존됩니다.
    """
    h, w, _ = frame.shape
    target_w, target_h = target_size
    target_aspect = target_w / target_h
    
    # 1. 현재 원본 너비(w)에 맞는 16:9 높이 계산
    new_h = int(w / target_aspect) 
    
    # 2. 상하 중앙 좌표 계산 (Center Crop)
    start_y = (h - new_h) // 2
    
    # 3. 자르기 (이 과정에서 비율 왜곡이 사라짐)
    if start_y < 0: # 혹시 원본이 너무 납작한 경우 대비
        cropped_frame = frame 
    else:
        cropped_frame = frame[start_y:start_y+new_h, :] 

    # 4. SVD 입력 해상도로 리사이즈
    return cv2.resize(cropped_frame, (target_w, target_h))

print(f"💎 SVD 정석 데이터 생성 (비율 보정 + ID 매칭)")
os.makedirs(FAKE_VIDEO_DIR, exist_ok=True)

try:
    pipe = StableVideoDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16, variant="fp16")
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

    # 파일명 매칭: 000.mp4 -> svd_000.mp4
    name_only = os.path.splitext(video_name)[0]
    save_name = f"svd_{name_only}.mp4"
    save_path = os.path.join(FAKE_VIDEO_DIR, save_name)

    if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
        processed_count += 1; pbar.update(1); continue

    try:
        cap = cv2.VideoCapture(os.path.join(REAL_VIDEO_DIR, video_name))
        ret, frame = cap.read(); cap.release()
        if not ret: continue

        # ⚡ 비율 보정 적용
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_np = prepare_image_for_svd(frame) 
        image = Image.fromarray(image_np)

        result = pipe(image, decode_chunk_size=2, num_inference_steps=25, generator=torch.manual_seed(42))
        export_to_video(result.frames[0], save_path, fps=7)
        
        # 메모리 정리
        del result; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
        time.sleep(1) # 발열 제어
        
        processed_count += 1
        pbar.update(1)
        pbar.set_description(f"Done: {save_name}")

    except Exception as e:
        print(f"❌ {save_name} 에러: {e}")
        if "out of memory" in str(e).lower(): break
        continue

pbar.close()