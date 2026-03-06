import torch
import os
import cv2
import numpy as np
import gc
# time 모듈 삭제 (휴식 없이 풀가동)
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
    [비율 보정 유지]
    속도는 올리되, 비율은 1:1로 정확하게 유지합니다.
    """
    h, w, _ = frame.shape
    target_w, target_h = target_size
    target_aspect = target_w / target_h
    
    new_h = int(w / target_aspect) 
    start_y = (h - new_h) // 2
    
    if start_y < 0:
        cropped_frame = frame
    else:
        cropped_frame = frame[start_y:start_y+new_h, :] 

    return cv2.resize(cropped_frame, (target_w, target_h))

print(f"💎 SVD 고속 생성 모드 (비율 보정 O, 속도 제한 해제)")
os.makedirs(FAKE_VIDEO_DIR, exist_ok=True)

try:
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.float16, 
        variant="fp16"
    )
    # CPU Offload는 유지해야 12GB에서 돌아갑니다
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

    name_only = os.path.splitext(video_name)[0]
    save_name = f"svd_{name_only}.mp4"
    save_path = os.path.join(FAKE_VIDEO_DIR, save_name)

    if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
        processed_count += 1; pbar.update(1); continue

    try:
        cap = cv2.VideoCapture(os.path.join(REAL_VIDEO_DIR, video_name))
        ret, frame = cap.read(); cap.release()
        if not ret: continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_np = prepare_image_for_svd(frame) 
        image = Image.fromarray(image_np)

        # ⚡ [속도 복구 핵심] decode_chunk_size를 8로 상향
        # 아까 2였던 것을 8로 올려서 처리 속도를 대폭 높입니다.
        generator = torch.manual_seed(42)
        result = pipe(
            image, 
            decode_chunk_size=8, 
            num_inference_steps=25, 
            generator=generator
        )
        export_to_video(result.frames[0], save_path, fps=7)
        
        # 메모리 정리 (속도를 위해 sleep 삭제)
        del result, image, image_np
        gc.collect()                 
        torch.cuda.empty_cache()     
        torch.cuda.synchronize()     
        
        processed_count += 1
        pbar.update(1)
        pbar.set_description(f"Done: {save_name}")

    except Exception as e:
        print(f"\n❌ {save_name} 에러: {e}")
        # 혹시 속도 높이다가 메모리 부족 뜨면 그때만 알려줌
        if "out of memory" in str(e).lower():
            print("🚨 VRAM 부족 발생. (속도가 너무 빨라 메모리가 찼습니다)")
            break
        continue

pbar.close()