import torch
import os
import cv2
import numpy as np
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_video
from tqdm import tqdm

# ==========================================
# ⚙️ 설정 (고화질 유지)
# ==========================================
BASE_DIR = "C:/Users/leejy/Desktop/test_experiment/dataset"
REAL_VIDEO_DIR = os.path.join(BASE_DIR, "0_main_train", "real")
FAKE_VIDEO_DIR = os.path.join(BASE_DIR, "0_main_train", "fake")
TARGET_COUNT = 300

# ✅ 화질 타협 없음! (XT 모델 사용)
MODEL_ID = "stabilityai/stable-video-diffusion-img2vid-xt"

print(f"💎 SVD 고화질 모드 (메모리 최적화 적용)")
os.makedirs(FAKE_VIDEO_DIR, exist_ok=True)

try:
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.float16, 
        variant="fp16"
    )
    
    # 🚨 [핵심 수정] 강제 GPU 할당(pipe.to("cuda"))을 뺍니다!
    # 대신 라이브러리가 알아서 메모리를 관리하게 맡깁니다.
    # 이렇게 하면 VRAM이 부족해도 느려지지 않고 효율적으로 돌아갑니다.
    pipe.enable_model_cpu_offload()
    
    # 추가 메모리 최적화 (화질 영향 없음)
    pipe.enable_attention_slicing()
    
    print("✅ 모델 로딩 완료! (CPU Offload + Slicing)")

except Exception as e:
    print(f"❌ 오류: {e}")
    exit()

# ==========================================
# 🎬 생성 루프
# ==========================================
real_videos = sorted([f for f in os.listdir(REAL_VIDEO_DIR) if f.endswith('.mp4')])
existing_fakes = [f for f in os.listdir(FAKE_VIDEO_DIR) if f.endswith('.mp4')]
current_count = len(existing_fakes)

print(f"📊 현재 {current_count}개 완료. {TARGET_COUNT}개까지 진행합니다.")
pbar = tqdm(total=TARGET_COUNT, initial=current_count)

count = 0
for video_name in real_videos:
    if count >= TARGET_COUNT:
        break

    file_idx = count + 1
    save_filename = f"fake_svd_{file_idx:03d}.mp4"
    save_path = os.path.join(FAKE_VIDEO_DIR, save_filename)

    # 이미 있으면 패스
    if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
        count += 1
        continue

    try:
        video_path = os.path.join(REAL_VIDEO_DIR, video_name)
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            count += 1
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 화질 유지를 위해 해상도 유지 (1024x576)
        image = cv2.resize(frame, (1024, 576))
        from PIL import Image
        image = Image.fromarray(image)

        # 생성 (Inference)
        # decode_chunk_size=2: 마지막에 비디오 합칠 때 VRAM 터지는 것 방지
        frames = pipe(
            image, 
            decode_chunk_size=2, 
            num_inference_steps=25, # 화질을 위해 25스텝 유지
            generator=torch.manual_seed(42)
        ).frames[0]

        export_to_video(frames, save_path, fps=7)
        
        pbar.update(1)
        pbar.set_description(f"Making {save_filename}")

    except Exception as e:
        print(f"\n❌ 에러: {e}")
        # VRAM 부족 메시지가 뜨면 알려줌
        if "out of memory" in str(e).lower():
            print("🚨 다른 프로그램(유튜브, 크롬 등)을 끄고 다시 시도해보세요.")
        break
    
    count += 1

pbar.close()
print("\n🎉 완료!")