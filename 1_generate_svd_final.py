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
# 🚨 [하드웨어 최적화 패치] 
# RTX 4070S 12GB 환경에서 속도 저하 방지
# ==========================================
if not hasattr(torch, 'xpu'):
    class MockXPU:
        @staticmethod
        def is_available(): return False
        @staticmethod
        def empty_cache(): pass
        @staticmethod
        def device_count(): return 0
        @staticmethod
        def synchronize(): pass
    torch.xpu = MockXPU()

# [경로 설정]
INPUT_IMAGE_DIR = r"C:\Users\leejy\Desktop\test_experiment\dataset\svd_input"
FAKE_VIDEO_DIR = r"C:\Users\leejy\Desktop\test_experiment\dataset\fake"

# [하드웨어 최적화 설정]
torch.backends.cuda.matmul.allow_tf32 = True 
MODEL_ID = "stabilityai/stable-video-diffusion-img2vid-xt"

def main():
    os.makedirs(FAKE_VIDEO_DIR, exist_ok=True)

    # 1. 파이프라인 로드 (fp16 최적화)
    print("🔄 SVD 모델 로드 중...")
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.float16, 
        variant="fp16"
    )
    
    # 🚨 [핵심 최적화 1] 모델 오프로딩 활성화
    # 연산이 끝난 모듈을 즉시 CPU로 넘겨 VRAM을 실시간 확보합니다.
    pipe.enable_model_cpu_offload()
    
    # 🚨 [핵심 최적화 2] 메모리 효율적 어텐션 슬라이싱
    # 4070S의 연산 효율을 위해 메모리 효율적 어텐션을 사용합니다.
    pipe.enable_attention_slicing()

    # 대상 이미지 리스트 확보
    images = [f for f in os.listdir(INPUT_IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    images_to_process = []
    
    for img_name in images:
        save_path = os.path.join(FAKE_VIDEO_DIR, f"svd_{os.path.splitext(img_name)[0]}.mp4")
        if not (os.path.exists(save_path) and os.path.getsize(save_path) > 0):
            images_to_process.append(img_name)
            
    print(f"💎 총 {len(images)}개 중 {len(images_to_process)}개 생성 예정")
    
    for img_name in tqdm(images_to_process, desc="SVD 생성 중"):
        save_path = os.path.join(FAKE_VIDEO_DIR, f"svd_{os.path.splitext(img_name)[0]}.mp4")
        
        try:
            # 1. 이미지 로드
            image_path = os.path.join(INPUT_IMAGE_DIR, img_name)
            image = Image.open(image_path).convert("RGB")
            
            # 2. 비디오 생성 (Rev.18 파라미터 고정)
            # 🔧 [수정] torch.autocast 제거
            # - 파이프라인이 이미 fp16으로 로드되어 있고,
            #   enable_model_cpu_offload()와 autocast가 동시에 작동하면
            #   CPU↔GPU 오프로딩 중 dtype 변환이 중복 발생하여 속도 저하 유발
            generator = torch.manual_seed(42)
            output = pipe(
                image, 
                decode_chunk_size=4,    # VRAM 피크 제어
                num_inference_steps=20, 
                generator=generator,
                motion_bucket_id=127,
                noise_aug_strength=0.02
            )
            frames = output.frames[0]
            
            # 3. 비디오 저장 (25 FPS)
            export_to_video(frames, save_path, fps=25)
            
            # 🚨 [핵심 최적화 3] 루프 종료 시마다 메모리 강제 청소
            # 이 단계가 누락되면 VRAM 파편화로 인해 s/it이 점차 느려집니다.
            del output
            del frames
            gc.collect()             # 파이썬 가비지 컬렉터 실행
            torch.cuda.empty_cache() # PyTorch 캐시 비우기
            torch.cuda.synchronize() # GPU 연산 완료 동기화

        except Exception as e:
            print(f"❌ {img_name} 생성 중 에러 발생: {e}")
            torch.cuda.empty_cache()
            continue

    print("✅ 모든 SVD 영상 생성 완료!")

if __name__ == "__main__":
    main()