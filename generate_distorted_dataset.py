import os
import subprocess
import glob
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# =========================================================
# ⚙️ [설정] 경로 및 "Full-Stack" 왜곡 파라미터 (Rev.21)
# =========================================================
BASE_DIR = r"C:\Users\leejy\Desktop\test_experiment\dataset\final_dataset_v2"
SRC_ROOT = os.path.join(BASE_DIR, "raw", "test")

# FFmpeg 경로 (환경 변수에 없다면 절대 경로 입력)
FFMPEG_CMD = "ffmpeg"

# 🔧 [안전장치] 비트레이트 파싱 헬퍼
def parse_bitrate(value_str):
    """ '2200k', '2M' 등을 정수(bps)로 변환 """
    value_str = str(value_str).strip().lower()
    if value_str.endswith('k'):
        return int(float(value_str[:-1]) * 1000)
    elif value_str.endswith('m'):
        return int(float(value_str[:-1]) * 1000000)
    elif value_str.isdigit():
        return int(value_str)
    return 2000000 # Default fallback

# 🔧 [핵심] 플랫폼별 정밀 인코딩 설정 (The Matrix of Distortion)
PLATFORM_CONFIGS = {
    "instagram": {
        # ✅ [Fix] 비율 유지 + 패딩 (Letterbox): 16:9 영상을 640x640 정사각형 중앙에 배치
        "scale": "scale=640:640:force_original_aspect_ratio=decrease,pad=640:640:(ow-iw)/2:(oh-ih)/2",
        "bitrate": "2200k",
        "fps": 30,
        "gop": "30",
        "bframes": "0",
        "profile": "main",      # Main Profile (No B-frame conflict check needed, but kept 0 for mobile)
        "preset": "medium"
    },
    "youtube": {
        # FHD 유지 (16:9 -> 16:9 이므로 패딩 불필요)
        "scale": "scale=1920:1080",
        "bitrate": "500k",
        "fps": 30,
        "gop": "60",
        "bframes": "2",         # ✅ [Check] High Profile은 B-frame 지원
        "profile": "high",
        "preset": "slow"
    },
    "kakao_high": {
        "scale": "scale=1920:1080",
        "bitrate": "2700k",
        "fps": 30,
        "gop": "30",
        "bframes": "0",
        "profile": "high",
        "preset": "fast"
    },
    "kakao_normal": {
        "scale": "scale=960:540",
        "bitrate": "670k",
        "fps": 24,              # [Thesis Point] 24fps Temporal Drop
        "gop": "30",
        "bframes": "0",         # ✅ [Check] Baseline Profile은 B-frame 미지원 (0 필수)
        "profile": "baseline",
        "preset": "veryfast"
    }
}

def process_video_advanced(task_info):
    """
    [Rev.21] 무결성 검증 완료: Type-Safe, Profile-Safe, Ratio-Safe
    """
    src_path, dst_path, cfg = task_info
    
    if os.path.exists(dst_path):
        return

    try:
        # 1. 비트레이트 및 버퍼 계산 (Integer 연산)
        target_bps = parse_bitrate(cfg['bitrate'])
        buf_size = target_bps * 2  # 정수 상태에서 연산
        
        # 2. FFmpeg 명령어 조립
        cmd = [
            FFMPEG_CMD, "-y", 
            "-i", src_path,
            
            # [영상 필터] Smart Scale & Padding
            "-vf", cfg['scale'],
            
            # [프레임레이트]
            "-r", str(cfg['fps']),
            
            # [비트레이트 제어] 문자열로 변환하여 전달
            "-b:v", str(target_bps),
            "-maxrate", str(target_bps),
            "-bufsize", str(buf_size),
            
            # [코덱 및 프로파일]
            "-c:v", "libx264",
            "-profile:v", cfg['profile'],
            "-preset", cfg['preset'],
            
            # [GOP & B-Frame 구조]
            "-g", str(cfg['gop']),
            "-keyint_min", str(cfg['gop']),
            "-bf", str(cfg['bframes']),
            
            # [공통 설정]
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",  # 포렌식 지문 (Moov 앞쪽 이동)
            "-an",                      # 오디오 제거 (SVD 전용)
            "-loglevel", "error",
            
            dst_path
        ]
        
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
    except subprocess.CalledProcessError:
        print(f"❌ 변환 실패 (FFmpeg Error): {os.path.basename(src_path)}")
    except Exception as e:
        print(f"⚠️ 시스템 오류: {e} | 파일: {os.path.basename(src_path)}")

def main():
    print(f"🚀 [Rev.21 Final] 완벽하게 통제된 왜곡 데이터셋 생성 시작")
    print(f"📂 원본 경로: {SRC_ROOT}")
    print(f"✨ 특징: Smart Padding(Insta), Safe Buffer Calc, Profile Consistency")
    
    tasks = []

    # 1. 작업 큐 생성
    for platform, cfg in PLATFORM_CONFIGS.items():
        print(f"\n🌍 Target Platform: {platform.upper()}")
        print(f"   ⚙️  Config: {cfg['scale']} | FPS: {cfg['fps']} | Profile: {cfg['profile']}")
        
        for label in ["real", "fake"]:
            src_dir = os.path.join(SRC_ROOT, label)
            dst_dir = os.path.join(BASE_DIR, platform, label)
            
            os.makedirs(dst_dir, exist_ok=True)
            
            if not os.path.exists(src_dir):
                print(f"   ⚠️  경로 없음: {src_dir}")
                continue

            files = glob.glob(os.path.join(src_dir, "*.mp4"))
            for src_path in files:
                filename = os.path.basename(src_path)
                dst_path = os.path.join(dst_dir, filename)
                tasks.append((src_path, dst_path, cfg))

    # 2. 병렬 처리 실행
    if not tasks:
        print("❌ 처리할 작업이 없습니다.")
        return

    print(f"\n🔥 총 {len(tasks)}개 영상 생성 중 (Multi-threading)...")
    
    # I/O 바운드 작업이므로 코어 수보다 넉넉하게 잡아도 됨 (8~16)
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(tqdm(executor.map(process_video_advanced, tasks), total=len(tasks), unit="vid"))

    print(f"\n✅ 데이터셋 구축 완료! 위치: {BASE_DIR}")

if __name__ == "__main__":
    main()