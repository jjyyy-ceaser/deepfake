import os
import subprocess
from tqdm import tqdm
import glob

# =========================================================
# 📂 경로 설정
# =========================================================
# 1. 원본 영상(30개)을 넣을 폴더 (이 폴더가 없으면 만드세요)pyt
INPUT_DIR = "raw_samples"  

# 2. 전처리된 결과물이 저장될 폴더 (자동 생성됨)
OUTPUT_DIR = "dataset/sns_analysis/00_Original" 

# =========================================================
# ⚙️ 전처리 규격 (Golden Standard for Forensic Research)
# =========================================================
TARGET_RES = "1920:1080"  # FHD 표준
TARGET_FPS = "30"         # 고정 프레임 (CFR)
TARGET_CRF = "18"         # 시각적 무손실 (Visually Lossless)
TARGET_TIME = "10"        # 분석 효율을 위한 10초 컷

def preprocess_video(input_path, output_path):
    """
    FFmpeg를 사용하여 영상을 연구용 표준 규격으로 강제 변환
    Key Features: Letterbox Padding, CFR, YUV420P, High Profile
    """
    cmd = [
        "ffmpeg", 
        "-y",                               # 덮어쓰기 허용
        "-i", input_path,                   # 입력 파일
        
        # 🎥 [핵심 필터 체인]
        # 1. scale: 비율 유지하며 1920x1080 안에 맞춤 (줄이거나 늘림)
        # 2. pad: 남는 공간을 검은색(Letterbox)으로 채워 정확히 1080p 맞춤
        # 3. fps: 30fps 고정 (VFR 제거)
        # 4. format: yuv420p 픽셀 포맷 강제 (SNS 업로드 호환성 100% 보장)
        "-vf", f"scale={TARGET_RES}:force_original_aspect_ratio=decrease,pad={TARGET_RES}:(ow-iw)/2:(oh-ih)/2,fps={TARGET_FPS},format=yuv420p",
        
        "-t", TARGET_TIME,                  # 앞부분 10초만 사용
        
        "-c:v", "libx264",                  # 코덱: H.264 (AVC)
        "-profile:v", "high",               # 프로파일: High (고화질)
        "-crf", TARGET_CRF,                 # 화질: 18 (원본 보존)
        
        "-c:a", "aac",                      # 오디오: AAC
        "-b:a", "128k",                     # 오디오 비트레이트: 128k
        "-ac", "2",                         # 오디오 채널: Stereo
        
        "-movflags", "+faststart",          # 웹 최적화 (메타데이터 앞쪽 배치)
        output_path
    ]
    
    # 실행 (로그는 에러만 출력하여 깔끔하게)
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    # 폴더 생성
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"📁 결과 폴더 생성 완료: {OUTPUT_DIR}")
    
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)
        print(f"⚠️ '{INPUT_DIR}' 폴더가 없습니다. 폴더를 생성했으니 영상 30개를 여기에 넣어주세요!")
        return

    # 지원 파일 확장자
    raw_files = glob.glob(os.path.join(INPUT_DIR, "*.*"))
    valid_exts = ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v']
    target_files = [f for f in raw_files if os.path.splitext(f)[1].lower() in valid_exts]
    
    if not target_files:
        print(f"⚠️ '{INPUT_DIR}' 폴더에 영상 파일이 없습니다.")
        return
    
    print("="*60)
    print(f"🧹 [Standardization] 영상 전처리 시작 (총 {len(target_files)}개)")
    print(f"🎯 규격: {TARGET_RES} | {TARGET_FPS}fps | H.264 High | CRF {TARGET_CRF} | YUV420P")
    print("="*60)
    
    success_count = 0
    
    for file_path in tqdm(target_files, desc="Processing"):
        filename = os.path.basename(file_path)
        name_only = os.path.splitext(filename)[0]
        output_path = os.path.join(OUTPUT_DIR, f"{name_only}.mp4")
        
        if preprocess_video(file_path, output_path):
            success_count += 1
        else:
            print(f"❌ 실패: {filename}")
            
    print("\n" + "="*60)
    print(f"✨ 전처리 완료! 성공: {success_count} / 전체: {len(target_files)}")
    print(f"📂 결과물 위치: {OUTPUT_DIR}")
    print("👉 이제 이 파일들을 각 SNS 플랫폼에 업로드하세요.")
    print("="*60)

if __name__ == "__main__":
    main()