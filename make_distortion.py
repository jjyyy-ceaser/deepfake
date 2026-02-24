import os
import glob
import subprocess
from tqdm import tqdm

# =====================================================================
# 1. 환경 설정 및 경로 정의
# =====================================================================
BASE_DIR = r"C:\Users\leejy\Desktop\test_experiment\dataset\test"
MAIN_CASES = ["case1", "case4"]

# 포렌식 리포트 전체 평균치 기반 정밀 왜곡 프로파일
# 구조: 플랫폼명: (해상도, 비디오코덱, 평균_CRF)
DISTORTION_PROFILES = {
    "youtube": ("640:360", "libx264", "45"),
    "instagram": ("1276:720", "libx264", "35"),
    "kakao_normal": ("1280:720", "libx264", "24"),
    "kakao_high": ("1920:1080", "libx265", "22") # HEVC
}

# =====================================================================
# 2. 5대 핵심 왜곡 요소 통합 FFmpeg 로직
# =====================================================================
def apply_distortion_and_save(input_path, output_path, profile):
    scale, codec, crf = profile
    
    command = [
        "ffmpeg", 
        "-y",                     # 덮어쓰기 허용
        "-i", input_path,         # 입력 파일
        
        # [요소 1] 해상도 강제 조정
        "-vf", f"scale={scale}",  
        
        # [요소 2] 코덱 및 압축률(CRF) 적용
        "-c:v", codec,            
        "-crf", crf,              
        
        # [요소 3] 색상 서브샘플링 훼손 (데이터 압축 최적화)
        "-pix_fmt", "yuv420p",    
        
        # [요소 4] 프레임 레이트 30fps 고정 (모바일 표준화)
        "-r", "30",               
        
        # [요소 5] 메타데이터 헤더 강제 전진 배치 (웹 스트리밍 최적화)
        "-movflags", "+faststart",
        
        "-c:a", "copy",           # 오디오 원본 유지
        output_path               # 출력 파일
    ]
    
    # 프로세스 실행 (터미널 출력 숨김)
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

# =====================================================================
# 3. 폴더 자동 생성 및 데이터 일괄 처리 실행부
# =====================================================================
def generate_datasets_for_all_cases():
    print("🚀 [다중 차원 데이터 왜곡 파이프라인 가동 시작]")
    
    for current_case in MAIN_CASES:
        print("\n" + "="*70)
        print(f"🎬 현재 처리 중인 핵심 조건: [{current_case.upper()}]")
        print("="*70)
        
        for category in ["real", "fake"]:
            # 원본 데이터 읽기 경로: test/case1/raw/real 등
            source_dir = os.path.join(BASE_DIR, current_case, "raw", category)
            video_files = sorted(glob.glob(os.path.join(source_dir, "*.mp4")))
            
            if not video_files:
                print(f"⚠️ {source_dir} 경로에 영상이 없어 건너뜁니다.")
                continue
                
            print(f"\n▶ [{category.upper()}] 데이터 세트 변환 준비 (총 {len(video_files)}개)")
            
            for platform, profile in DISTORTION_PROFILES.items():
                # [자동 폴더 생성 로직] test/case1/youtube/real 등의 구조를 자동 구축
                output_dir = os.path.join(BASE_DIR, current_case, platform, category)
                os.makedirs(output_dir, exist_ok=True)
                
                print(f"  └─ ⚙️ 적용: {platform} (CRF: {profile[2]}, 코덱: {profile[1]}) | 폴더 확인 완료")
                
                # 진행률 표시와 함께 변환 시작
                for video_path in tqdm(video_files, desc=f"[{current_case}] {platform} 변환", leave=False):
                    filename = os.path.basename(video_path)
                    output_path = os.path.join(output_dir, filename)
                    
                    # 이미 변환이 완료된 파일은 건너뛰기 (효율성 확보)
                    if not os.path.exists(output_path):
                        apply_distortion_and_save(video_path, output_path, profile)

    print("\n✅ Case 1 및 Case 4에 대한 모든 플랫폼 변환과 폴더 생성이 완벽히 종료되었습니다.")

if __name__ == "__main__":
    generate_datasets_for_all_cases()