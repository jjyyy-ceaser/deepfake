import os
import subprocess
import sys
from tqdm import tqdm

# =========================================================
# ⚙️ 설정 (User Configuration)
# =========================================================

# 1. FFmpeg 실행 명령어 설정
# (기본적으로 'ffmpeg'을 사용하되, 혹시 안 되면 절대 경로를 입력하세요)
# 예: FFMPEG_PATH = r"C:\ffmpeg\bin\ffmpeg.exe"
FFMPEG_PATH = "ffmpeg" 

# 2. 폴더 경로 설정
BASE_DIR = "dataset"
TRAIN_SRC = os.path.join(BASE_DIR, "raw_train")  # 학습용 원본 (Real 300, Fake 135)
TEST_SRC = os.path.join(BASE_DIR, "raw_test")    # 테스트용 원본 (SVD 30, Pika 30...)
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_cases")

# 3. 4가지 변형 케이스 정의
CASES = {
    # Case 1: 원본 (변형 없음, 포맷만 통일)
    "case1_original": [], 
    
    # Case 2: 저화질 (360p 해상도)
    "case2_lowres": ["-vf", "scale=-2:360"], 
    
    # Case 3: 고압축 (CRF 40)
    "case3_compress": ["-c:v", "libx264", "-crf", "40"], 
    
    # Case 4: 혼합 (360p + CRF 40) -> Worst Case
    "case4_mixed": ["-vf", "scale=-2:360", "-c:v", "libx264", "-crf", "40"]
}

# =========================================================
# 🚀 실행 로직 (Processing Logic)
# =========================================================

def run_ffmpeg(in_path, out_path, params):
    """FFmpeg 명령어를 생성하고 실행합니다."""
    
    # 기본 명령어 구성 (빠른 변환을 위해 preset fast, 오디오 제거 -an)
    cmd = [FFMPEG_PATH, '-y', '-i', in_path] + params + ['-preset', 'fast', '-an', out_path]
    
    # Case 1(원본)인 경우 재인코딩 없이 복사만 수행 (속도 최적화)
    if not params:
        cmd = [FFMPEG_PATH, '-y', '-i', in_path, '-c', 'copy', '-an', out_path]

    try:
        # 윈도우에서 실행 시 창이 뜨지 않게 설정 (subprocess.DEVNULL)
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        
        if result.returncode != 0:
            print(f"\n❌ [Error] 변환 실패: {in_path}")
            print(f"   Reason: {result.stderr}")
            
    except FileNotFoundError:
        print(f"\n🚨 [System Error] FFmpeg을 찾을 수 없습니다.")
        print(f"   코드 상단의 'FFMPEG_PATH' 변수에 ffmpeg.exe의 전체 경로를 직접 넣어주세요.")
        sys.exit(1)

def process_folder(src_root_dir, type_name):
    """
    폴더 구조를 유지하면서 4가지 케이스로 변환합니다.
    src_root_dir: raw_train 또는 raw_test
    type_name: 'train' 또는 'test' (저장 폴더명)
    """
    if not os.path.exists(src_root_dir):
        print(f"⚠️ 경고: 원본 폴더가 없습니다 -> {src_root_dir}")
        return

    print(f"\n🚀 [{type_name.upper()}] 데이터 전처리 시작...")
    
    # 전체 파일 개수 파악 (Progress Bar용)
    total_files = 0
    for root, _, files in os.walk(src_root_dir):
        total_files += len([f for f in files if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))])

    # 파일 순회 및 변환
    with tqdm(total=total_files * len(CASES), desc=f"Processing {type_name}") as pbar:
        for root, dirs, files in os.walk(src_root_dir):
            for file in files:
                if not file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    continue
                
                src_path = os.path.join(root, file)
                
                # 상대 경로 계산 (예: real, fake, svd/fake ...)
                rel_path = os.path.relpath(root, src_root_dir)
                
                for case_name, params in CASES.items():
                    # 저장 경로 생성: dataset/processed_cases/train/case1_original/real/
                    save_dir = os.path.join(PROCESSED_DIR, type_name, case_name, rel_path)
                    os.makedirs(save_dir, exist_ok=True)
                    
                    dst_path = os.path.join(save_dir, file)
                    
                    # 이미 변환된 파일이 있으면 건너뛰기 (시간 절약)
                    if not os.path.exists(dst_path):
                        run_ffmpeg(src_path, dst_path, params)
                    
                    pbar.update(1)

    print(f"✅ [{type_name.upper()}] 전처리 완료.\n")

def main():
    print(f"🛠️ FFmpeg 경로 확인: {FFMPEG_PATH}")
    print("="*60)
    
    # 1. 학습 데이터 처리 (Real 300, Fake 135)
    process_folder(TRAIN_SRC, "train")
    
    # 2. 테스트 데이터 처리 (SVD, Pika, Runway, FF++)
    process_folder(TEST_SRC, "test")
    
    print("="*60)
    print("🎉 모든 데이터 전처리가 완료되었습니다.")
    print(f"📂 저장 위치: {PROCESSED_DIR}")

if __name__ == "__main__":
    main()