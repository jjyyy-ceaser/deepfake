import os
import subprocess
from tqdm import tqdm

def normalize_videos(input_folder, output_folder):
    # 1. 출력 폴더가 없으면 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"폴더 생성 완료: {output_folder}")

    # 2. 정규화할 파일 목록 가져오기
    files = [f for f in os.listdir(input_folder) if f.endswith(('.mp4', '.mov', '.mkv'))]
    
    if not files:
        print(f"⚠️ '{input_folder}' 폴더에 영상 파일이 없습니다! 파일을 옮겨주세요.")
        return

    print(f"총 {len(files)}개의 영상 정규화를 시작합니다.")

    # 3. 루프를 돌며 FFmpeg 실행
    for file in tqdm(files, desc="정규화 진행 중"):
        input_path = os.path.join(input_folder, file)
        output_path = os.path.join(output_folder, f"std_{file}")
        
        # 연구용 표준 규격 설정 (1080p, 30fps, 10초, CRF 18)
        command = [
            'ffmpeg', '-i', input_path,
            '-vf', 'scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,fps=30',
            '-t', '10',
            '-c:v', 'libx264', '-crf', '18', '-preset', 'slower',
            '-y', output_path
        ]
        
        # 프로세스 실행 (에러 발생 시 출력)
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"\n❌ 에러 발생 ({file}): {result.stderr}")

# --- 여기서부터 실제 실행부 (Main) ---
if __name__ == "__main__":
    # 연구원님의 폴더 구조에 맞게 설정
    RAW_DIR = "raw_samples"       # 원본 영상 30개를 이 폴더에 넣으세요
    OUT_DIR = "normalized_samples" # 정규화된 영상이 저장될 곳
    
    # 함수 실행
    normalize_videos(RAW_DIR, OUT_DIR)
    
    print("\n✅ 모든 작업이 완료되었습니다!")