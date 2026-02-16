import os
import subprocess
from tqdm import tqdm

# === 설정 ===
BASE_DIR = "dataset"
TRAIN_SRC = os.path.join(BASE_DIR, "raw_train")
TEST_SRC = os.path.join(BASE_DIR, "raw_test")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_cases")

# 4가지 Case 정의
CASES = {
    "case1_original": [], # 변형 없음
    "case2_lowres": ["-vf", "scale=-2:360"], # 360p
    "case3_compress": ["-c:v", "libx264", "-crf", "40"], # CRF 40
    "case4_mixed": ["-vf", "scale=-2:360", "-c:v", "libx264", "-crf", "40"] # 둘 다
}

def run_ffmpeg(in_path, out_path, params):
    cmd = ['ffmpeg', '-y', '-i', in_path] + params + ['-preset', 'fast', '-an', out_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def process_folder(src_root, type_name):
    # type_name: 'train' 또는 'test'
    print(f"🚀 Processing {type_name} data...")
    
    for root, dirs, files in os.walk(src_root):
        for file in files:
            if not file.lower().endswith(('.mp4', '.avi', '.mov')): continue
            
            src_path = os.path.join(root, file)
            rel_path = os.path.relpath(root, src_root) # 예: real, fake, svd, pika...
            
            for case_name, params in CASES.items():
                # 저장 경로: dataset/processed_cases/train/case1_original/real/파일명.mp4
                save_dir = os.path.join(PROCESSED_DIR, type_name, case_name, rel_path)
                os.makedirs(save_dir, exist_ok=True)
                dst_path = os.path.join(save_dir, file)
                
                if not os.path.exists(dst_path):
                    if case_name == "case1_original":
                        # 원본은 복사 대신 심볼릭 링크(또는 단순 복사)
                        run_ffmpeg(src_path, dst_path, []) 
                    else:
                        run_ffmpeg(src_path, dst_path, params)

    print(f"✅ {type_name} 완료.")

if __name__ == "__main__":
    process_folder(TRAIN_SRC, "train")
    process_folder(TEST_SRC, "test")