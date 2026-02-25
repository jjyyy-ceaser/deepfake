import os
import shutil
import random
import subprocess

# ==========================================
# 1. 환경 및 경로 설정
# ==========================================
RAW_BASE = r"C:\Users\leejy\Desktop\test_experiment\dataset\raw_data"
OUTPUT_BASE = r"C:\Users\leejy\Desktop\test_experiment\dataset\split_datasets"

TRAIN_COUNT = 135
TEST_COUNT = 30
SEED = 42

def process_video_ffmpeg(input_path, output_path):
    """FFmpeg를 사용하여 영상을 360p 해상도 및 CRF 40으로 강제 압축합니다."""
    command = [
        'ffmpeg', '-y', '-i', input_path,
        '-vf', 'scale=-2:360',  # 세로 360 픽셀 고정, 가로 비율 유지
        '-c:v', 'libx264', '-crf', '40', # 극단적 압축 손실 발생
        '-preset', 'fast',
        '-c:a', 'copy',         # 오디오 손실 없이 복사
        output_path
    ]
    # 실행 중 발생하는 콘솔 로그를 숨김 처리 (깔끔한 출력을 위해)
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def build_datasets():
    # ---------------------------------------------------------
    # Step 1. 절대 오차 없는 파일 매칭 (이름표 기준)
    # ---------------------------------------------------------
    random.seed(SEED)
    
    raw_real_dir = os.path.join(RAW_BASE, "real")
    raw_fake_dir = os.path.join(RAW_BASE, "fake")
    
    real_files = os.listdir(raw_real_dir)
    fake_files = os.listdir(raw_fake_dir)
    
    paired_files = []
    missing_pairs = 0

    print("🔍 원본 데이터 쌍(Pair) 검증 중...")
    
    for real_name in real_files:
        if not real_name.endswith('.mp4'): continue
            
        # Real 번호 추출 (예: '00000.mp4' -> 0)
        real_idx = int(real_name.split('.')[0])
        
        # Fake 짝꿍 이름 계산 (예: 0 + 1 -> 'fake_svd_001.mp4')
        fake_name = f"fake_svd_{real_idx + 1:03d}.mp4"
        
        # 실제 Fake 파일이 존재하는지 검증 후 결합
        if fake_name in fake_files:
            paired_files.append((real_name, fake_name))
        else:
            missing_pairs += 1
            print(f"  ⚠️ 짝꿍 누락: {real_name}의 짝인 {fake_name}이 존재하지 않습니다.")

    print(f"✅ 완벽하게 짝이 맞는 데이터: 총 {len(paired_files)}쌍 (누락: {missing_pairs}건)")
    
    if len(paired_files) < (TRAIN_COUNT + TEST_COUNT):
        print(f"\n🚨 에러: 온전한 쌍({len(paired_files)}개)이 분할 목표치({TRAIN_COUNT + TEST_COUNT}개)보다 부족하여 작업을 중단합니다.")
        return

    # ---------------------------------------------------------
    # Step 2. 데이터 무작위 셔플 및 Train/Test 분할
    # ---------------------------------------------------------
    random.shuffle(paired_files)
    train_pairs = paired_files[:TRAIN_COUNT]
    test_pairs = paired_files[TRAIN_COUNT:TRAIN_COUNT + TEST_COUNT]

    # ---------------------------------------------------------
    # Step 3. Dataset A (원본 품질) 구축
    # ---------------------------------------------------------
    print(f"\n[1/2] Dataset A (원본 품질) 구축 중... (Train: {len(train_pairs)}쌍, Test: {len(test_pairs)}쌍)")
    for split_name, pairs in [("train", train_pairs), ("test", test_pairs)]:
        dir_real = os.path.join(OUTPUT_BASE, "dataset_A", split_name, "real")
        dir_fake = os.path.join(OUTPUT_BASE, "dataset_A", split_name, "fake")
        os.makedirs(dir_real, exist_ok=True)
        os.makedirs(dir_fake, exist_ok=True)
        
        for real_name, fake_name in pairs:
            shutil.copy2(os.path.join(raw_real_dir, real_name), os.path.join(dir_real, real_name))
            shutil.copy2(os.path.join(raw_fake_dir, fake_name), os.path.join(dir_fake, fake_name))
            
    print("✅ Dataset A 복사 완료.")

    # ---------------------------------------------------------
    # Step 4. Dataset B (극한 왜곡 적용) 구축 - Test 쌍만 처리
    # ---------------------------------------------------------
    print(f"\n[2/2] Dataset B (360p, CRF40 왜곡) 구축 중... (Test: {len(test_pairs)}쌍)")
    print("      이 작업은 FFmpeg 인코딩을 거치므로 CPU 성능에 따라 몇 분 정도 소요될 수 있습니다.")
    
    dir_real_b = os.path.join(OUTPUT_BASE, "dataset_B", "test", "real")
    dir_fake_b = os.path.join(OUTPUT_BASE, "dataset_B", "test", "fake")
    os.makedirs(dir_real_b, exist_ok=True)
    os.makedirs(dir_fake_b, exist_ok=True)
    
    for idx, (real_name, fake_name) in enumerate(test_pairs, 1):
        print(f"  -> 왜곡 인코딩 진행 중... ({idx}/{TEST_COUNT})", end="\r")
        
        src_real = os.path.join(raw_real_dir, real_name)
        src_fake = os.path.join(raw_fake_dir, fake_name)
        
        dst_real = os.path.join(dir_real_b, real_name)
        dst_fake = os.path.join(dir_fake_b, fake_name)
        
        process_video_ffmpeg(src_real, dst_real)
        process_video_ffmpeg(src_fake, dst_fake)
        
    print("\n✅ Dataset B 왜곡 및 생성 완료.")
    print(f"\n✨ 모든 파이프라인 작업이 성공적으로 끝났습니다!\n경로: {OUTPUT_BASE}")

if __name__ == "__main__":
    build_datasets()