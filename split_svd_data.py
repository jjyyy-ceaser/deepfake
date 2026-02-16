import os
import shutil
import random

# 경로 설정 (사용자 환경에 맞게 수정)
BASE_DIR = "dataset"
TRAIN_FAKE_DIR = os.path.join(BASE_DIR, "raw_train", "fake") # 여기에 165개가 있다고 가정
TEST_SVD_DIR = os.path.join(BASE_DIR, "raw_test", "svd")

def split_data():
    # 1. 폴더 확인
    if not os.path.exists(TRAIN_FAKE_DIR):
        print(f"❌ 오류: 학습용 Fake 폴더가 없습니다: {TRAIN_FAKE_DIR}")
        return
    
    os.makedirs(TEST_SVD_DIR, exist_ok=True)
    
    # 2. 파일 리스트 가져오기
    files = [f for f in os.listdir(TRAIN_FAKE_DIR) if f.endswith(('.mp4', '.avi', '.mov'))]
    total_files = len(files)
    
    print(f"📦 현재 Fake 데이터 총 개수: {total_files}개")
    
    if total_files < 165:
        print("⚠️ 경고: 데이터가 165개보다 적습니다. 경로를 확인하세요.")
    
    # 3. 이미 분할되어 있는지 확인
    test_files = os.listdir(TEST_SVD_DIR)
    if len(test_files) >= 30:
        print("✅ 이미 테스트 폴더에 데이터가 30개 이상 있습니다. 분할을 건너뜁니다.")
        return

    # 4. 랜덤으로 30개 선택하여 이동 (Move)
    random.seed(42) # 재현성을 위해 시드 고정
    move_files = random.sample(files, 30)
    
    print(f"🚀 30개를 뽑아 테스트 폴더로 이동합니다...")
    
    for f in move_files:
        src = os.path.join(TRAIN_FAKE_DIR, f)
        dst = os.path.join(TEST_SVD_DIR, f)
        shutil.move(src, dst) # 복사가 아니라 '이동'입니다!
        
    # 5. 결과 확인
    train_cnt = len(os.listdir(TRAIN_FAKE_DIR))
    test_cnt = len(os.listdir(TEST_SVD_DIR))
    
    print("-" * 30)
    print(f"🎉 분할 완료!")
    print(f"   - 학습용(Train) Fake 남은 개수: {train_cnt}개 (Real 300개와 학습)")
    print(f"   - 테스트용(Test) SVD 이동 개수: {test_cnt}개")
    print("-" * 30)

if __name__ == "__main__":
    split_data()