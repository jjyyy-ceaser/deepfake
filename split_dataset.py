import os
import shutil
import random

# ==========================================
# ⚙️ 경로 설정 (선생님 폴더 번호에 맞춤)
# ==========================================
BASE_DIR = "C:/Users/leejy/Desktop/test_experiment/dataset"
SOURCE_REAL_DIR = os.path.join(BASE_DIR, "0_main_train", "real")
SOURCE_FAKE_DIR = os.path.join(BASE_DIR, "0_main_train", "fake")

# 목적지 설정
TRAIN_PURE_DIR = os.path.join(BASE_DIR, "2_exp_train_pure") # 2번으로 변경
TEST_CASE1_DIR = os.path.join(BASE_DIR, "3_test_type1_svd", "case1") # 3번으로 변경

def main():
    # 폴더 생성
    for d in [TRAIN_PURE_DIR, TEST_CASE1_DIR]:
        os.makedirs(os.path.join(d, "real"), exist_ok=True)
        os.makedirs(os.path.join(d, "fake"), exist_ok=True)

    # 파일 리스트 확보 (파일명 기준 정렬)
    real_files = sorted([f for f in os.listdir(SOURCE_REAL_DIR) if f.endswith('.mp4')])
    
    random.seed(42)
    random.shuffle(real_files)

    # 165개 기준 8:2 분할 (학습 132, 테스트 33)
    split_idx = int(len(real_files) * 0.8)
    train_list = real_files[:split_idx]
    test_list = real_files[split_idx:]

    def copy_pairs(file_list, target_root):
        count = 0
        for r_file in file_list:
            file_num = int(os.path.splitext(r_file)[0])
            f_file = f"fake_svd_{file_num + 1:03d}.mp4" # Real 0 -> Fake 001 매칭
            
            src_r = os.path.join(SOURCE_REAL_DIR, r_file)
            src_f = os.path.join(SOURCE_FAKE_DIR, f_file)
            
            if os.path.exists(src_r) and os.path.exists(src_f):
                shutil.copy2(src_r, os.path.join(target_root, "real", r_file))
                shutil.copy2(src_f, os.path.join(target_root, "fake", f_file))
                count += 1
        return count

    print(f"📦 데이터 분리 중... (0번 -> 2번, 3번)")
    tr_cnt = copy_pairs(train_list, TRAIN_PURE_DIR)
    te_cnt = copy_pairs(test_list, TEST_CASE1_DIR)
    
    print(f"✅ 완료! 학습용(2번): {tr_cnt}쌍 / 테스트용(3번): {te_cnt}쌍")

if __name__ == "__main__":
    main()