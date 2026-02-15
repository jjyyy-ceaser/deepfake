import os
import shutil
import random
from tqdm import tqdm

# 경로 설정 - 빨간 줄이 계속 뜨면 이 줄만 직접 타이핑해 보세요.
BASE_DIR = r"C:\Users\leejy\Desktop\test_experiment\dataset"
SRC_REAL = os.path.join(BASE_DIR, "0_main_train", "real")
SRC_FAKE = os.path.join(BASE_DIR, "0_main_train", "fake")
DST_ROOT = os.path.join(BASE_DIR, "2_exp_train_pure")

def main():
    if os.path.exists(DST_ROOT):
        shutil.rmtree(DST_ROOT)

    for split in ["train", "test"]:
        for cls in ["real", "fake"]:
            os.makedirs(os.path.join(DST_ROOT, split, cls), exist_ok=True)

    valid_pairs = []
    print("🔎 165쌍 전수 조사를 시작합니다...")

    # 1부터 165까지 루프
    for i in range(1, 166):
        # 가짜: fake_svd_001.mp4 (3자리 패딩)
        # 진짜: 00000.mp4 (5자리 패딩)
        f_name = f"fake_svd_{i:03d}.mp4" 
        r_name = f"{i-1:05d}.mp4"

        f_path = os.path.join(SRC_FAKE, f_name)
        r_path = os.path.join(SRC_REAL, r_name)

        # 파일이 존재하면 추가
        if os.path.exists(f_path) and os.path.exists(r_path):
            valid_pairs.append((r_name, f_name))
        else:
            # 혹시 확장자가 없는 파일명일 경우를 대비한 2차 체크
            f_name_alt = f"fake_svd_{i:03d}"
            r_name_alt = f"{i-1:05d}"
            if os.path.exists(os.path.join(SRC_FAKE, f_name_alt)) and \
               os.path.exists(os.path.join(SRC_REAL, r_name_alt)):
                valid_pairs.append((r_name_alt, f_name_alt))

    print(f"✅ 드디어 찾았습니다: {len(valid_pairs)}쌍")

    if len(valid_pairs) != 165:
        print(f"⚠️ 경고: {165 - len(valid_pairs)}쌍이 누락되었습니다. 경로를 다시 확인하세요.")

    # 8:2 분할 (Train 132 / Test 33)
    random.seed(42)
    random.shuffle(valid_pairs)
    split_idx = int(len(valid_pairs) * 0.8)
    train_p, test_p = valid_pairs[:split_idx], valid_pairs[split_idx:]

    def copy_files(pairs, split_name):
        for r_f, f_f in tqdm(pairs, desc=f"📦 {split_name} 복사"):
            shutil.copy2(os.path.join(SRC_REAL, r_f), os.path.join(DST_ROOT, split_name, "real", r_f))
            shutil.copy2(os.path.join(SRC_FAKE, f_f), os.path.join(DST_ROOT, split_name, "fake", f_f))

    copy_files(train_p, "train")
    copy_files(test_p, "test")
    print(f"\n✨ 성공! Train: {len(train_p)} / Test: {len(test_p)}")

if __name__ == "__main__":
    main()