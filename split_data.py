import os
import shutil
import random
from tqdm import tqdm

# ==========================================
# ⚙️ 설정 (경로 및 비율)
# ==========================================
BASE_DIR = r"C:\Users\leejy\Desktop\test_experiment\dataset"
SRC_REAL_DIR = os.path.join(BASE_DIR, "real")
SRC_FAKE_DIR = os.path.join(BASE_DIR, "fake")

# 결과가 저장될 폴더 (디버그 코드에서 쓴 경로와 일치시킴)
DST_BASE_DIR = os.path.join(BASE_DIR, "final_dataset_v2")

# 비율 설정 (Train : Test = 8 : 2)
TRAIN_RATIO = 0.8 

def split_dataset():
    print(f"🚀 데이터셋 분할 시작 (Train:Test = {int(TRAIN_RATIO*10)}:{int((1-TRAIN_RATIO)*10)})")
    
    # 1. 파일 목록 확인
    if not os.path.exists(SRC_REAL_DIR) or not os.path.exists(SRC_FAKE_DIR):
        print("❌ 오류: 원본 데이터 폴더(real/fake)를 찾을 수 없습니다.")
        return

    # Real 파일 기준 ID 추출 (확장자 제외한 이름)
    # 예: "001.mp4" -> "001"
    real_files = [f for f in os.listdir(SRC_REAL_DIR) if f.endswith('.mp4')]
    all_ids = [os.path.splitext(f)[0] for f in real_files]
    
    total_count = len(all_ids)
    print(f"🔍 총 데이터 개수: {total_count}쌍 (Real + Fake)")

    # 2. 랜덤 셔플 (재현성을 위해 시드 고정)
    random.seed(42)
    random.shuffle(all_ids)

    # 3. 분할 지점 계산
    split_idx = int(total_count * TRAIN_RATIO)
    train_ids = all_ids[:split_idx]
    test_ids = all_ids[split_idx:]

    print(f"📊 분할 결과 -> Train: {len(train_ids)}개, Test: {len(test_ids)}개")

    # 4. 파일 이동 (복사) 함수
    def copy_files(ids, split_name):
        # 목표 폴더 생성 (예: final_dataset_v2/train/real)
        dst_real = os.path.join(DST_BASE_DIR, split_name, "real")
        dst_fake = os.path.join(DST_BASE_DIR, split_name, "fake")
        os.makedirs(dst_real, exist_ok=True)
        os.makedirs(dst_fake, exist_ok=True)

        print(f"🚚 [{split_name.upper()}] 파일 복사 중...")
        
        for file_id in tqdm(ids):
            # (1) Real 파일 복사
            src_r = os.path.join(SRC_REAL_DIR, f"{file_id}.mp4")
            dst_r = os.path.join(dst_real, f"{file_id}.mp4")
            
            # (2) Fake 파일 복사 (파일명 규칙: svd_{ID}.mp4)
            src_f = os.path.join(SRC_FAKE_DIR, f"svd_{file_id}.mp4")
            dst_f = os.path.join(dst_fake, f"svd_{file_id}.mp4")

            # 파일이 둘 다 있을 때만 복사 (하나라도 없으면 건너뜀)
            if os.path.exists(src_r) and os.path.exists(src_f):
                shutil.copy2(src_r, dst_r)
                shutil.copy2(src_f, dst_f)
            else:
                print(f"⚠️ 경고: 짝이 맞지 않는 파일 발견 -> ID: {file_id}")

    # 5. 실제 실행
    copy_files(train_ids, "train")
    copy_files(test_ids, "test")

    print(f"\n🎉 모든 작업 완료!")
    print(f"📂 저장 위치: {DST_BASE_DIR}")
    print(f"   ├─ train/ (Real, Fake)")
    print(f"   └─ test/  (Real, Fake)")

if __name__ == "__main__":
    split_dataset()