import os
import shutil
import random
from collections import defaultdict

# ======================================================
# [설정] 경로를 최종 확인해라
# ======================================================
REAL_DIR = r"D:\data\processed_real"
FAKE_DIR = r"D:\data\dataset\fake"
# 사용자가 요청한 'raw' 서브폴더를 포함한 경로
BASE_OUTPUT = r"D:\data\final_dataset\raw"

random.seed(42) # 실험 재현성 고정
# ======================================================

def create_folders():
    paths = [
        os.path.join(BASE_OUTPUT, "train", "real"),
        os.path.join(BASE_OUTPUT, "train", "fake"),
        os.path.join(BASE_OUTPUT, "test", "real"),
        os.path.join(BASE_OUTPUT, "test", "fake")
    ]
    for p in paths:
        if not os.path.exists(p):
            os.makedirs(p)
            print(f"📂 폴더 생성: {p}")
    return paths

def main():
    create_folders()

    # 1. Fake 폴더의 파일들을 읽어 ID별로 묶는다.
    # 파일명 예: 01160--GLiEItdhO5A_1--AniPortraitAudio.mp4
    fake_files = [f for f in os.listdir(FAKE_DIR) if f.endswith(".mp4")]
    
    id_map = defaultdict(list)
    for f in fake_files:
        try:
            # 유튜브 ID 추출 (GLiEItdhO5A 부분)
            # -- 뒤의 첫 번째 덩어리에서 _ 앞부분을 가져온다.
            yt_id = f.split("--")[1].rsplit("_", 1)[0]
            id_map[yt_id].append(f)
        except:
            continue

    unique_ids = list(id_map.keys())
    random.shuffle(unique_ids)

    # 2. 8:2 비율 분할
    split_idx = int(len(unique_ids) * 0.8)
    train_ids = unique_ids[:split_idx]
    test_ids = unique_ids[split_idx:]

    print(f"📊 총 인물(ID) 수: {len(unique_ids)}개")
    print(f"📈 Train ID: {len(train_ids)}개 | Test ID: {len(test_ids)}개")
    print("-" * 50)

    def process_split(target_ids, split_name):
        count = 0
        for yt_id in target_ids:
            for fake_name in id_map[yt_id]:
                # Real 파일명 규칙 반영: real_ + fake_name
                real_name = f"real_{fake_name}"
                
                # 확장자 처리 (이전 단계에서 .mp4가 누락되었거나 중복되었을 경우 대비)
                if not real_name.lower().endswith('.mp4'):
                    real_name += ".mp4"
                
                src_fake = os.path.join(FAKE_DIR, fake_name)
                src_real = os.path.join(REAL_DIR, real_name)
                
                # 가끔 전처리 중 실패한 파일이 있을 수 있으므로 존재 확인 후 복사
                if os.path.exists(src_fake) and os.path.exists(src_real):
                    shutil.copy(src_fake, os.path.join(BASE_OUTPUT, split_name, "fake", fake_name))
                    shutil.copy(src_real, os.path.join(BASE_OUTPUT, split_name, "real", real_name))
                    count += 1
        return count

    train_total = process_split(train_ids, "train")
    test_total = process_split(test_ids, "test")

    print(f"✅ [Train] {train_total}쌍 복사 완료")
    print(f"✅ [Test] {test_total}쌍 복사 완료")
    print(f"🚀 모든 데이터가 {BASE_OUTPUT}에 정리되었습니다.")

if __name__ == "__main__":
    main()