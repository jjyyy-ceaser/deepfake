import os
import shutil
import random
from collections import defaultdict
from pathlib import Path

# ======================================================
# [설정] 경로 수정 (여기가 핵심입니다!)
# ======================================================
# 1. 스크린샷에 있는 '정상 데이터' 경로로 변경
REAL_DIR = Path(r"D:\data\dataset\real")
FAKE_DIR = Path(r"D:\data\dataset\fake")

# 2. 최종 저장될 C드라이브 경로
BASE_OUTPUT = Path(r"C:\Users\leejy\Desktop\test_experiment\dataset\final_dataset\raw")

random.seed(42) # 실험 재현성 고정
# ======================================================

def create_folders():
    # 기존 폴더가 있다면 안전하게 삭제 후 재생성 (찌꺼기 제거)
    if BASE_OUTPUT.exists():
        try:
            shutil.rmtree(BASE_OUTPUT)
            print(f"🧹 기존 폴더 정리 완료: {BASE_OUTPUT}")
        except:
            pass

    paths = [
        BASE_OUTPUT / "train" / "real",
        BASE_OUTPUT / "train" / "fake",
        BASE_OUTPUT / "test" / "real",
        BASE_OUTPUT / "test" / "fake"
    ]
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)
    print("📂 폴더 구조 생성 완료")

def main():
    create_folders()

    # 1. Fake 폴더 파일 리스트업
    fake_files = [f for f in os.listdir(FAKE_DIR) if f.endswith(".mp4")]
    
    # 2. ID별 그룹화 (단순 매칭 로직 유지)
    id_map = defaultdict(list)
    for f in fake_files:
        try:
            # 유튜브 ID 추출: 01160--GLiEItdhO5A_1--Ani... -> GLiEItdhO5A
            yt_id = f.split("--")[1].rsplit("_", 1)[0]
            id_map[yt_id].append(f)
        except:
            continue

    unique_ids = list(id_map.keys())
    random.shuffle(unique_ids)

    # 3. 8:2 비율 분할
    split_idx = int(len(unique_ids) * 0.8)
    train_ids = unique_ids[:split_idx]
    test_ids = unique_ids[split_idx:]

    print(f"📊 총 인물(ID) 수: {len(unique_ids)}개")
    print(f"📈 Train ID: {len(train_ids)}개 | Test ID: {len(test_ids)}개")
    print("-" * 50)

    def process_split(target_ids, split_name):
        count = 0
        missing = 0
        for yt_id in target_ids:
            for fake_name in id_map[yt_id]:
                # 📌 [핵심 로직] 재영님의 코드 그대로 사용 (단순 접두어 추가)
                real_name = f"real_{fake_name}"
                
                src_fake = FAKE_DIR / fake_name
                src_real = REAL_DIR / real_name
                
                # 파일 존재 확인 (D드라이브 dataset\real에 있는지)
                if src_fake.exists() and src_real.exists():
                    shutil.copy2(src_fake, BASE_OUTPUT / split_name / "fake" / fake_name)
                    shutil.copy2(src_real, BASE_OUTPUT / split_name / "real" / real_name)
                    count += 1
                else:
                    # 혹시라도 파일이 없으면 로그 출력
                    # print(f"⚠️ 파일 없음: {real_name}")
                    missing += 1
        return count, missing

    print("📦 파일 복사 시작...")
    train_success, train_miss = process_split(train_ids, "train")
    test_success, test_miss = process_split(test_ids, "test")

    print("=" * 50)
    print(f"✅ [Train] 성공: {train_success}쌍 (실패: {train_miss})")
    print(f"✅ [Test]  성공: {test_success}쌍 (실패: {test_miss})")
    print(f"👉 총 합계: {train_success + test_success}쌍")
    print(f"🚀 저장 경로: {BASE_OUTPUT}")

if __name__ == "__main__":
    main()