import os
import shutil
from tqdm import tqdm

# ==========================================
# ⚙️ 경로 설정
# ==========================================
SRC_DIR = r"D:\data\35666"
DST_REAL_DIR = r"C:\Users\leejy\Desktop\test_experiment\dataset\real"
DST_FAKE_DIR = r"C:\Users\leejy\Desktop\test_experiment\dataset\fake"

# 목표 개수 (아까 300개라고 하셨으니 300으로 설정)
TARGET_COUNT = 300

def organize_dataset():
    print(f"🚀 데이터셋 정제 시작...")
    
    # 폴더 생성
    os.makedirs(DST_REAL_DIR, exist_ok=True)
    os.makedirs(DST_FAKE_DIR, exist_ok=True)

    # 모든 mp4 파일 목록 가져오기
    all_files = [f for f in os.listdir(SRC_DIR) if f.lower().endswith('.mp4')]
    
    used_ids = set()
    count = 0

    print(f"🔍 중복 ID 제거 및 복사 중 (목표: {TARGET_COUNT}개)...")
    
    for filename in tqdm(all_files):
        # 💡 [중요] ID 추출 로직
        # CelebV-HQ 파일명이 'ID_VideoName.mp4' 형태라면 첫 번째 언더바(_) 기준 앞부분이 ID입니다.
        # 만약 파일명 형식이 다르다면 이 부분을 수정해야 합니다.
        identity_id = filename.split('_')[0]

        # 이미 뽑은 인물이 아니라면 복사 진행
        if identity_id not in used_ids:
            src_path = os.path.join(SRC_DIR, filename)
            
            # 000.mp4, 001.mp4... 형식으로 저장 (3자리 숫자)
            new_name = f"{count:03d}.mp4"
            dst_path = os.path.join(DST_REAL_DIR, new_name)

            shutil.copy2(src_path, dst_path)
            
            used_ids.add(identity_id)
            count += 1

        # 목표치 도달 시 종료
        if count >= TARGET_COUNT:
            break

    print(f"\n✅ 작업 완료!")
    print(f"📊 저장된 Real 영상 수: {count}개")
    print(f"📂 저장 경로: {DST_REAL_DIR}")
    print(f"📁 Fake 저장 폴더 준비됨: {DST_FAKE_DIR}")

if __name__ == "__main__":
    organize_dataset()