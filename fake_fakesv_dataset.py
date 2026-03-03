import json
import os
import shutil
from tqdm import tqdm

# ==========================================
# 📂 경로 설정 (본인 환경에 맞게 수정하세요)
# ==========================================
# 1. 다운로드받은 FakeSV 영상들이 있는 폴더
SOURCE_VIDEO_DIR = r"D:\data\videos" 
# 2. FakeSV 코드 폴더 내의 data.json 경로
DATA_JSON_PATH = r"D:\data\data.json"
# 3. 우리 논문 프로젝트의 가짜 데이터 저장 폴더
TARGET_DIR = r"C:\Users\leejy\Desktop\test_experiment\dataset\fakeSV"

def collect_fake_videos():
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)

    fake_ids = []
    
    # 1. data.json을 읽어 가짜 영상 ID 리스트업
    with open(DATA_JSON_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            # '假'와 '辟谣' 태그를 모두 가짜로 수집
            if item['annotation'] in ['假', '辟谣']:
                fake_ids.append(item['video_id'])

    print(f"🔍 가짜 영상 {len(fake_ids)}개를 찾았습니다. 복사를 시작합니다...")

    # 2. 해당 ID의 영상을 타겟 폴더로 복사
    count = 0
    for vid in tqdm(fake_ids):
        video_name = f"{vid}.mp4" # 확장자가 .mp4라고 가정
        src = os.path.join(SOURCE_VIDEO_DIR, video_name)
        dst = os.path.join(TARGET_DIR, video_name)

        if os.path.exists(src):
            shutil.copy(src, dst)
            count += 1

    print(f"\n✅ 완료! 총 {count}개의 가짜 영상이 {TARGET_DIR}로 이동되었습니다.")

if __name__ == "__main__":
    collect_fake_videos()