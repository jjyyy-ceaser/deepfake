import os
from huggingface_hub import HfApi, hf_hub_download

# 설정
REPO_ID = "luchaoqi/TalkingHeadBench"
SAVE_DIR = r"C:\Users\leejy\Desktop\test_experiment\dataset\TalkingHeadBench"
TARGET_MODEL = "LivePortrait"  # 원하는 모델명 (LivePortrait, Hallo 등)
TARGET_COUNT = 300

api = HfApi()

# 1. 저장소의 모든 파일 목록 가져오기
print("저장소 파일 목록을 불러오는 중...")
all_files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")

# 2. Real과 Fake 파일 분리
real_files = [f for f in all_files if f.startswith("real/") and f.endswith(".mp4")]
fake_files = [f for f in all_files if f.startswith(f"fake/{TARGET_MODEL}/") and f.endswith(".mp4")]

# 3. 파일 이름(ID)을 기준으로 페어 매칭
# 예: real/001.mp4 와 fake/LivePortrait/001.mp4 매칭
pairs = []
for fake_path in fake_files:
    file_name = os.path.basename(fake_path)
    expected_real = f"real/{file_name}"
    
    if expected_real in real_files:
        pairs.append((expected_real, fake_path))
    
    if len(pairs) >= TARGET_COUNT:
        break

print(f"총 {len(pairs)}개의 페어를 찾았습니다. 다운로드를 시작합니다.")

# 4. 다운로드 실행
for i, (real_path, fake_path) in enumerate(pairs):
    print(f"[{i+1}/{len(pairs)}] 다운로드 중: {os.path.basename(real_path)}")
    
    # Real 다운로드
    hf_hub_download(repo_id=REPO_ID, filename=real_path, repo_type="dataset", local_dir=SAVE_DIR)
    # Fake 다운로드
    hf_hub_download(repo_id=REPO_ID, filename=fake_path, repo_type="dataset", local_dir=SAVE_DIR)

print(f"\n모든 다운로드가 완료되었습니다! 저장 위치: {SAVE_DIR}")