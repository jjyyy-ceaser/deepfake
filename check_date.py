import os
import glob

# 선생님의 현재 코드 설정과 똑같이 맞춤
# 경로를 더 깊게 설정합니다. (윈도우에서는 역슬래시 \\ 대신 슬래시 / 써도 잘 됩니다)
DATA_DIR = "dataset/0_main_train"
CURRENT_DIR = os.getcwd()

print(f"📍 현재 작업 위치: {CURRENT_DIR}")
print(f"🔎 찾는 폴더 위치: {os.path.join(CURRENT_DIR, DATA_DIR)}")

# 실제 파일 찾기
real_path = os.path.join(DATA_DIR, "real", "*.mp4")
fake_path = os.path.join(DATA_DIR, "fake", "*.mp4")

print(f"\n📡 검색 패턴 (Real): {real_path}")
real_files = glob.glob(real_path)
print(f"   👉 찾은 개수: {len(real_files)}개")

print(f"\n📡 검색 패턴 (Fake): {fake_path}")
fake_files = glob.glob(fake_path)
print(f"   👉 찾은 개수: {len(fake_files)}개")

if len(real_files) == 0 and len(fake_files) == 0:
    print("\n🚨 [결과] 파일이 없습니다! dataset 폴더가 비었거나, 폴더명이 바뀌었는지 확인하세요.")
else:
    print("\n✅ [결과] 파일이 있습니다. 코드가 왜 멈췄는지 다시 확인해봐야 합니다.")