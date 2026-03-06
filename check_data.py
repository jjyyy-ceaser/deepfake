import cv2
import glob
from pathlib import Path

# 경로 확인 필요
TEST_DIR = Path(r"C:\Users\leejy\Desktop\test_experiment\dataset\final_dataset\raw\test")

print("🔍 데이터 진단 시작...")

for label in ["real", "fake"]:
    folder = TEST_DIR / label
    files = list(folder.glob("*.mp4"))
    print(f"\n📂 [{label.upper()}] 파일 개수: {len(files)}개")
    
    fail_count = 0
    for p in files[:5]: # 5개만 샘플 테스트
        cap = cv2.VideoCapture(str(p))
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print(f"  ❌ 읽기 실패 (Black Frame): {p.name}")
            fail_count += 1
        else:
            print(f"  ✅ 읽기 성공: {p.name} | 크기: {frame.shape}")

    if fail_count > 0:
        print(f"  ⚠️ 경고: {label} 데이터 일부를 OpenCV가 못 읽고 있습니다!")