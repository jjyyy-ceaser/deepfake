import cv2
import os
import glob

# 테스트할 폴더 경로 (v2 경로)
TEST_DIR = r"C:\Users\leejy\Desktop\test_experiment\dataset\final_dataset_v2\train\real"

def check_video():
    print(f"📂 폴더 확인 중: {TEST_DIR}")
    files = glob.glob(os.path.join(TEST_DIR, "*.mp4"))
    
    if not files:
        print("❌ 폴더에 mp4 파일이 없습니다! 경로를 확인하세요.")
        return

    # 첫 번째 파일만 테스트
    target_file = files[0]
    print(f"▶️ 테스트 파일: {target_file}")
    print(f"   (파일 크기: {os.path.getsize(target_file) / 1024:.2f} KB)")

    cap = cv2.VideoCapture(target_file)
    
    if not cap.isOpened():
        print("❌ [치명적 오류] OpenCV가 파일을 열지 못했습니다.")
        print("   -> 원인 1: 경로에 한글이 있나요? (OpenCV는 한글 경로 싫어함)")
        print("   -> 원인 2: ffmpeg 코덱 문제일 수 있습니다.")
    else:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"✅ 파일 열기 성공!")
        print(f"   - 총 프레임: {total}")
        print(f"   - 해상도: {width} x {height}")
        
        ret, frame = cap.read()
        if ret:
            print("✅ 첫 프레임 읽기 성공! (색상 정보 있음)")
        else:
            print("❌ 파일은 열렸는데 프레임을 못 읽습니다. (코덱 문제 유력)")
            
    cap.release()

if __name__ == "__main__":
    check_video()