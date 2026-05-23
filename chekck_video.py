
import cv2
from pathlib import Path

folders = [
    Path(r"D:\data\FF_Data_c23\original_sequences\youtube\c23\videos"),
    Path(r"D:\data\FF_Data_c23\manipulated_sequences\Face2Face\c23\videos"),
]

bad = []

for folder in folders:
    if not folder.exists():
        print(f"폴더 없음: {folder}")
        continue

    for p in folder.glob("*.mp4"):
        cap = cv2.VideoCapture(str(p))
        ok = cap.isOpened()
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if ok else 0
        ret, frame = cap.read() if ok else (False, None)
        cap.release()

        if (not ok) or frames <= 0 or (not ret):
            bad.append(p)

print(f"검사 완료. 문제 파일 수: {len(bad)}")

for p in bad[:50]:
    print(p)