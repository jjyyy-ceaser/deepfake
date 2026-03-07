import cv2
import torch
import os
import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image
from tqdm import tqdm
from pathlib import Path

# [경로 설정]
SOURCE_DIR = Path(r"C:\Users\leejy\Desktop\test_experiment\dataset\source_raw")
REAL_ALIGNED_DIR = Path(r"C:\Users\leejy\Desktop\test_experiment\dataset\real")
SVD_INPUT_DIR = Path(r"C:\Users\leejy\Desktop\test_experiment\dataset\svd_input")

# 목표 설정
TARGET_W, TARGET_H = 1024, 576
TARGET_FPS = 25
REQUIRED_FRAMES = 25

def get_smart_crop_coordinates(frame, detector, scale=2.5):
    """
    [핵심 수정] Smart Fit Crop
    - 얼굴 중심을 절대 사수함.
    - 크롭 박스가 화면을 벗어나면, 비율(16:9)을 유지한 채로 박스를 줄여서라도 화면 안에 넣음.
    - 절대 중앙으로 초기화(Fallback)하지 않음.
    """
    h, w, _ = frame.shape
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # 1. 얼굴 탐지
    boxes, _ = detector.detect(img_pil)
    target_aspect = TARGET_W / TARGET_H # 1.77...

    # A. 얼굴 못 찾음 -> 어쩔 수 없이 중앙 크롭
    if boxes is None:
        crop_h = int(h * 0.8)
        crop_w = int(crop_h * target_aspect)
        # 화면보다 크면 화면에 맞춤
        if crop_w > w:
            crop_w = w
            crop_h = int(crop_w / target_aspect)
        center_x, center_y = w // 2, h // 2
        
    # B. 얼굴 찾음 -> Smart Logic 가동
    else:
        box = boxes[0]
        face_h = box[3] - box[1]
        face_cx = (box[0] + box[2]) / 2
        face_cy = (box[1] + box[3]) / 2
        
        # 초기 목표 크기 (얼굴 높이 * scale)
        crop_h = int(face_h * scale)
        crop_w = int(crop_h * target_aspect)
        
        center_x, center_y = int(face_cx), int(face_cy)

    # [Logic 1] 크기 보정 (Fit to Screen Size)
    # 계산된 박스가 원본 화면보다 크면, 비율 유지하며 줄임
    if crop_w > w:
        crop_w = w
        crop_h = int(crop_w / target_aspect)
    if crop_h > h:
        crop_h = h
        crop_w = int(crop_h * target_aspect)

    # [Logic 2] 좌표 계산
    x1 = int(center_x - (crop_w / 2))
    y1 = int(center_y - (crop_h / 2))
    x2 = x1 + crop_w
    y2 = y1 + crop_h

    # [Logic 3] 위치 보정 (Shift inside Screen)
    # 왼쪽/위로 벗어남 -> 오른쪽/아래로 밈
    if x1 < 0: 
        x2 += abs(x1)
        x1 = 0
    if y1 < 0: 
        y2 += abs(y1)
        y1 = 0
        
    # 오른쪽/아래로 벗어남 -> 왼쪽/위로 밈
    if x2 > w: 
        x1 -= (x2 - w)
        x2 = w
    if y2 > h: 
        y1 -= (y2 - h)
        y2 = h
        
    # [Final Check] 미세 오차로 인한 범위 이탈 강제 클램핑
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)
        
    return x1, y1, x2, y2

def process_video(video_path, detector):
    cap = cv2.VideoCapture(str(video_path))
    
    # 1. 프레임 읽기 (최대 25개)
    raw_frames = []
    while len(raw_frames) < REQUIRED_FRAMES:
        ret, frame = cap.read()
        if not ret: break
        raw_frames.append(frame)
    cap.release()

    if not raw_frames: return

    # 2. 패딩 (Boomerang Mode)
    if len(raw_frames) < REQUIRED_FRAMES:
        print(f"⚠️ Padding: {video_path.name} ({len(raw_frames)}f)")
        pad_source = raw_frames[-2::-1] if len(raw_frames) > 1 else raw_frames
        while len(raw_frames) < REQUIRED_FRAMES:
            needed = REQUIRED_FRAMES - len(raw_frames)
            raw_frames.extend(pad_source[:needed])

    # 3. Smart Crop 좌표 계산 (첫 프레임 기준)
    try:
        x1, y1, x2, y2 = get_smart_crop_coordinates(raw_frames[0], detector)
    except Exception as e:
        print(f"Skipping {video_path.name}: Crop calc failed ({e})")
        return

    # 유효성 검사
    if (x2 - x1) < 10 or (y2 - y1) < 10:
        print(f"Skipping {video_path.name}: Crop area too small")
        return

    # 4. 저장
    save_name = video_path.name
    real_out_path = REAL_ALIGNED_DIR / save_name
    out = cv2.VideoWriter(str(real_out_path), cv2.VideoWriter_fourcc(*'mp4v'), TARGET_FPS, (TARGET_W, TARGET_H))
    
    # SVD 입력 이미지 저장
    first_crop = raw_frames[0][y1:y2, x1:x2]
    final_img = cv2.resize(first_crop, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite(str(SVD_INPUT_DIR / f"{video_path.stem}.png"), final_img)
    
    # 비디오 저장
    for frame in raw_frames:
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            final = np.zeros((TARGET_H, TARGET_W, 3), dtype=np.uint8)
        else:
            final = cv2.resize(crop, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LANCZOS4)
        out.write(final)
        
    out.release()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Real 데이터 표준화 (Smart Fit & 25f Fix) 시작 [Device: {device}]")
    
    # MTCNN 설정 (가장 큰 얼굴 1개만)
    detector = MTCNN(select_largest=True, post_process=False, device=device)
    
    REAL_ALIGNED_DIR.mkdir(parents=True, exist_ok=True)
    SVD_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    videos = list(SOURCE_DIR.glob("*.mp4"))
    for v in tqdm(videos):
        process_video(v, detector)

if __name__ == "__main__":
    main()