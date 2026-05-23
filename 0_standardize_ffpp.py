import cv2
import torch
import os
import csv
import numpy as np
from pathlib import Path
from PIL import Image
from facenet_pytorch import MTCNN

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


# ============================================================
# FF++ Face2Face c23 external dataset standardization
#
# 목적:
#   FF++ Face2Face 외부 데이터셋을 기존 SVD 실험과 동일한 입력 조건으로 표준화
#
# 입력:
#   final_dataset_v2/ffpp_source/real
#   final_dataset_v2/ffpp_source/fake
#
# 출력:
#   final_dataset_v2/raw_ffpp/real
#   final_dataset_v2/raw_ffpp/fake
#
# 표준화 기준:
#   - 25 frames
#   - 25 fps
#   - 1024 x 576
#   - 첫 프레임 기준 MTCNN face-centered smart crop
# ============================================================

BASE_DIR = Path(r"C:\Users\leejy\Desktop\test_experiment\dataset\final_dataset_v2")

SOURCE_ROOT = BASE_DIR / "ffpp_source"
OUTPUT_ROOT = BASE_DIR / "raw_ffpp"

SOURCE_REAL_DIR = SOURCE_ROOT / "real"
SOURCE_FAKE_DIR = SOURCE_ROOT / "fake"

OUTPUT_REAL_DIR = OUTPUT_ROOT / "real"
OUTPUT_FAKE_DIR = OUTPUT_ROOT / "fake"

LOG_PATH = BASE_DIR / "ffpp_standardization_log.csv"

TARGET_W, TARGET_H = 1024, 576
TARGET_FPS = 25
REQUIRED_FRAMES = 25

# 기존 파일이 있어도 다시 만들지 여부
# raw_ffpp 안에 이전에 잘못 들어간 비표준화 파일이 있으면, 실행 전에 raw_ffpp를 비우는 것을 권장
OVERWRITE = True

# 첫 프레임 crop preview 저장 여부
SAVE_PREVIEW_FRAMES = False
PREVIEW_DIR = BASE_DIR / "ffpp_standardization_preview"


def get_target_id_from_filename(path: Path, label: str) -> str:
    """
    real:
      008.mp4 -> 008

    fake:
      008_990.mp4 -> 008
      ffpp_008.mp4 -> 008
    """
    stem = path.stem

    if label == "fake":
        if stem.startswith("ffpp_"):
            return stem.replace("ffpp_", "", 1)

        if "_" in stem:
            return stem.split("_")[0]

    return stem


def validate_source_structure():
    if not SOURCE_REAL_DIR.exists():
        raise FileNotFoundError(f"real 입력 폴더가 없습니다: {SOURCE_REAL_DIR}")

    if not SOURCE_FAKE_DIR.exists():
        raise FileNotFoundError(f"fake 입력 폴더가 없습니다: {SOURCE_FAKE_DIR}")

    real_files = sorted(SOURCE_REAL_DIR.glob("*.mp4"))
    fake_files = sorted(SOURCE_FAKE_DIR.glob("*.mp4"))

    print("============================================================")
    print("[FF++ 표준화 입력 확인]")
    print(f"SOURCE_REAL_DIR: {SOURCE_REAL_DIR}")
    print(f"SOURCE_FAKE_DIR: {SOURCE_FAKE_DIR}")
    print(f"real 개수: {len(real_files)}")
    print(f"fake 개수: {len(fake_files)}")
    print("============================================================")

    if len(real_files) == 0:
        raise RuntimeError("real 입력 mp4 파일이 없습니다.")

    if len(fake_files) == 0:
        raise RuntimeError("fake 입력 mp4 파일이 없습니다.")

    real_ids = {get_target_id_from_filename(p, "real") for p in real_files}
    fake_target_ids = [get_target_id_from_filename(p, "fake") for p in fake_files]

    matched = [tid for tid in fake_target_ids if tid in real_ids]
    unmatched = [tid for tid in fake_target_ids if tid not in real_ids]

    print(f"fake target_id 기준 real 매칭 수: {len(matched)} / {len(fake_files)}")

    if unmatched:
        print("[WARN] real과 매칭되지 않는 fake target_id 일부:")
        for tid in unmatched[:20]:
            print(f"  - {tid}")

    if len(matched) == 0:
        raise RuntimeError("fake target_id와 real 파일명이 전혀 매칭되지 않습니다. 파일명 구조를 확인하세요.")

    return real_files, fake_files


def get_smart_crop_coordinates(frame, detector, scale=2.5):
    """
    기존 0_standardize.py의 Smart Fit Crop 로직 유지.
    - 첫 프레임에서 얼굴 탐지
    - 얼굴 중심 crop
    - 16:9 비율 유지
    - 화면을 벗어나면 crop box를 화면 안쪽으로 보정
    - 얼굴 탐지 실패 시 중앙 crop
    """
    h, w, _ = frame.shape
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    boxes, _ = detector.detect(img_pil)
    target_aspect = TARGET_W / TARGET_H

    face_detected = boxes is not None and len(boxes) > 0

    if not face_detected:
        crop_h = int(h * 0.8)
        crop_w = int(crop_h * target_aspect)

        if crop_w > w:
            crop_w = w
            crop_h = int(crop_w / target_aspect)

        center_x, center_y = w // 2, h // 2

    else:
        box = boxes[0]
        face_h = box[3] - box[1]
        face_cx = (box[0] + box[2]) / 2
        face_cy = (box[1] + box[3]) / 2

        crop_h = int(face_h * scale)
        crop_w = int(crop_h * target_aspect)

        center_x, center_y = int(face_cx), int(face_cy)

    if crop_w > w:
        crop_w = w
        crop_h = int(crop_w / target_aspect)

    if crop_h > h:
        crop_h = h
        crop_w = int(crop_h * target_aspect)

    x1 = int(center_x - (crop_w / 2))
    y1 = int(center_y - (crop_h / 2))
    x2 = x1 + crop_w
    y2 = y1 + crop_h

    if x1 < 0:
        x2 += abs(x1)
        x1 = 0

    if y1 < 0:
        y2 += abs(y1)
        y1 = 0

    if x2 > w:
        x1 -= (x2 - w)
        x2 = w

    if y2 > h:
        y1 -= (y2 - h)
        y2 = h

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    return x1, y1, x2, y2, face_detected


def is_valid_video(path: Path) -> bool:
    if not path.exists():
        return False

    if path.stat().st_size <= 0:
        return False

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return False

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, _ = cap.read()
    cap.release()

    return frame_count > 0 and ret


def read_first_n_frames(video_path: Path, required_frames: int):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        cap.release()
        raise RuntimeError("cv2.VideoCapture open failed")

    input_fps = cap.get(cv2.CAP_PROP_FPS)
    input_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    input_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    input_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    raw_frames = []

    while len(raw_frames) < required_frames:
        ret, frame = cap.read()
        if not ret:
            break
        raw_frames.append(frame)

    cap.release()

    return raw_frames, {
        "input_fps": input_fps,
        "input_frame_count": input_frame_count,
        "input_w": input_w,
        "input_h": input_h,
    }


def pad_frames_boomerang(frames, required_frames: int):
    if len(frames) == 0:
        return frames

    if len(frames) >= required_frames:
        return frames[:required_frames]

    pad_source = frames[-2::-1] if len(frames) > 1 else frames

    while len(frames) < required_frames:
        needed = required_frames - len(frames)
        frames.extend(pad_source[:needed])

    return frames[:required_frames]


def standardize_video(video_path: Path, out_path: Path, detector, label: str):
    result = {
        "label": label,
        "src": str(video_path),
        "dst": str(out_path),
        "target_id": get_target_id_from_filename(video_path, label),
        "status": "",
        "message": "",
        "input_fps": "",
        "input_frame_count": "",
        "input_w": "",
        "input_h": "",
        "output_fps": TARGET_FPS,
        "output_frames": REQUIRED_FRAMES,
        "output_w": TARGET_W,
        "output_h": TARGET_H,
        "x1": "",
        "y1": "",
        "x2": "",
        "y2": "",
        "face_detected": "",
        "padded": "",
    }

    if out_path.exists() and not OVERWRITE and is_valid_video(out_path):
        result["status"] = "skip_existing_valid"
        return result

    if out_path.exists() and OVERWRITE:
        try:
            out_path.unlink()
        except Exception:
            pass

    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        raw_frames, meta = read_first_n_frames(video_path, REQUIRED_FRAMES)

        result["input_fps"] = meta["input_fps"]
        result["input_frame_count"] = meta["input_frame_count"]
        result["input_w"] = meta["input_w"]
        result["input_h"] = meta["input_h"]

        if len(raw_frames) == 0:
            result["status"] = "failed_no_frame"
            result["message"] = "no readable frame"
            return result

        original_read_frames = len(raw_frames)
        raw_frames = pad_frames_boomerang(raw_frames, REQUIRED_FRAMES)

        result["padded"] = original_read_frames < REQUIRED_FRAMES

        x1, y1, x2, y2, face_detected = get_smart_crop_coordinates(raw_frames[0], detector)

        result["x1"] = x1
        result["y1"] = y1
        result["x2"] = x2
        result["y2"] = y2
        result["face_detected"] = face_detected

        if (x2 - x1) < 10 or (y2 - y1) < 10:
            result["status"] = "failed_small_crop"
            result["message"] = "crop area too small"
            return result

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, TARGET_FPS, (TARGET_W, TARGET_H))

        if not writer.isOpened():
            result["status"] = "failed_writer_open"
            result["message"] = "cv2.VideoWriter open failed"
            return result

        preview_frame = None

        for idx, frame in enumerate(raw_frames):
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                final = np.zeros((TARGET_H, TARGET_W, 3), dtype=np.uint8)
            else:
                final = cv2.resize(crop, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LANCZOS4)

            if idx == 0:
                preview_frame = final.copy()

            writer.write(final)

        writer.release()

        if SAVE_PREVIEW_FRAMES and preview_frame is not None:
            preview_label_dir = PREVIEW_DIR / label
            preview_label_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(preview_label_dir / f"{video_path.stem}.jpg"), preview_frame)

        if not is_valid_video(out_path):
            result["status"] = "failed_invalid_output"
            result["message"] = "output video validation failed"
            return result

        result["status"] = "done"
        return result

    except Exception as e:
        result["status"] = "system_error"
        result["message"] = str(e)
        return result


def save_log(results):
    fieldnames = [
        "label",
        "src",
        "dst",
        "target_id",
        "status",
        "message",
        "input_fps",
        "input_frame_count",
        "input_w",
        "input_h",
        "output_fps",
        "output_frames",
        "output_w",
        "output_h",
        "x1",
        "y1",
        "x2",
        "y2",
        "face_detected",
        "padded",
    ]

    with open(LOG_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"\n로그 저장 완료: {LOG_PATH}")


def print_summary(results):
    print("\n============================================================")
    print("[표준화 결과 요약]")
    print("============================================================")

    total = len(results)
    done = sum(1 for r in results if r["status"] == "done")
    skip = sum(1 for r in results if r["status"] == "skip_existing_valid")
    fail = total - done - skip

    print(f"전체: {total}")
    print(f"완료: {done}")
    print(f"스킵: {skip}")
    print(f"실패: {fail}")

    print("\n[출력 파일 개수]")
    print(f"raw_ffpp/real: {len(list(OUTPUT_REAL_DIR.glob('*.mp4'))) if OUTPUT_REAL_DIR.exists() else 0}")
    print(f"raw_ffpp/fake: {len(list(OUTPUT_FAKE_DIR.glob('*.mp4'))) if OUTPUT_FAKE_DIR.exists() else 0}")

    if fail > 0:
        print("\n[실패 파일 일부]")
        for row in results:
            if row["status"] not in ["done", "skip_existing_valid"]:
                print(f"{row['status']} | {row['src']}")
                if row["message"]:
                    print(f"  message: {row['message'][:300]}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("============================================================")
    print("FF++ Face2Face 외부 데이터셋 표준화 시작")
    print(f"Device     : {device}")
    print(f"SOURCE_ROOT: {SOURCE_ROOT}")
    print(f"OUTPUT_ROOT: {OUTPUT_ROOT}")
    print(f"Target     : {TARGET_W}x{TARGET_H}, {TARGET_FPS}fps, {REQUIRED_FRAMES}frames")
    print("============================================================")

    real_files, fake_files = validate_source_structure()

    OUTPUT_REAL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FAKE_DIR.mkdir(parents=True, exist_ok=True)

    detector = MTCNN(select_largest=True, post_process=False, device=device)

    jobs = []

    for p in real_files:
        jobs.append(("real", p, OUTPUT_REAL_DIR / p.name))

    for p in fake_files:
        jobs.append(("fake", p, OUTPUT_FAKE_DIR / p.name))

    results = []

    for label, src_path, out_path in tqdm(jobs, total=len(jobs), unit="vid"):
        result = standardize_video(src_path, out_path, detector, label)
        results.append(result)

    save_log(results)
    print_summary(results)

    print("\n다음 단계:")
    print("1. raw_ffpp/real, raw_ffpp/fake 개수가 입력과 같은지 확인")
    print("2. ffpp_standardization_log.csv에서 failed 행이 없는지 확인")
    print("3. 이상 없으면 generate_distorted_ffpp_dataset.py 실행")


if __name__ == "__main__":
    main()