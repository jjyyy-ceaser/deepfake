import os
import glob
import csv
import shutil
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# ============================================================
# FF++ Face2Face c23 external dataset distortion generation
#
# 입력:
#   C:\Users\leejy\Desktop\test_experiment\dataset\final_dataset_v2\raw_ffpp\real
#   C:\Users\leejy\Desktop\test_experiment\dataset\final_dataset_v2\raw_ffpp\fake
#
# 출력:
#   instagram_ffpp / youtube_ffpp / kakao_high_ffpp / kakao_normal_ffpp
#
# 목적:
#   기존 SVD 실험의 플랫폼 왜곡 설정을 FF++ 외부 데이터셋에도 동일하게 적용
# ============================================================

BASE_DIR = Path(r"C:\Users\leejy\Desktop\test_experiment\dataset\final_dataset_v2")
SRC_ROOT = BASE_DIR / "raw_ffpp"

FFMPEG_CMD = "ffmpeg"
FFPROBE_CMD = "ffprobe"

# 기존 generate_distorted_dataset.py와 동일한 병렬 처리 기준
# CPU 부담이 크면 4로 낮추면 됨
MAX_WORKERS = 8

LOG_PATH = BASE_DIR / "ffpp_distortion_generation_log.csv"

PLATFORM_CONFIGS = {
    "instagram_ffpp": {
        "scale": "scale=640:640:force_original_aspect_ratio=decrease,pad=640:640:(ow-iw)/2:(oh-ih)/2",
        "bitrate": "2200k",
        "fps": 30,
        "gop": "30",
        "bframes": "0",
        "profile": "main",
        "preset": "medium",
    },
    "youtube_ffpp": {
        "scale": "scale=1920:1080",
        "bitrate": "500k",
        "fps": 30,
        "gop": "60",
        "bframes": "2",
        "profile": "high",
        "preset": "slow",
    },
    "kakao_high_ffpp": {
        "scale": "scale=1920:1080",
        "bitrate": "2700k",
        "fps": 30,
        "gop": "30",
        "bframes": "0",
        "profile": "high",
        "preset": "fast",
    },
    "kakao_normal_ffpp": {
        "scale": "scale=960:540",
        "bitrate": "670k",
        "fps": 24,
        "gop": "30",
        "bframes": "0",
        "profile": "baseline",
        "preset": "veryfast",
    },
}


def parse_bitrate(value_str):
    value_str = str(value_str).strip().lower()

    if value_str.endswith("k"):
        return int(float(value_str[:-1]) * 1000)

    if value_str.endswith("m"):
        return int(float(value_str[:-1]) * 1000000)

    if value_str.isdigit():
        return int(value_str)

    return 2000000


def check_command_exists(command_name):
    return shutil.which(command_name) is not None


def is_valid_video(video_path):
    """
    출력 파일이 이미 있을 때, 단순 존재 여부가 아니라 ffprobe로 읽히는지 확인.
    깨진 파일이면 삭제 후 다시 생성하도록 함.
    """
    video_path = Path(video_path)

    if not video_path.exists():
        return False

    if video_path.stat().st_size <= 0:
        return False

    if not check_command_exists(FFPROBE_CMD):
        # ffprobe가 없으면 최소한 파일 크기만 보고 판단
        return True

    cmd = [
        FFPROBE_CMD,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate",
        "-of", "csv=p=0",
        str(video_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=20,
        )
        return result.returncode == 0 and result.stdout.strip() != ""
    except Exception:
        return False


def get_fake_target_id(fake_path):
    """
    FF++ Face2Face 파일명:
      008_990.mp4 -> target_id = 008
    """
    stem = Path(fake_path).stem

    if "_" in stem:
        return stem.split("_")[0]

    if stem.startswith("ffpp_"):
        return stem.replace("ffpp_", "", 1)

    return stem


def validate_raw_ffpp_structure():
    real_dir = SRC_ROOT / "real"
    fake_dir = SRC_ROOT / "fake"

    if not real_dir.exists():
        raise FileNotFoundError(f"real 입력 폴더가 없습니다: {real_dir}")

    if not fake_dir.exists():
        raise FileNotFoundError(f"fake 입력 폴더가 없습니다: {fake_dir}")

    real_files = sorted(real_dir.glob("*.mp4"))
    fake_files = sorted(fake_dir.glob("*.mp4"))

    print("============================================================")
    print("[입력 데이터 확인]")
    print(f"real_dir : {real_dir}")
    print(f"fake_dir : {fake_dir}")
    print(f"real 개수: {len(real_files)}")
    print(f"fake 개수: {len(fake_files)}")
    print("============================================================")

    if len(real_files) == 0 or len(fake_files) == 0:
        raise RuntimeError("real 또는 fake mp4 파일이 없습니다.")

    real_ids = {p.stem for p in real_files}
    fake_target_ids = [get_fake_target_id(p) for p in fake_files]

    matched = [tid for tid in fake_target_ids if tid in real_ids]
    unmatched = [tid for tid in fake_target_ids if tid not in real_ids]

    print(f"fake target_id 기준 real 매칭 수: {len(matched)} / {len(fake_files)}")

    if unmatched:
        print("[WARN] real과 매칭되지 않는 fake target_id 일부:")
        for tid in unmatched[:20]:
            print(f"  - {tid}")

    if len(matched) == 0:
        raise RuntimeError("fake target_id와 real 파일명이 전혀 매칭되지 않습니다. 파일명을 확인하세요.")

    return real_files, fake_files


def process_video(task):
    platform, label, src_path, dst_path, cfg = task

    src_path = Path(src_path)
    dst_path = Path(dst_path)
    tmp_path = dst_path.with_name(dst_path.stem + ".tmp" + dst_path.suffix)

    if is_valid_video(dst_path):
        return {
            "platform": platform,
            "label": label,
            "src": str(src_path),
            "dst": str(dst_path),
            "status": "skip_existing_valid",
            "message": "",
        }

    if dst_path.exists() and not is_valid_video(dst_path):
        try:
            dst_path.unlink()
        except Exception as e:
            return {
                "platform": platform,
                "label": label,
                "src": str(src_path),
                "dst": str(dst_path),
                "status": "failed_remove_invalid_output",
                "message": str(e),
            }

    if tmp_path.exists():
        try:
            tmp_path.unlink()
        except Exception:
            pass

    os.makedirs(dst_path.parent, exist_ok=True)

    target_bps = parse_bitrate(cfg["bitrate"])
    buf_size = target_bps * 2

    cmd = [
        FFMPEG_CMD,
        "-y",
        "-i", str(src_path),

        "-vf", cfg["scale"],
        "-r", str(cfg["fps"]),

        "-b:v", str(target_bps),
        "-maxrate", str(target_bps),
        "-bufsize", str(buf_size),

        "-c:v", "libx264",
        "-profile:v", cfg["profile"],
        "-preset", cfg["preset"],

        "-g", str(cfg["gop"]),
        "-keyint_min", str(cfg["gop"]),
        "-bf", str(cfg["bframes"]),

        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        "-loglevel", "error",

        str(tmp_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:
            if tmp_path.exists():
                tmp_path.unlink()
            return {
                "platform": platform,
                "label": label,
                "src": str(src_path),
                "dst": str(dst_path),
                "status": "ffmpeg_error",
                "message": result.stderr.strip(),
            }

        if not is_valid_video(tmp_path):
            if tmp_path.exists():
                tmp_path.unlink()
            return {
                "platform": platform,
                "label": label,
                "src": str(src_path),
                "dst": str(dst_path),
                "status": "invalid_output",
                "message": "ffprobe validation failed",
            }

        os.replace(tmp_path, dst_path)

        return {
            "platform": platform,
            "label": label,
            "src": str(src_path),
            "dst": str(dst_path),
            "status": "done",
            "message": "",
        }

    except Exception as e:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass

        return {
            "platform": platform,
            "label": label,
            "src": str(src_path),
            "dst": str(dst_path),
            "status": "system_error",
            "message": str(e),
        }


def build_tasks():
    tasks = []

    for platform, cfg in PLATFORM_CONFIGS.items():
        for label in ["real", "fake"]:
            src_dir = SRC_ROOT / label
            dst_dir = BASE_DIR / platform / label
            os.makedirs(dst_dir, exist_ok=True)

            files = sorted(glob.glob(str(src_dir / "*.mp4")))

            print(f"[작업 목록] {platform} / {label}: {len(files)}개")

            for src_path in files:
                filename = os.path.basename(src_path)
                dst_path = dst_dir / filename
                tasks.append((platform, label, src_path, dst_path, cfg))

    return tasks


def save_log(results):
    fieldnames = ["platform", "label", "src", "dst", "status", "message"]

    with open(LOG_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"\n로그 저장 완료: {LOG_PATH}")


def print_output_counts():
    print("\n============================================================")
    print("[출력 폴더별 mp4 개수 확인]")
    print("============================================================")

    for platform in PLATFORM_CONFIGS.keys():
        for label in ["real", "fake"]:
            folder = BASE_DIR / platform / label
            count = len(list(folder.glob("*.mp4"))) if folder.exists() else 0
            print(f"{platform:18s} / {label:4s}: {count}")


def main():
    print("============================================================")
    print("FF++ Face2Face c23 distortion generation")
    print(f"BASE_DIR   : {BASE_DIR}")
    print(f"SRC_ROOT   : {SRC_ROOT}")
    print(f"MAX_WORKERS: {MAX_WORKERS}")
    print("============================================================")

    if not check_command_exists(FFMPEG_CMD):
        raise RuntimeError("ffmpeg 명령어를 찾을 수 없습니다. ffmpeg 설치 또는 PATH 등록을 확인하세요.")

    if not check_command_exists(FFPROBE_CMD):
        print("[WARN] ffprobe를 찾을 수 없습니다. 출력 검증은 파일 존재/크기 기준으로만 수행됩니다.")

    validate_raw_ffpp_structure()

    tasks = build_tasks()

    if not tasks:
        print("처리할 영상이 없습니다.")
        return

    print("\n============================================================")
    print(f"총 처리 대상: {len(tasks)}개")
    print("왜곡 생성 시작")
    print("============================================================")

    results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for result in tqdm(executor.map(process_video, tasks), total=len(tasks), unit="vid"):
            results.append(result)

    done = sum(1 for r in results if r["status"] == "done")
    skip = sum(1 for r in results if r["status"] == "skip_existing_valid")
    fail = len(results) - done - skip

    print("\n============================================================")
    print("왜곡 생성 완료")
    print(f"생성 완료: {done}")
    print(f"기존 정상 파일 skip: {skip}")
    print(f"실패: {fail}")
    print("============================================================")

    if fail > 0:
        print("\n실패 파일 일부:")
        for r in results:
            if r["status"] not in ["done", "skip_existing_valid"]:
                print(f"{r['status']} | {r['src']} -> {r['dst']}")
                if r["message"]:
                    print(f"  message: {r['message'][:300]}")

    save_log(results)
    print_output_counts()


if __name__ == "__main__":
    main()