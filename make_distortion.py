import os
import subprocess
from pathlib import Path
from tqdm import tqdm

# ======================================================
# [설정] 경로 (시스템 호환성 강화)
# ======================================================
BASE_DIR = Path(r"C:\Users\leejy\Desktop\test_experiment\dataset\final_dataset").resolve()
RAW_BASE = BASE_DIR / "raw"

# [Forensic Profile Analysis]
# kakao_high의 경우 hevc 인코더가 없다면 libx264로 자동 전환하도록 예외처리 추가 가능
DISTORTION_PROFILES = {
    "youtube": {
        "res": "640:360", "codec": "libx264", "crf": "45",
        "movflags": "+faststart", "tag": None
    },
    "instagram": {
        "res": "1276:720", "codec": "libx264", "crf": "35",
        "movflags": "+faststart", "tag": None
    },
    "kakao_high": {
        "res": "1920:1080", "codec": "libx265", "crf": "22",
        "movflags": None, "tag": "hvc1"
    },
    "kakao_normal": {
        "res": "1280:720", "codec": "libx264", "crf": "24",
        "movflags": "+faststart", "tag": None
    },
}
# ======================================================

def apply_distortion(input_path, output_path, profile):
    """플랫폼별 물리적/구조적 왜곡 적용"""
    
    # [핵심 수정] 윈도우 경로 역슬래시 이슈 방지를 위해 문자열 변환 후 replace
    # 하지만 subprocess에서는 그냥 str()로 넘기는 게 정석입니다.
    # 혹시 모를 한글 경로 등을 대비해 절대 경로로 변환합니다.
    input_str = str(input_path.resolve())
    output_str = str(output_path.resolve())

    cmd = [
        "ffmpeg", "-y",  # 덮어쓰기 강제
        "-i", input_str,
        "-vf", f"scale={profile['res']}",
        "-c:v", profile['codec'],
        "-crf", profile['crf'],
        "-c:a", "aac",
        "-b:a", "128k",
        "-pix_fmt", "yuv420p",
        "-loglevel", "error"
    ]

    if profile["movflags"]:
        cmd.extend(["-movflags", profile["movflags"]])

    if profile["tag"]:
        cmd.extend(["-tag:v", profile["tag"]])

    cmd.append(output_str)
    
    # 실행 (인코딩 에러 발생 시 즉시 확인 가능하도록)
    subprocess.run(cmd, check=True, capture_output=True, text=True)

def main():
    if not RAW_BASE.exists():
        print(f"❌ 에러: 원본 폴더를 찾을 수 없습니다: {RAW_BASE}")
        return

    splits = ["train", "test"]
    labels = ["real", "fake"]

    print(f"🚀 [Fix] 왜곡 데이터셋 생성 시작 (경로 호환성 패치)")
    print(f"📍 원본 위치: {RAW_BASE}")

    for split in splits:
        for label in labels:
            src_dir = RAW_BASE / split / label
            if not src_dir.exists(): continue
            
            videos = list(src_dir.glob("*.mp4"))
            if not videos: continue

            print(f"\n📂 처리 중: {split}/{label} ({len(videos)}개)")

            for v_path in tqdm(videos, desc="플랫폼 변환"):
                for platform, profile in DISTORTION_PROFILES.items():
                    # 폴더 생성
                    dst_dir = BASE_DIR / platform / split / label
                    dst_dir.mkdir(parents=True, exist_ok=True)
                    
                    out_path = dst_dir / v_path.name
                    
                    # 0KB 파일이거나 파일이 없으면 생성 시도
                    if not out_path.exists() or out_path.stat().st_size == 0:
                        try:
                            apply_distortion(v_path, out_path, profile)
                        except subprocess.CalledProcessError as e:
                            # H.265 인코더가 없을 경우에 대한 예외 처리 메시지
                            if "Unknown encoder 'libx265'" in e.stderr:
                                print(f"\n⚠️ H.265(libx265) 코덱이 없습니다. ffmpeg 버전을 확인하세요.")
                                return # 더 이상 진행 불가
                            
                            print(f"\n❌ 실패 ({platform}): {v_path.name}")
                            print(f"   [FFmpeg Error] {e.stderr.strip()}")
                        except Exception as e:
                            print(f"\n⚠️ 기타 오류: {e}")

    print("\n" + "="*50)
    print(f"🎉 데이터셋 구축 완료. {BASE_DIR}")

if __name__ == "__main__":
    main()