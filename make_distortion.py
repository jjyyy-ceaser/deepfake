import os
import shutil
import subprocess
from pathlib import Path
from tqdm import tqdm

# ==========================================
# ⚙️ 경로 및 플랫폼 왜곡 설정 (Forensic Report 기반)
# ==========================================
BASE_DIR = r"dataset/test"
CASES = ["case1_original", "case4_mixed"]

# 포렌식 리포트(final_forensic_report.csv)의 평균치 반영
DISTORTION_PROFILES = {
    "instagram": {"res": "1276:720", "codec": "libx264", "crf": "34"},
    "kakao_high": {"res": "1920:1080", "codec": "libx265", "crf": "22"}, # HEVC 적용
    "kakao_low": {"res": "1280:720", "codec": "libx264", "crf": "30"},
}

def apply_ffmpeg_distortion(input_path, output_path, profile):
    """FFmpeg를 호출하여 물리적 왜곡 적용"""
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-vf", f"scale={profile['res']}",
        "-vcodec", profile['codec'],
        "-crf", profile['crf'],
        "-pix_fmt", "yuv420p",
        "-loglevel", "error",
        str(output_path)
    ]
    subprocess.run(cmd)

def main():
    for case_name in CASES:
        case_root = Path(BASE_DIR) / case_name
        if not case_root.exists(): continue
        
        print(f"\n🚀 Case 처리 시작: {case_name}")
        
        # [Fake] pika, runway, svd 및 [Real] 폴더 탐색
        categories = ["fake", "real"]
        
        for cat in categories:
            cat_path = case_root / cat
            if not cat_path.exists(): continue
            
            # 하위 생성기 폴더(pika, runway, svd) 혹은 real 폴더 자체
            if cat == "fake":
                target_dirs = [d for d in cat_path.iterdir() if d.is_dir()]
            else:
                target_dirs = [cat_path] # real은 생성기 구분이 없으므로 자기 자신

            for target_dir in target_dirs:
                print(f"  🔹 대상 폴더: {target_dir.relative_to(BASE_DIR)}")
                
                # 1. 원본 영상 확보 (.mp4 기준)
                videos = list(target_dir.glob("*.mp4"))
                if not videos: continue
                
                # 2. 하위 폴더 생성 (original, instagram, kakao_high, kakao_low)
                platforms = ["original"] + list(DISTORTION_PROFILES.keys())
                for plat in platforms:
                    (target_dir / plat).mkdir(parents=True, exist_ok=True)

                # 3. 파일 이동 및 왜곡 생성
                for v_path in tqdm(videos, desc=f"    {target_dir.name} 변환 중", leave=False):
                    # 원본을 original 폴더로 이동
                    original_dest = target_dir / "original" / v_path.name
                    shutil.move(str(v_path), str(original_dest))
                    
                    # 이동된 원본을 소스로 사용하여 플랫폼별 왜곡 생성
                    for plat, profile in DISTORTION_PROFILES.items():
                        out_path = target_dir / plat / v_path.name
                        # 이미 존재하면 스킵 (중단 후 재시작 대비)
                        if not out_path.exists():
                            apply_ffmpeg_distortion(original_dest, out_path, profile)

if __name__ == "__main__":
    main()