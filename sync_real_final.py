import os
import glob
import numpy as np
import librosa
from scipy import signal
import subprocess
import warnings

# 불필요한 경고 메시지 무시
warnings.filterwarnings("ignore")

# ======================================================
# [설정] 경로를 최종 확인하세요.
# ======================================================
FAKE_DIR = r"D:\data\dataset\fake"        # 가짜 영상 경로
RAW_DIR = r"D:\data\youtube"               # 이름 변경된 원본 경로
OUTPUT_DIR = r"D:\data\dataset\real"        # 결과물 저장 경로
# ======================================================

def find_audio_offset(short_path, long_path, sr=16000):
    """두 영상의 오디오 파형을 대조하여 시작 지점을 찾습니다."""
    try:
        # 오디오 로드 (16kHz 샘플링)
        y_short, _ = librosa.load(short_path, sr=sr)
        duration = len(y_short) / sr
        y_long, _ = librosa.load(long_path, sr=sr)

        # Cross-correlation 연산으로 일치 구간 탐색
        correlation = signal.correlate(y_long, y_short, mode='valid', method='fft')
        peak_index = np.argmax(np.abs(correlation))
        
        start_time = peak_index / sr
        return start_time, duration
    except Exception:
        return None, None

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"📂 저장 폴더 생성 완료: {OUTPUT_DIR}")

    # 원본 파일 맵 생성 (ID: 파일경로)
    raw_files_map = {}
    for root, dirs, files in os.walk(RAW_DIR):
        for f in files:
            if f.lower().endswith(('.mp4', '.mkv', '.webm', '.avi')):
                # 파일명에서 확장자를 제외한 ID만 추출
                file_id = os.path.splitext(f)[0]
                raw_files_map[file_id] = os.path.join(root, f)

    fake_files = glob.glob(os.path.join(FAKE_DIR, "*.mp4"))
    print(f"🚀 총 {len(fake_files)}개의 영상 전처리를 시작합니다.")
    print("-" * 50)

    success, skip, fail = 0, 0, 0

    for i, fake_path in enumerate(fake_files):
        filename = os.path.basename(fake_path)
        
        # 가짜 영상 파일명에서 유튜브 ID 추출
        # 예: 01176--zZrDihnANpM_1--AniPortraitAudio.mp4 -> zZrDihnANpM
        try:
            yt_id = filename.split("--")[1].rsplit("_", 1)[0]
        except Exception:
            continue

        # 원본 파일 매칭
        target_raw_path = raw_files_map.get(yt_id)
        
        if not target_raw_path:
            fail += 1
            continue
        
        save_path = os.path.join(OUTPUT_DIR, f"real_{filename}")
        # 확장자가 .mp4가 아니면 강제 수정
        if not save_path.lower().endswith('.mp4'):
            save_path = os.path.splitext(save_path)[0] + ".mp4"

        if os.path.exists(save_path):
            skip += 1
            continue

        print(f"[{i+1}/{len(fake_files)}] 🔍 분석 및 추출 중: {yt_id}")

        # 오디오 분석
        start, dur = find_audio_offset(fake_path, target_raw_path)

        if start is not None:
            try:
                # ffmpeg 직접 호출 (재인코딩 포함하여 0xC00D36C4 에러 방지)
                cmd = [
                    'ffmpeg', '-y', 
                    '-ss', str(start), 
                    '-t', str(dur), 
                    '-i', target_raw_path, 
                    '-c:v', 'libx264', # 표준 코덱 지정
                    '-c:a', 'aac',     # 표준 오디오 코덱 지정
                    '-strict', 'experimental',
                    '-pix_fmt', 'yuv420p', # 호환성 높은 픽셀 포맷
                    save_path
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                print(f"   ✅ 완료: {start:.2f}s 지점")
                success += 1
            except Exception as e:
                print(f"   💥 오류: {e}")
                fail += 1
        else:
            print(f"   ❌ 싱크 실패")
            fail += 1

    print("=" * 50)
    print(f"🎉 모든 작업 종료. 성공: {success}, 건너뜀: {skip}, 실패: {fail}")

if __name__ == "__main__":
    main()