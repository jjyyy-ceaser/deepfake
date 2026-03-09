import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.fftpack import fft2, fftshift

# ⚙️ 경로 설정 (Raw vs Youtube 폴더)
RAW_DIR = r"C:\Users\leejy\Desktop\test_experiment\dataset\final_dataset_v2\test\fake"
# 비교할 왜곡 도메인 (예: Youtube)
DIST_DIR = r"C:\Users\leejy\Desktop\test_experiment\dataset\youtube\fake" 
SAVE_DIR = r"C:\Users\leejy\Desktop\test_experiment\results\frequency_analysis"
os.makedirs(SAVE_DIR, exist_ok=True)

def get_psd_1d(frame):
    """2D 이미지를 1D 파워 스펙트럼으로 변환 (Azimuthal Average)"""
    h, w = frame.shape
    f_shift = fftshift(fft2(frame))
    magnitude = np.abs(f_shift)
    psd2d = magnitude**2
    
    # 중심에서의 거리 계산 (방사형 평균)
    y, x = np.indices(psd2d.shape)
    r = np.sqrt((x - w//2)**2 + (y - h//2)**2).astype(int)
    
    # 거리별 평균 에너지 계산 (1D)
    tbin = np.bincount(r.ravel(), psd2d.ravel())
    nr = np.bincount(r.ravel())
    radial_profile = tbin / (nr + 1e-9)
    
    return radial_profile

def load_middle_frame(path):
    """
    [수정] SVD 생성 영상의 1번 프레임(Condition)이 아닌,
    조작 흔적이 누적된 '중앙 프레임'을 추출함.
    """
    if not os.path.exists(path): return None
    cap = cv2.VideoCapture(path)
    
    # 총 프레임 수 확인
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if length == 0: 
        cap.release(); return None
    
    # 🔧 [핵심 Fix] 중앙 지점으로 점프
    mid_idx = length // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_idx)
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret: return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def analyze_frequency_drop():
    print("🔬 [10.2] 주파수 도메인 분석 (Space-Frequency Drop 검증 - Middle Frame)")
    
    # 샘플 파일 찾기 (같은 이름의 파일 비교)
    if not os.path.exists(RAW_DIR) or not os.path.exists(DIST_DIR):
        print("❌ 경로가 존재하지 않습니다.")
        return

    raw_files = os.listdir(RAW_DIR)[:1] # 1개만 샘플링
    if not raw_files: 
        print("❌ 분석할 파일이 없습니다.")
        return

    filename = raw_files[0]
    raw_path = os.path.join(RAW_DIR, filename)
    dist_path = os.path.join(DIST_DIR, filename)
    
    # 🔧 중앙 프레임 로드
    img_raw = load_middle_frame(raw_path)
    img_dist = load_middle_frame(dist_path)

    if img_raw is None or img_dist is None: 
        print(f"❌ 영상 로드 실패: {filename}")
        return

    # 1D Power Spectrum 추출
    psd_raw = get_psd_1d(img_raw)
    psd_dist = get_psd_1d(img_dist)

    # 시각화 (Log-frequency Power Spectrum)
    plt.figure(figsize=(10, 6))
    plt.plot(np.log10(psd_raw + 1e-9), label='Raw (SVD Original)', color='blue', linewidth=2)
    plt.plot(np.log10(psd_dist + 1e-9), label='YouTube (Distorted)', color='red', linestyle='--', linewidth=2)
    
    plt.title(f"Log-frequency Power Spectrum: {filename} (Frame {12})", fontsize=14)
    plt.xlabel("Frequency Bin (Spatial Distance from Center)", fontsize=12)
    plt.ylabel("Log Power Energy", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(SAVE_DIR, "frequency_drop_analysis.png")
    plt.savefig(save_path)
    print(f"   📊 스펙트럼 분석 차트 저장됨: {save_path}")

if __name__ == "__main__":
    analyze_frequency_drop()