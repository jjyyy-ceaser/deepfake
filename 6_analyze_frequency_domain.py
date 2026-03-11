import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.fftpack import fft2, fftshift

# ======================================================
# ⚙️ [설정] 베이스 경로 및 타겟 파일
# ======================================================
BASE_DIR = r"C:\Users\leejy\Desktop\test_experiment\dataset\final_dataset_v2"
FILENAME = "svd_003.mp4" # 분석할 타겟 파일 이름

# 도메인별 실제 파일 경로 매핑 (생성 스크립트 구조 반영)
DOMAIN_PATHS = {
    "Raw (Original)": os.path.join(BASE_DIR, "raw", "test", "fake", FILENAME),
    "YouTube":        os.path.join(BASE_DIR, "youtube", "fake", FILENAME),
    "Instagram":      os.path.join(BASE_DIR, "instagram", "fake", FILENAME),
    "Kakao (High)":   os.path.join(BASE_DIR, "kakao_high", "fake", FILENAME),
    "Kakao (Normal)": os.path.join(BASE_DIR, "kakao_normal", "fake", FILENAME)
}

SAVE_DIR = r"C:\Users\leejy\Desktop\test_experiment\results\frequency_analysis"
os.makedirs(SAVE_DIR, exist_ok=True)

# 차트 시각화를 위한 선 스타일 및 색상 설정
PLOT_STYLES = {
    "Raw (Original)": {"color": "black", "style": "-",  "width": 2.5, "alpha": 1.0},
    "YouTube":        {"color": "red",   "style": "--", "width": 1.5, "alpha": 0.8},
    "Instagram":      {"color": "blue",  "style": "-.", "width": 1.5, "alpha": 0.8},
    "Kakao (High)":   {"color": "green", "style": ":",  "width": 2.0, "alpha": 0.8},
    "Kakao (Normal)": {"color": "purple","style": "--", "width": 1.5, "alpha": 0.8}
}

def get_psd_1d(frame):
    """2D 이미지를 1D 파워 스펙트럼으로 변환 (방사형 평균)"""
    h, w = frame.shape
    f_shift = fftshift(fft2(frame))
    magnitude = np.abs(f_shift)
    psd2d = magnitude**2
    
    y, x = np.indices(psd2d.shape)
    r = np.sqrt((x - w//2)**2 + (y - h//2)**2).astype(int)
    
    tbin = np.bincount(r.ravel(), psd2d.ravel())
    nr = np.bincount(r.ravel())
    radial_profile = tbin / (nr + 1e-9)
    
    return radial_profile

def load_middle_frame(path):
    """영상의 '중앙 프레임'을 흑백으로 추출"""
    if not os.path.exists(path): return None
    cap = cv2.VideoCapture(path)
    
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if length == 0: 
        cap.release(); return None
    
    mid_idx = length // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_idx)
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret: return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def analyze_all_domains_frequency():
    print(f"🔬 [10.2] 주파수 도메인 분석 (모든 도메인 통합 비교: {FILENAME})")
    
    plt.figure(figsize=(12, 8))
    valid_plots = 0
    
    for domain_name, file_path in DOMAIN_PATHS.items():
        if not os.path.exists(file_path):
            print(f"   ⚠️ [Skip] 파일 없음: {domain_name}")
            continue
            
        img = load_middle_frame(file_path)
        if img is None:
            print(f"   ⚠️ [Skip] 프레임 로드 실패: {domain_name}")
            continue
            
        psd = get_psd_1d(img)
        
        # 설정된 스타일로 플롯 그리기
        style = PLOT_STYLES[domain_name]
        plt.plot(np.log10(psd + 1e-9), 
                 label=domain_name, 
                 color=style["color"], 
                 linestyle=style["style"], 
                 linewidth=style["width"],
                 alpha=style["alpha"])
                 
        valid_plots += 1
        print(f"   ✅ {domain_name:<15} 스펙트럼 추출 완료")

    if valid_plots == 0:
        print("\n❌ 분석할 수 있는 데이터가 없습니다. 파일 경로를 확인해주세요.")
        return

    # 그래프 꾸미기
    plt.title(f"Log-frequency Power Spectrum Comparison across SNS Platforms", fontsize=16, pad=15)
    plt.xlabel("Frequency Bin (Spatial Distance from Center -> Higher Frequency)", fontsize=14)
    plt.ylabel("Log Power Energy", fontsize=14)
    plt.legend(fontsize=12, loc="upper right")
    plt.grid(True, alpha=0.3)
    
    # 여백 최적화 및 저장
    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, f"frequency_compare_all_{FILENAME.split('.')[0]}.png")
    plt.savefig(save_path, dpi=300)
    print(f"\n📊 통합 비교 차트 저장 완료: {save_path}")

if __name__ == "__main__":
    analyze_all_domains_frequency()