import cv2
import numpy as np
import os
import pandas as pd
import subprocess
import struct
from tqdm import tqdm
import glob
import math

# 📂 경로 설정
BASE_DIR = "dataset/sns_analysis"
ORIGINAL_DIR = os.path.join(BASE_DIR, "00_Original")
PLATFORMS = ["01_YouTube", "02_Instagram", "03_Facebook", "04_KakaoTalk", "05_Telegram"]

def get_ffprobe_metadata(video_path):
    """
    FFprobe를 사용하여 상세 코덱 정보 및 비트레이트 추출
    (Yang et al. 2024 근거: 코덱 프로파일 및 레벨 분석)
    """
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,codec_name,profile,avg_frame_rate,bit_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    try:
        output = subprocess.check_output(cmd).decode('utf-8').strip().split('\n')
        # 출력 순서: width, height, codec, profile, fps, bitrate
        width = int(output[0])
        height = int(output[1])
        codec = output[2]
        profile = output[3]
        
        fps_str = output[4].split('/')
        fps = float(fps_str[0]) / float(fps_str[1]) if len(fps_str) == 2 else float(output[4])
        
        bitrate = int(output[5]) if output[5].isdigit() else 0
        
        return {
            "width": width, "height": height, "codec": codec, 
            "profile": profile, "fps": fps, "bitrate": bitrate
        }
    except Exception as e:
        print(f"⚠️ FFprobe Error on {video_path}: {e}")
        return None

def parse_mp4_box_sequence(video_path):
    """
    MP4 파일의 최상위 Box Sequence(Atom) 구조 추출
    (Yang et al. 2024 근거: 플랫폼 식별 지문)
    """
    boxes = []
    file_size = os.path.getsize(video_path)
    
    with open(video_path, "rb") as f:
        while f.tell() < file_size:
            try:
                # Read Box Size (4 bytes) and Type (4 bytes)
                size_bytes = f.read(4)
                type_bytes = f.read(4)
                
                if len(size_bytes) < 4 or len(type_bytes) < 4:
                    break
                    
                size = struct.unpack(">I", size_bytes)[0]
                box_type = type_bytes.decode('utf-8', errors='ignore')
                
                boxes.append(box_type)
                
                if size == 0: # Last box
                    break
                if size == 1: # Extended size (skip logic for simplicity)
                    f.seek(8, 1) # Skip large size
                    
                # Skip to next box
                f.seek(size - 8, 1)
            except Exception:
                break
                
    return "-".join(boxes)  # 예: "ftyp-moov-mdat"

def estimate_crf(orig_bitrate, dist_bitrate, dist_res):
    """
    비트레이트 손실률 기반 CRF 추정 (Montibeller et al. heuristic)
    """
    if orig_bitrate == 0 or dist_bitrate == 0:
        return 0
    
    loss_ratio = (orig_bitrate - dist_bitrate) / orig_bitrate
    
    # Heuristic Formula: 손실률이 높을수록 CRF가 높음 (기본값 23 기준)
    # 실제 Montibeller 수식은 복잡하지만, 여기선 근사치 적용
    estimated_crf = 23 + (loss_ratio * 20) 
    return round(estimated_crf, 1)

def calculate_blockiness(image):
    """
    8x8 격자 블록 아티팩트 강도 측정 (Li et al. PLADA 근거)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 수평/수직 경계 강도 계산
    # 8번째 픽셀마다 경계가 뚜렷하면 Blockiness가 높은 것임
    h, w = gray.shape
    
    # 간단한 고주파 필터링 후 8의 배수 위치의 에너지 측정
    edge_h = np.abs(gray[1:, :] - gray[:-1, :])
    edge_v = np.abs(gray[:, 1:] - gray[:, :-1])
    
    # 8의 배수 인덱스에서의 에지 강도 평균
    block_energy_h = np.mean(edge_h[7::8, :])
    block_energy_v = np.mean(edge_v[:, 7::8])
    
    # 일반적인 에지 강도 평균 (비교군)
    non_block_energy_h = np.mean(edge_h)
    non_block_energy_v = np.mean(edge_v)
    
    # 블록 비율 (1.0 이상이면 블록 현상 존재)
    score = (block_energy_h + block_energy_v) / (non_block_energy_h + non_block_energy_v + 1e-6)
    return score

def main():
    results = []
    orig_files = glob.glob(os.path.join(ORIGINAL_DIR, "*.mp4"))
    print(f"🔬 Forensic 분석 시작: 원본 {len(orig_files)}개")
    
    for orig_path in tqdm(orig_files, desc="Processing"):
        filename = os.path.basename(orig_path)
        
        # 1. 원본 메타데이터 (FFprobe)
        orig_meta = get_ffprobe_metadata(orig_path)
        if not orig_meta: continue
        
        for platform in PLATFORMS:
            platform_name = platform.split("_")[1]
            dist_path = os.path.join(BASE_DIR, platform, filename)
            
            if not os.path.exists(dist_path): continue
            
            # 2. SNS 영상 메타데이터 (FFprobe)
            dist_meta = get_ffprobe_metadata(dist_path)
            
            # 3. 구조적 왜곡: Box Sequence (Yang et al.)
            box_seq = parse_mp4_box_sequence(dist_path)
            
            # 4. 물리적 왜곡: CRF 추정 (Montibeller et al.)
            est_crf = estimate_crf(orig_meta['bitrate'], dist_meta['bitrate'], dist_meta['height'])
            
            # 5. 기만적 아티팩트: 8x8 Blockiness (Li et al.)
            # 영상의 첫 프레임과 중간 프레임 샘플링
            cap = cv2.VideoCapture(dist_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
            ret, frame = cap.read()
            block_score = calculate_blockiness(frame) if ret else 0
            cap.release()
            
            results.append({
                "filename": filename,
                "platform": platform_name,
                "orig_codec": orig_meta['codec'],
                "dist_codec": dist_meta['codec'],        # H.264 vs HEVC 확인
                "dist_profile": dist_meta['profile'],    # Main vs High 확인
                "orig_res": f"{orig_meta['width']}x{orig_meta['height']}",
                "dist_res": f"{dist_meta['width']}x{dist_meta['height']}",
                "box_sequence": box_seq,                 # 구조적 지문 (예: ftyp-moov-mdat)
                "bitrate_drop_rate": round((orig_meta['bitrate'] - dist_meta['bitrate']) / orig_meta['bitrate'], 2),
                "estimated_crf": est_crf,                # 추정 CRF
                "block_effect_score": round(block_score, 3) # 8x8 블록 강도
            })
            
    df = pd.DataFrame(results)
    df.to_csv("sns_forensic_report.csv", index=False)
    print("\n🎉 Forensic Report Generated: sns_forensic_report.csv")
    print(df.groupby("platform")[["estimated_crf", "block_effect_score", "bitrate_drop_rate"]].mean())

if __name__ == "__main__":
    main()