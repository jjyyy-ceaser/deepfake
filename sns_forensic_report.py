import os
import struct
import subprocess
import glob
import math
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm

# =========================================================
# 📂 설정: Rename 코드와 100% 동일한 구조
# =========================================================
BASE_DIR = r"C:\Users\leejy\Desktop\test_experiment\dataset\sns_analysis"
ORIGINAL_DIR = os.path.join(BASE_DIR, "0_original")

# 분석 대상 플랫폼 리스트 (폴더명, 태그)
TARGET_PLATFORMS = [
    {"folder": "1_youtube",       "tag": "YT"},
    {"folder": "2_instagram",     "tag": "IG"},
    {"folder": "3_kakao_normal",  "tag": "KK_NM"},
    {"folder": "3_kakao_high",    "tag": "KK_HQ"}
]

# =========================================================
# 🛠️ 함수 정의 (복붙 실수 방지를 위해 한 덩어리로 제공)
# =========================================================

def parse_mp4_atoms(file_path):
    """MP4 구조(Box Sequence) 추출"""
    atoms = []
    try:
        file_size = os.path.getsize(file_path)
        with open(file_path, "rb") as f:
            while f.tell() < file_size:
                size_bytes = f.read(4)
                type_bytes = f.read(4)
                if len(size_bytes) < 4 or len(type_bytes) < 4: break
                
                atom_size = struct.unpack(">I", size_bytes)[0]
                atom_type = type_bytes.decode('utf-8', errors='ignore')
                
                if atom_type.isalnum(): atoms.append(atom_type)
                
                if atom_size == 0: break 
                if atom_size == 1: 
                    f.seek(8, 1)
                    if atom_type == 'mdat': break
                else: f.seek(atom_size - 8, 1)
    except Exception: return "Error"
    return "-".join(atoms)

def get_video_metadata(file_path):
    """FFprobe로 메타데이터 추출"""
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", 
           "-show_entries", "stream=width,height,codec_name,profile,avg_frame_rate,bit_rate", 
           "-of", "default=noprint_wrappers=1:nokey=1", file_path]
    try:
        output = subprocess.check_output(cmd).decode('utf-8').strip().split('\n')
        if len(output) < 6: return None
        
        width = int(output[0]) if output[0].isdigit() else 0
        height = int(output[1]) if output[1].isdigit() else 0
        codec = output[2]
        profile = output[3]
        
        fps_val = output[4]
        if '/' in fps_val:
            num, den = fps_val.split('/')
            fps = float(num) / float(den)
        else:
            fps = float(fps_val)
            
        bitrate = int(output[5]) if output[5].isdigit() else 0
        
        return {"width": width, "height": height, "codec": codec, "profile": profile, "fps": fps, "bitrate": bitrate}
    except Exception as e:
        return None

def estimate_crf(orig_bitrate, dist_bitrate):
    """비트레이트 기반 CRF 추정"""
    if orig_bitrate == 0 or dist_bitrate == 0: return 0
    ratio = orig_bitrate / dist_bitrate
    if ratio < 1: ratio = 1
    return round(18 + (6 * math.log2(ratio)), 2)

def measure_block_artifact(file_path):
    """8x8 블록 노이즈 측정"""
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened(): return 0.0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total_frames // 2))
    ret, frame = cap.read()
    cap.release()
    
    if not ret: return 0.0
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    row_diff = np.abs(gray[1:, :] - gray[:-1, :])
    col_diff = np.abs(gray[:, 1:] - gray[:, :-1])
    
    block_energy_h = np.mean(row_diff[7::8, :])
    block_energy_v = np.mean(col_diff[:, 7::8])
    
    non_block_h = (np.sum(row_diff) - np.sum(row_diff[7::8, :])) / (row_diff.size - row_diff[7::8, :].size)
    non_block_v = (np.sum(col_diff) - np.sum(col_diff[:, 7::8])) / (col_diff.size - col_diff[:, 7::8].size)
    
    return round((block_energy_h + block_energy_v) / (non_block_h + non_block_v + 1e-6), 4)

def main():
    if not os.path.exists(ORIGINAL_DIR):
        print(f"❌ 원본 폴더를 찾을 수 없습니다: {ORIGINAL_DIR}")
        return

    # 원본 파일 목록 (S01_ORG.mp4 ...)
    orig_files = sorted(glob.glob(os.path.join(ORIGINAL_DIR, "*.mp4")))
    results = []
    
    print(f"🕵️ Forensic Report 시작: 원본 {len(orig_files)}개 분석")
    
    for orig_path in tqdm(orig_files, desc="Total Progress"):
        orig_filename = os.path.basename(orig_path)
        
        # 파일명에서 Index 추출 (S01_ORG.mp4 -> S01)
        if "_" in orig_filename:
            file_index = orig_filename.split('_')[0] 
        else:
            print(f"⚠️ 파일명 형식 오류 (Skip): {orig_filename}")
            continue

        orig_meta = get_video_metadata(orig_path)
        if not orig_meta: continue

        # 각 플랫폼별 대응 파일 찾기
        for info in TARGET_PLATFORMS:
            platform_folder = os.path.join(BASE_DIR, info["folder"])
            target_filename = f"{file_index}_{info['tag']}.mp4" # 예: S01_KK_HQ.mp4
            dist_path = os.path.join(platform_folder, target_filename)
            
            if not os.path.exists(dist_path):
                # 파일이 없으면 조용히 넘어감 (해당 플랫폼 테스트 안 했을 수도 있으니)
                continue
            
            dist_meta = get_video_metadata(dist_path)
            if not dist_meta: continue
            
            # 분석 수행
            box_seq = parse_mp4_atoms(dist_path)
            est_crf = estimate_crf(orig_meta['bitrate'], dist_meta['bitrate'])
            blockiness = measure_block_artifact(dist_path)
            
            results.append({
                "Index": file_index,
                "Platform": info["folder"].upper(),
                "Orig_Res": f"{orig_meta['width']}x{orig_meta['height']}",
                "Dist_Res": f"{dist_meta['width']}x{dist_meta['height']}",
                "FPS_Diff": round(orig_meta['fps'] - dist_meta['fps'], 2),
                "Codec": dist_meta['codec'],
                "Box_Sequence": box_seq,
                "Est_CRF": est_crf,
                "Blockiness": blockiness,
                "Bitrate_Loss(%)": round((1 - dist_meta['bitrate']/orig_meta['bitrate'])*100, 1)
            })

    # CSV 저장
    df = pd.DataFrame(results)
    save_path = os.path.join(BASE_DIR, "final_forensic_report.csv")
    df.to_csv(save_path, index=False)
    
    print("\n" + "="*50)
    print(f"🎉 분석 완료! 리포트 저장됨: {save_path}")
    print("="*50)
    
    if not df.empty:
        print(df.groupby("Platform")[["Est_CRF", "Blockiness", "Bitrate_Loss(%)"]].mean())
    else:
        print("⚠️ 결과 데이터가 없습니다. 파일명이나 폴더 경로를 확인해주세요.")

if __name__ == "__main__":
    main()