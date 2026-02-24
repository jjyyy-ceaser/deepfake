import os
import struct
import subprocess
import glob
import math
import numpy as np
import cv2
import pandas as pd
import json  # JSON 파싱을 위해 추가
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
# 🛠️ 함수 정의
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
    """FFprobe를 JSON 모드로 실행하여 데이터 밀림 현상을 완벽 차단"""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0", 
        "-show_entries", "stream=width,height,codec_name,profile,avg_frame_rate,bit_rate", 
        "-show_entries", "format=bit_rate,duration", # 비트레이트 누락 방지를 위해 format 정보 추가
        "-of", "json", file_path
    ]
    try:
        output = subprocess.check_output(cmd).decode('utf-8')
        data = json.loads(output)
        
        if 'streams' not in data or len(data['streams']) == 0:
            return None
            
        stream = data['streams'][0]
        fmt = data.get('format', {})
        
        # 1. 해상도 및 코덱 (Key 직접 접근)
        width = int(stream.get('width', 0))
        height = int(stream.get('height', 0))
        codec = stream.get('codec_name', 'unknown')
        profile = stream.get('profile', 'unknown')
        
        # 2. FPS 계산 (예: "30000/1001" 처리)
        fps_val = stream.get('avg_frame_rate', '0/0')
        if '/' in fps_val:
            num, den = map(float, fps_val.split('/'))
            fps = num / den if den != 0 else 0
        else:
            fps = float(fps_val)
            
        # 3. 비트레이트 (스트림 -> 포맷 -> 직접 계산 순으로 탐색)
        bitrate = int(stream.get('bit_rate', 0))
        if bitrate == 0:
            bitrate = int(fmt.get('bit_rate', 0))
            
        # 극한의 상황 (플랫폼이 비트레이트 메타데이터를 지웠을 때) 파일 크기로 역산
        if bitrate == 0:
            file_size = os.path.getsize(file_path)
            duration = float(fmt.get('duration', 10.0))
            if duration > 0:
                bitrate = int((file_size * 8) / duration)
        
        return {
            "width": width, "height": height, 
            "codec": codec, "profile": profile, 
            "fps": fps, "bitrate": bitrate
        }
    except Exception as e:
        print(f"⚠️ Metadata Error ({file_path}): {e}")
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

    orig_files = sorted(glob.glob(os.path.join(ORIGINAL_DIR, "*.mp4")))
    results = []
    
    print(f"🕵️ Forensic Report 시작: 원본 {len(orig_files)}개 분석")
    
    for orig_path in tqdm(orig_files, desc="Total Progress"):
        orig_filename = os.path.basename(orig_path)
        
        if "_" in orig_filename:
            file_index = orig_filename.split('_')[0] 
        else:
            continue

        orig_meta = get_video_metadata(orig_path)
        if not orig_meta: continue

        for info in TARGET_PLATFORMS:
            platform_folder = os.path.join(BASE_DIR, info["folder"])
            target_filename = f"{file_index}_{info['tag']}.mp4"
            dist_path = os.path.join(platform_folder, target_filename)
            
            if not os.path.exists(dist_path):
                continue
            
            dist_meta = get_video_metadata(dist_path)
            if not dist_meta: continue
            
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

    df = pd.DataFrame(results)
    save_path = os.path.join(BASE_DIR, "final_forensic_report.csv")
    df.to_csv(save_path, index=False)
    
    print("\n" + "="*50)
    print(f"🎉 분석 완료! 리포트 저장됨: {save_path}")
    print("="*50)

if __name__ == "__main__":
    main()