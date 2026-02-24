import os
import subprocess
import math
import numpy as np
import cv2
import pandas as pd
import json
from tqdm import tqdm

# =========================================================
# 📂 설정: Case 1(원본)과 Case 4(변형) 비교를 위한 경로 지정
# =========================================================
BASE_DIR = r"C:\Users\leejy\Desktop\test_experiment\dataset\processed_cases\train"

# Case 1을 원본으로, Case 4를 비교 대상으로 설정
ORIGINAL_DIR = os.path.join(BASE_DIR, r"case1_original\real")
DISTORTED_DIR = os.path.join(BASE_DIR, r"case4_mixed\real")

# =========================================================
# 🛠️ 핵심 분석 함수 (유지)
# =========================================================
def get_video_metadata(file_path):
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0", 
        "-show_entries", "stream=width,height,codec_name,profile,avg_frame_rate,bit_rate", 
        "-show_entries", "format=bit_rate,duration",
        "-of", "json", file_path
    ]
    try:
        output = subprocess.check_output(cmd).decode('utf-8')
        data = json.loads(output)
        
        if 'streams' not in data or len(data['streams']) == 0:
            return None
            
        stream = data['streams'][0]
        fmt = data.get('format', {})
        
        width = int(stream.get('width', 0))
        height = int(stream.get('height', 0))
        codec = stream.get('codec_name', 'unknown')
        
        fps_val = stream.get('avg_frame_rate', '0/0')
        if '/' in fps_val:
            num, den = map(float, fps_val.split('/'))
            fps = num / den if den != 0 else 0
        else:
            fps = float(fps_val)
            
        bitrate = int(stream.get('bit_rate', 0))
        if bitrate == 0:
            bitrate = int(fmt.get('bit_rate', 0))
            
        if bitrate == 0:
            file_size = os.path.getsize(file_path)
            duration = float(fmt.get('duration', 10.0))
            if duration > 0:
                bitrate = int((file_size * 8) / duration)
        
        return {
            "width": width, "height": height, 
            "codec": codec, "fps": fps, "bitrate": bitrate
        }
    except Exception as e:
        print(f"⚠️ Metadata Error ({file_path}): {e}")
        return None

def estimate_crf(orig_bitrate, dist_bitrate):
    if orig_bitrate == 0 or dist_bitrate == 0: return 0
    ratio = orig_bitrate / dist_bitrate
    if ratio < 1: ratio = 1
    return round(18 + (6 * math.log2(ratio)), 2)

def measure_block_artifact(file_path):
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

# =========================================================
# 🚀 검증 실행 로직
# =========================================================
def main():
    if not os.path.exists(ORIGINAL_DIR):
        print(f"❌ 원본 폴더(Case 1)를 찾을 수 없습니다: {ORIGINAL_DIR}")
        return
        
    orig_files = [f for f in os.listdir(ORIGINAL_DIR) if f.endswith('.mp4')]
    results = []
    
    print(f"🕵️ Case 4 타당성 검증 시작: 총 {len(orig_files)}개 파일 분석")
    
    for filename in tqdm(orig_files, desc="Validation Progress"):
        orig_path = os.path.join(ORIGINAL_DIR, filename)
        dist_path = os.path.join(DISTORTED_DIR, filename)
        
        # Case 4 파일이 존재하는지 확인
        if not os.path.exists(dist_path):
            continue

        orig_meta = get_video_metadata(orig_path)
        dist_meta = get_video_metadata(dist_path)
        
        if not orig_meta or not dist_meta: 
            continue
            
        est_crf = estimate_crf(orig_meta['bitrate'], dist_meta['bitrate'])
        blockiness = measure_block_artifact(dist_path)
        bitrate_loss = round((1 - dist_meta['bitrate'] / orig_meta['bitrate']) * 100, 1)
        
        results.append({
            "Filename": filename,
            "Target": "CASE4_MIXED",
            "Orig_Res": f"{orig_meta['width']}x{orig_meta['height']}",
            "Dist_Res": f"{dist_meta['width']}x{dist_meta['height']}",
            "Orig_Bitrate": orig_meta['bitrate'],
            "Dist_Bitrate": dist_meta['bitrate'],
            "Est_CRF": est_crf,
            "Blockiness": blockiness,
            "Bitrate_Loss(%)": bitrate_loss
        })

    if results:
        df = pd.DataFrame(results)
        
        # 전체 평균 수치 도출
        avg_loss = df["Bitrate_Loss(%)"].mean()
        avg_crf = df["Est_CRF"].mean()
        avg_block = df["Blockiness"].mean()
        
        print("\n" + "="*50)
        print("📊 [Case 4 검증 결과 요약]")
        print(f"평균 비트레이트 손실률: {avg_loss:.1f}% (목표: 유튜브 93.6%)")
        print(f"평균 추정 CRF: {avg_crf:.1f} (목표: 유튜브 44.9)")
        print(f"평균 Blockiness: {avg_block:.4f} (목표: 유튜브 1.198)")
        print("="*50)
        
        save_path = os.path.join(BASE_DIR, "case4_validation_report.csv")
        df.to_csv(save_path, index=False)
        print(f"✅ 리포트 저장 완료: {save_path}")
    else:
        print("⚠️ 분석할 매칭 파일이 없습니다. 경로를 다시 확인해주세요.")

if __name__ == "__main__":
    main()