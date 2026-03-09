import os
import struct
import subprocess
import pandas as pd
import json
from tqdm import tqdm

# ======================================================
# ⚙️ [설정] 분석할 데이터셋 경로 및 결과 파일
# ======================================================
DATASET_ROOT = r"C:\Users\leejy\Desktop\test_experiment\dataset\sns"
DOMAINS = ["raw", "youtube", "instagram", "kakao_high", "kakao_normal"]

# 결과 저장 경로
OUTPUT_DIR = r"C:\Users\leejy\Desktop\test_experiment\results"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "distortion_analysis_report.csv")
OUTPUT_XLSX = os.path.join(OUTPUT_DIR, "distortion_analysis_report.xlsx")

# FFprobe 경로 (환경변수 등록 가정)
FFPROBE_CMD = "ffprobe"

def get_sota_metadata(file_path):
    """
    [SOTA급 정밀 분석] 논문 심사 방어용 핵심 지표 + 🛡️ 무결성 패치(Fallback)
    
    🛡️ Fallback Logic (이중 안전장치):
    1. Bitrate: 메타데이터 누락 시 -> (파일크기 * 8) / 재생시간으로 정밀 역산
    2. Frame Count: 메타데이터 누락 시 -> (재생시간 * FPS)로 추정
    """
    try:
        # 1. FFprobe 명령어로 Format(컨테이너) 정보와 Stream(비디오) 정보를 모두 가져옴
        cmd = [
            FFPROBE_CMD, 
            "-v", "error", 
            "-select_streams", "v:0", 
            # format=duration,size,bit_rate 추가 (역산용 데이터)
            "-show_entries", "format=duration,size,bit_rate:stream=width,height,codec_name,bit_rate,pix_fmt,r_frame_rate,nb_frames,duration,color_space,color_transfer", 
            "-of", "json", 
            file_path
        ]
        
        # 윈도우 팝업 방지
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, startupinfo=startupinfo)
        info = json.loads(result.stdout)
        
        if 'streams' in info and len(info['streams']) > 0:
            stream = info['streams'][0]
            fmt = info.get('format', {})
            
            # --- 기본 정보 ---
            w = int(stream.get('width', 0))
            h = int(stream.get('height', 0))
            codec = stream.get('codec_name', 'unknown')
            pix_fmt = stream.get('pix_fmt', 'unknown')
            c_space = stream.get('color_space', 'unknown')
            c_trans = stream.get('color_transfer', 'unknown')
            
            # --- FPS 계산 ---
            fps_str = stream.get('r_frame_rate', '0/1')
            if '/' in fps_str:
                num, den = map(int, fps_str.split('/'))
                fps = round(num / den, 2) if den != 0 else 0
            else:
                fps = round(float(fps_str), 2)
            
            # --- Duration 확보 (Stream 우선, 없으면 Format) ---
            duration = float(stream.get('duration', 0))
            if duration == 0:
                duration = float(fmt.get('duration', 0))
                
            # --- 🛡️ [Fallback 1] Bitrate 계산 ---
            # 1순위: Stream Header, 2순위: Format Header, 3순위: (파일크기/시간) 역산
            bitrate = 0
            if 'bit_rate' in stream:
                bitrate = float(stream['bit_rate'])
            elif 'bit_rate' in fmt:
                bitrate = float(fmt['bit_rate'])
            
            # 메타데이터가 없거나 0이면 역산 (kbps)
            if bitrate == 0 and duration > 0:
                file_size_bits = os.path.getsize(file_path) * 8
                bitrate = file_size_bits / duration
            
            bitrate_kbps = round(bitrate / 1000, 2)
            
            # --- 🛡️ [Fallback 2] Frame Count 계산 ---
            # 1순위: Stream Header, 2순위: (FPS * 시간) 역산
            frame_count = 0
            if 'nb_frames' in stream and stream['nb_frames'] != 'N/A':
                try:
                    frame_count = int(stream['nb_frames'])
                except:
                    frame_count = 0
            
            # nb_frames가 없거나 0이면 역산
            if frame_count == 0 and duration > 0 and fps > 0:
                frame_count = int(duration * fps)
            
            return w, h, codec, bitrate_kbps, pix_fmt, fps, frame_count, c_space, c_trans
            
    except Exception:
        pass
        
    return 0, 0, "error", 0.0, "error", 0.0, 0, "error", "error"

def analyze_mp4_structure(file_path):
    """
    MP4 Box 구조 분석 (스트리밍 최적화 여부)
    """
    atoms = []
    try:
        file_size = os.path.getsize(file_path)
        with open(file_path, "rb") as f:
            while f.tell() < file_size:
                header = f.read(8)
                if len(header) < 8: break
                
                size = struct.unpack(">I", header[:4])[0]
                box_type = header[4:].decode('latin1')
                atoms.append(box_type)
                
                if size == 1:
                    f.seek(8, 1)
                    size = struct.unpack(">Q", f.read(8))[0]
                    f.seek(size - 16, 1)
                elif size == 0:
                    break
                else:
                    f.seek(size - 8, 1)
                    
                if 'moov' in atoms and 'mdat' in atoms:
                    break
    except Exception:
        return "error"
    
    return "-".join(atoms)

def run_analysis():
    print("🔬 [SOTA + Fallback] SNS 플랫폼 왜곡 정밀 분석 (데이터 무결성 강화)")
    print(f"📂 분석 대상: {DATASET_ROOT}\n")
    
    results = []
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for domain in DOMAINS:
        domain_path = os.path.join(DATASET_ROOT, domain)
        if not os.path.exists(domain_path):
            print(f"⚠️ [Skip] 폴더 없음: {domain}")
            continue
            
        files = []
        for root, _, filenames in os.walk(domain_path):
            for f in filenames:
                if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    files.append(os.path.join(root, f))
        
        if not files:
            continue
            
        print(f"🚀 Analyzing {domain.upper()} ({len(files)} files)...")
        
        for file_path in tqdm(files, unit="video"):
            w, h, codec, bitrate, pix_fmt, fps, frames, c_space, c_trans = get_sota_metadata(file_path)
            box_seq = analyze_mp4_structure(file_path)
            
            results.append({
                "Domain": domain,
                "Type": "Fake" if "fake" in file_path.lower() else "Real",
                "Filename": os.path.basename(file_path),
                "Resolution": f"{w}x{h}",
                "Codec": codec,
                "Bitrate(kbps)": bitrate,
                "Color_Fmt": pix_fmt,
                "FPS": fps,
                "Total_Frames": frames,
                "Color_Space": c_space,
                "Color_Transfer": c_trans,
                "Box_Structure": box_seq
            })

    if results:
        df = pd.DataFrame(results)
        
        # CSV 저장
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n💾 CSV 저장 완료: {OUTPUT_CSV}")
        
        # Excel 저장
        try:
            df.to_excel(OUTPUT_XLSX, index=False)
            print(f"📊 Excel 저장 완료: {OUTPUT_XLSX}")
        except Exception as e:
            print(f"⚠️ Excel 저장 실패: {e}")

        # 요약 출력
        print("\n[🧐 플랫폼별 왜곡 특성 요약]")
        summary = df.groupby("Domain").agg({
            "Resolution": lambda x: x.mode()[0] if not x.mode().empty else "-",
            "Total_Frames": "mean",
            "FPS": lambda x: x.mode()[0] if not x.mode().empty else "-",
            "Color_Space": lambda x: x.mode()[0] if not x.mode().empty else "-",
            "Bitrate(kbps)": "mean"
        }).sort_values("Bitrate(kbps)", ascending=False)
        
        print(summary.to_string())
        
    else:
        print("❌ 분석할 데이터가 없습니다.")

if __name__ == "__main__":
    run_analysis()