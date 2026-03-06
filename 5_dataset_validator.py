import os
import glob
import pandas as pd
from tqdm import tqdm

# ⚡ decord 라이브러리 필수
try:
    from decord import VideoReader, cpu
except ImportError:
    raise ImportError("❌ decord가 설치되지 않았습니다. 'pip install decord'를 실행하세요.")

# ==========================================
# ⚙️ 설정: 검사할 데이터셋 경로
# ==========================================
BASE_DIR = r"C:\Users\leejy\Desktop\test_experiment\dataset\final_dataset_v2\train"

def check_file(path):
    """ 파일 하나를 decord로 읽어보고 성공/실패 여부 반환 """
    try:
        # CPU 모드로 읽기 시도
        vr = VideoReader(path, ctx=cpu(0))
        
        # 1. 프레임 개수 확인
        if len(vr) <= 0:
            return False, "Empty Video (0 frames)"
        
        # 2. 실제 프레임 디코딩 시도 (첫 프레임)
        _ = vr[0] 
        
        return True, "OK"
    except Exception as e:
        return False, str(e)

def main():
    print(f"🚀 [Dataset Validator] 데이터 무결성 검사 시작")
    print(f"📂 대상 경로: {BASE_DIR}")
    
    # Real / Fake 파일 리스트 확보
    real_files = glob.glob(os.path.join(BASE_DIR, "real", "*.mp4"))
    fake_files = glob.glob(os.path.join(BASE_DIR, "fake", "*.mp4"))
    
    # 숨어있는 파일 재귀 탐색 (혹시 모를 경우)
    if not real_files:
        real_files = glob.glob(os.path.join(BASE_DIR, "real", "**", "*.mp4"), recursive=True)
    
    all_files = real_files + fake_files
    print(f"📊 총 검사 대상: {len(all_files)}개 (Real: {len(real_files)}, Fake: {len(fake_files)})")
    
    bad_files = []
    
    # TQDM으로 진행 상황 표시
    for path in tqdm(all_files, desc="검사 중"):
        is_valid, reason = check_file(path)
        if not is_valid:
            bad_files.append({"path": path, "reason": reason})
            # 터미널에 실시간 출력
            print(f"\n❌ [BAD] {os.path.basename(path)} -> {reason}")

    print("\n" + "="*50)
    print(f"🏁 검사 완료!")
    
    if bad_files:
        print(f"😱 발견된 불량 파일: {len(bad_files)}개")
        print("   -> 아래 파일들은 학습 시 '검은 화면'을 유발합니다.")
        print("   -> 삭제하거나 다시 다운로드해야 합니다.")
        
        # CSV로 저장
        df = pd.DataFrame(bad_files)
        df.to_csv("bad_files_report.csv", index=False)
        print("📄 상세 목록 저장됨: bad_files_report.csv")
        
        # 자동 삭제 옵션 (주석 처리됨)
        # for item in bad_files:
        #     os.remove(item['path'])
        #     print(f"🗑️ 삭제됨: {item['path']}")
    else:
        print(f"🎉 축하합니다! 모든 파일이 정상입니다. (Decord 호환성 100%)")
        print("   -> 이제 학습을 돌려도 검은 화면이 뜨지 않습니다.")

if __name__ == "__main__":
    main()