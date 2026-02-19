import os

# =========================================================
# ⚙️ 경로 설정 (여기를 정확히 확인해주세요!)
# =========================================================
BASE_DIR = r"C:\Users\leejy\Desktop\test_experiment\dataset\sns_analysis"

# [폴더명, 태그명] 매핑
# 실제 존재하는 폴더 이름과 정확히 일치해야 합니다.
FOLDERS_MAP = [
    {"path": "0_original",      "tag": "ORG"},   # 원본
    {"path": "1_youtube",       "tag": "YT"},    # 유튜브
    {"path": "2_instagram",     "tag": "IG"},    # 인스타
    {"path": "3_kakao_normal",  "tag": "KK_NM"}, # 카톡 일반
    {"path": "3_kakao_high",    "tag": "KK_HQ"}  # 카톡 고화질
]

def rename_all():
    print(f"🚀 파일명 정리 시작: {BASE_DIR}")
    
    for info in FOLDERS_MAP:
        folder_path = os.path.join(BASE_DIR, info["path"])
        tag = info["tag"]
        
        if not os.path.exists(folder_path):
            print(f"⚠️ 폴더 없음 (Pass): {folder_path}")
            continue
            
        # mp4 파일만 가져와서 이름순 정렬
        files = [f for f in os.listdir(folder_path) if f.lower().endswith('.mp4')]
        files.sort()
        
        print(f"📂 [{info['path']}] -> {len(files)}개 파일 처리 중...")
        
        for i, old_name in enumerate(files, start=1):
            # 이미 이름이 S01_TAG.mp4 형식이면 건너뜀 (중복 방지)
            expected_name = f"S{i:02d}_{tag}.mp4"
            if old_name == expected_name:
                continue
                
            old_file = os.path.join(folder_path, old_name)
            new_file = os.path.join(folder_path, expected_name)
            
            try:
                os.rename(old_file, new_file)
            except OSError as e:
                print(f"   ❌ 이름 변경 실패: {old_name} -> {e}")

    print("\n✨ 파일명 정리 완료! 이제 분석 코드를 실행하세요.")

if __name__ == "__main__":
    rename_all()