import os

# ======================================================
# [설정] 경로를 다시 확인하세요
# ======================================================
FAILED_LIST_PATH = r"D:\data\failed_list.txt"  # 실패 명단 파일
TARGET_DIR = r"D:\data\dataset\fake"          # 삭제할 파일이 있는 폴더
# ======================================================

def main():
    if not os.path.exists(FAILED_LIST_PATH):
        print(f"❌ 실패 리스트 파일을 찾을 수 없습니다: {FAILED_LIST_PATH}")
        return

    # 1. 삭제할 명단 읽기
    with open(FAILED_LIST_PATH, "r", encoding="utf-8") as f:
        files_to_delete = [line.strip() for line in f if line.strip()]

    if not files_to_delete:
        print("📝 삭제할 명단이 비어 있습니다.")
        return

    print(f"🚀 총 {len(files_to_delete)}개의 파일을 삭제합니다.")
    print("-" * 50)

    success_count = 0
    missing_count = 0

    # 2. 파일 삭제 루프
    for filename in files_to_delete:
        file_path = os.path.join(TARGET_DIR, filename)
        
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"✅ 삭제 성공: {filename}")
                success_count += 1
            except Exception as e:
                print(f"💥 삭제 실패 ({filename}): {e}")
        else:
            print(f"⏩ 이미 없음 (이미 삭제되었거나 경로 오류): {filename}")
            missing_count += 1

    print("-" * 50)
    print(f"🎉 정리 완료!")
    print(f"   - 실제 삭제됨: {success_count}개")
    print(f"   - 폴더에 없었음: {missing_count}개")
    print(f"📂 대상 폴더: {TARGET_DIR}")

if __name__ == "__main__":
    main()