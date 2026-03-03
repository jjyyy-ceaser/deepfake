import os
import yt_dlp
import re

# ======================================================
# [설정] 경로를 다시 한 번 확인해라
# ======================================================
RAW_DIR = r"D:\data\youtube"
YT_LIST_PATH = r"D:\data\yt_list.txt"
# ======================================================

def get_yt_title(url):
    """유튜브 주소에서 제목만 빠르게 가져온다."""
    ydl_opts = {'quiet': True, 'no_warnings': True, 'extract_flat': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return info.get('title', '')

def main():
    if not os.path.exists(YT_LIST_PATH):
        print(f"❌ 리스트 파일을 찾을 수 없다: {YT_LIST_PATH}")
        return

    # 1. 원본 폴더 파일 리스트 로드
    raw_files = [f for f in os.listdir(RAW_DIR) if f.lower().endswith(('.mp4', '.mkv', '.webm'))]
    print(f"📊 원본 폴더 내 {len(raw_files)}개의 파일을 대조한다.")

    # 2. 유튜브 리스트 읽기
    with open(YT_LIST_PATH, "r", encoding='utf-8') as f:
        urls = [line.strip() for line in f if line.strip()]

    print(f"🚀 총 {len(urls)}개의 주소를 처리한다.")
    print("-" * 50)

    for url in urls:
        # 1. URL에서 11자리 ID 먼저 추출 (NameError 방지)
        match = re.search(r"v=([a-zA-Z0-9_-]{11})", url)
        if not match:
            continue
        
        yt_id = match.group(1) # 여기서 yt_id가 확실히 정의된다.
        
        try:
            # 이미 이름이 ID로 되어 있는 파일이 있는지 확인
            if any(yt_id in f for f in os.listdir(RAW_DIR) if len(f.split('.')[0]) == 11):
                continue

            # 2. 제목 가져오기
            yt_title = get_yt_title(url)
            if not yt_title: continue

            # 파일명 규칙에 맞게 제목 정규화 (특수문자 제거 등)
            # 유튜브 제목에 흔히 쓰이는 특수문자들을 고려한다.
            clean_title = re.sub(r'[\\/*?:"<>|]', '', yt_title)
            
            for f_name in raw_files:
                # 파일 이름에 제목의 핵심 키워드가 들어있는지 확인
                # 제목이 너무 길면 잘릴 수 있으므로 앞부분 15자 정도를 대조한다.
                if clean_title[:15] in f_name or yt_title[:15] in f_name:
                    old_path = os.path.join(RAW_DIR, f_name)
                    ext = os.path.splitext(f_name)[1]
                    new_path = os.path.join(RAW_DIR, f"{yt_id}{ext}")
                    
                    if not os.path.exists(new_path):
                        os.rename(old_path, new_path)
                        print(f"✅ 변경: {f_name[:30]}... -> {yt_id}{ext}")
                        # 리스트에서 제거하여 다음 대조 속도 향상
                        raw_files.remove(f_name)
                        break
                        
        except Exception as e:
            print(f"⚠️ 처리 중 오류 (ID: {yt_id}): {e}")

    print("-" * 50)
    print("🎉 이름 변경이 완료되었다. 이제 sync_real_final.py를 다시 실행해라.")

if __name__ == "__main__":
    main()