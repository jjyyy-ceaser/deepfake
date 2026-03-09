import yt_dlp
import os

# ======================================================
# ⚙️ [설정] 파일 경로 및 저장 위치
# ======================================================
LIST_FILE_PATH = r"C:\Users\leejy\Desktop\test_experiment\instagram_list.txt"
# 🔧 방금 추출한 인스타그램 쿠키 파일 경로
COOKIE_FILE_PATH = r"C:\Users\leejy\Desktop\test_experiment\instagram_cookies.txt"
OUTPUT_FOLDER = os.path.join("downloads", "instagram")

def download_reels(url):
    """
    인스타그램 릴스 다운로드 (쿠키 인증 + Android 위장)
    """
    save_path = os.path.join(os.getcwd(), OUTPUT_FOLDER)
    
    ydl_opts = {
        'format': 'best',
        'outtmpl': f'{save_path}/%(uploader)s_%(id)s.%(ext)s',
        
        # 🚨 [핵심 Fix] 인스타그램 로그인 쿠키 삽입
        'cookiefile': COOKIE_FILE_PATH,
        
        # 차단 우회를 위한 모바일 에이전트
        'user_agent': 'Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36',
        
        'ignoreerrors': True,
        'nocheckcertificate': True,
        'download_archive': 'instagram_downloaded.txt',
        
        'http_headers': {
            'Referer': 'https://www.instagram.com/',
            'Origin': 'https://www.instagram.com',
        },
        'source_address': '0.0.0.0', 
    }

    try:
        print(f"\n📸 [Instagram+Cookie] 다운로드 시작: {url}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print(f"✅ 작업 끝\n" + "="*40)
        
    except Exception as e:
        print(f"\n❌ 다운로드 실패: {e}")

if __name__ == "__main__":
    # 1. 쿠키 파일 존재 여부 확인
    if not os.path.exists(COOKIE_FILE_PATH):
        print(f"🚨 [경고] 인스타그램 쿠키 파일이 없습니다: {COOKIE_FILE_PATH}")
        print("👉 브라우저에서 인스타그램 로그인 후 쿠키를 추출해 넣어주세요.")
        exit()

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    if os.path.exists(LIST_FILE_PATH):
        with open(LIST_FILE_PATH, "r", encoding="utf-8") as f:
            urls = f.readlines()
            
        for url in urls:
            if url.strip():
                download_reels(url.strip())
                
        print(f"\n🎉 인스타그램 작업 완료")
    else:
        print(f"❌ 목록 파일을 찾을 수 없습니다: {LIST_FILE_PATH}")