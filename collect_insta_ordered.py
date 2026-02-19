import yt_dlp
import os

# ==========================================
# ⚙️ 설정
# ==========================================
SAVE_DIR = r"C:\Users\leejy\Desktop\test_experiment\dataset\downloaded_from_insta"
LINKS_FILE = "insta_links.txt"
COOKIE_FILE = "cookies.txt"  # 👈 추출한 쿠키 파일명

def download_ordered_insta():
    if not os.path.exists(LINKS_FILE):
        print(f"❌ 에러: {LINKS_FILE} 파일이 없습니다.")
        return
    
    if not os.path.exists(COOKIE_FILE):
        print(f"❌ 에러: {COOKIE_FILE} 파일이 폴더에 없습니다. 쿠키를 먼저 추출하세요.")
        return

    with open(LINKS_FILE, "r") as f:
        urls = [line.strip() for line in f.readlines() if line.strip()]

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    print(f"🚀 총 {len(urls)}개의 영상을 cookies.txt를 사용하여 순서대로 수거합니다.")

    for i, url in enumerate(urls, start=1):
        file_name = f"S{i:02d}_IG"
        
        ydl_opts = {
            'format': 'best',
            'outtmpl': os.path.join(SAVE_DIR, f"{file_name}.%(ext)s"),
            'cookiefile': COOKIE_FILE,  # 👈 브라우저 직접 접근 대신 파일 사용
            'no_warnings': True,
            'ignoreerrors': True,
            # 인스타 차단 방지를 위한 유저 에이전트 설정
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }

        print(f"📥 [{i:02d}/30] 다운로드 중: {file_name} <- {url}")
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
        except Exception as e:
            print(f"   ⚠️ {file_name} 처리 중 오류 발생: {e}")

if __name__ == "__main__":
    download_ordered_insta()