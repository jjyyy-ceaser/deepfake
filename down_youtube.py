import yt_dlp
import os

# ======================================================
# ⚙️ [설정] 파일 경로
# ======================================================
LIST_FILE_PATH = r"C:\Users\leejy\Desktop\test_experiment\youtube_list.txt"
OUTPUT_FOLDER = "downloads"

def download_playlist(playlist_url):
    save_path = os.path.join(os.getcwd(), OUTPUT_FOLDER, "%(playlist_title)s")
    
    ydl_opts = {
        # 1. 화질 설정
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': f'{save_path}/%(title)s.%(ext)s',
        
        # 2. [핵심] 안드로이드 앱으로 위장 (쿠키 없이 실행)
        # 핫스팟(새 IP)에서는 이 방식이 차단될 확률이 가장 낮습니다.
        'extractor_args': {
            'youtube': {
                'player_client': ['android', 'web'],
                'player_skip': ['web_safari'], 
            }
        },
        
        # 3. 재생목록 설정
        'yes_playlist': True,
        'ignoreerrors': True,
        'download_archive': 'downloaded_list.txt',
        'nocheckcertificate': True,
        
        # 4. IPv4 강제 사용 (핫스팟 연결 시 IPv6 오류 방지)
        'source_address': '0.0.0.0', 
    }

    try:
        print(f"\n📱 [핫스팟+Android 모드] 다운로드 시작: {playlist_url}")
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([playlist_url])
            
        print(f"✅ [완료] 작업 끝\n" + "="*40)
        
    except Exception as e:
        print(f"\n❌ [오류] 다운로드 실패: {e}")

if __name__ == "__main__":
    if os.path.exists(LIST_FILE_PATH):
        print(f"📄 목록 파일 로드됨: {LIST_FILE_PATH}")
        with open(LIST_FILE_PATH, "r", encoding="utf-8") as f:
            urls = f.readlines()
        for url in urls:
            if url.strip():
                download_playlist(url.strip())
    else:
        print(f"❌ 목록 파일을 찾을 수 없습니다: {LIST_FILE_PATH}")