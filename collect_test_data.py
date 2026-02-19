import yt_dlp
import os

# 저장 경로
BASE_DIR = r"C:\Users\leejy\Desktop\test_experiment\dataset\downloaded_from_youtube"

# 다운로드할 전체 링크 (재생목록 3개 + 단일 영상 1개)
TARGET_URLS = [
    "https://youtube.com/playlist?list=PLQ0U6BphPpo33IuWgO_OA3OO2w4tdHPvx&si=7asnYWscRH5LLNZA",
    "https://youtube.com/playlist?list=PL-hsWDEPUtJyamb_zUJjRk5gR2LJKDYSF&si=tSzsdLQsKx0f_pWl",
    "https://youtube.com/playlist?list=PLcdwer1B0deLuSqSn44KiWAWEbGRnvkjA&si=G2-hVvoYTYTTUkqF",
    "https://youtu.be/AjDycYB2g4M"
]

def download_videos():
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)

    # 쿠키 옵션 추가된 설정
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',
        'outtmpl': os.path.join(BASE_DIR, '%(title)s_%(id)s.%(ext)s'),
        'yes_playlist': True,
        'ignoreerrors': True,
        'no_warnings': True,
        
        # 🔥 핵심: 크롬 브라우저의 쿠키를 사용하여 로그인된 상태로 다운로드
        # (엣지 사용자는 'chrome'을 'edge'로 바꾸세요)
        'cookiesfrombrowser': ('chrome',), 
    }

    print(f"🚀 총 {len(TARGET_URLS)}개의 링크 다운로드를 시작합니다 (쿠키 사용)...")

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for i, url in enumerate(TARGET_URLS):
            print(f"\n📥 [{i+1}/{len(TARGET_URLS)}] 처리 중...")
            try:
                ydl.download([url])
                print("   ✅ 완료!")
            except Exception as e:
                print(f"   ❌ 실패: {e}")

if __name__ == "__main__":
    download_videos()