import yt_dlp
import os

# 다운로드할 링크 리스트
urls = [
    "https://youtube.com/playlist?list=PLQ0U6BphPpo33IuWgO_OA3OO2w4tdHPvx&si=7asnYWscRH5LLNZA", # 재생목록 1
    "https://youtube.com/playlist?list=PL-hsWDEPUtJyamb_zUJjRk5gR2LJKDYSF&si=tSzsdLQsKx0f_pWl", # 재생목록 2
    "https://youtube.com/playlist?list=PLcdwer1B0deLuSqSn44KiWAWEbGRnvkjA&si=G2-hVvoYTYTTUkqF", # 재생목록 3
    "https://youtu.be/AjDycYB2g4M" # 단일 영상
]

# 저장할 폴더 생성
save_path = "downloaded_videos"
if not os.path.exists(save_path):
    os.makedirs(save_path)

def download_videos(url_list):
    ydl_opts = {
        # 연구용 최고 화질 설정 (영상+오디오 병합)
        'format': 'bestvideo+bestaudio/best',
        
        # 파일 저장 경로 및 이름 규칙: 폴더/제목_영상ID.확장자
        # (연구 데이터 관리를 위해 영상 ID를 포함하는 것을 추천합니다)
        'outtmpl': f'{save_path}/%(title)s_%(id)s.%(ext)s',
        
        # 재생목록의 모든 영상을 다운로드
        'yes_playlist': True,
        
        # 메타데이터 무시 및 에러 발생 시 건너뛰기 설정
        'ignoreerrors': True,
        
        # (선택) VP9 코덱을 선호한다면 아래 주석 해제 (단, 없으면 best로 다운됨)
        # 'format': 'bestvideo[vcodec^=vp9]+bestaudio/best',
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for url in url_list:
            print(f"📥 다운로드 시작: {url}")
            try:
                # 메타데이터 추출 (재생목록인지 확인용)
                info_dict = ydl.extract_info(url, download=True)
                
                # 결과 출력
                if 'entries' in info_dict: # 재생목록인 경우
                    print(f"✅ 재생목록 다운로드 완료: {info_dict.get('title', 'Unknown Playlist')}")
                else: # 단일 영상인 경우
                    print(f"✅ 영상 다운로드 완료: {info_dict.get('title', 'Unknown Video')}")
                    
            except Exception as e:
                print(f"❌ 오류 발생: {url} - {e}")

if __name__ == "__main__":
    download_videos(urls)