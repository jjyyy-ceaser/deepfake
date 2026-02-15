import os
import cv2
from pytubefix import YouTube

# ==========================================
# ⚙️ 설정 (젠슨 황 키노트 영상 적용!)
# ==========================================
BASE_DIR = "C:/Users/leejy/Desktop/test_experiment/dataset/2_generalization"

TARGET_URLS = {
    # 1. Runway (이미 받으셨으면 주석 # 유지)
    # "fake_runway": "https://www.youtube.com/watch?v=OHZKI50uHr8",  
    
    # 2. Pika (이미 받으셨으면 주석 # 유지)
    # "fake_pika": "https://www.youtube.com/watch?v=xSLyQdsBdZY",    

    # 3. Real (NVIDIA Keynote - 아주 훌륭한 Real 데이터)
    "real_ffpp": "https://www.youtube.com/watch?v=lQHK61IDFH4"     
}

CLIP_DURATION = 4   # 4초
MAX_CLIPS = 30      # 30개

# ==========================================
# 🚀 다운로드 및 자르기 로직
# ==========================================
def process_one_video(folder_name, url):
    save_dir = os.path.join(BASE_DIR, folder_name)
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n🚀 [{folder_name}] 다운로드 시작: {url}")
    
    temp_filename = f"temp_{folder_name}.mp4"
    
    try:
        yt = YouTube(url)
        # 화질 좋은 mp4 찾기
        stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
        if not stream:
            stream = yt.streams.filter(file_extension='mp4').order_by('resolution').desc().first()
            
        stream.download(filename=temp_filename)
        print("   ✅ 다운로드 성공! 자르기 진입...")
        
    except Exception as e:
        print(f"   ❌ 다운로드 에러: {e}")
        return

    # 자르기
    cap = cv2.VideoCapture(temp_filename)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30

    frame_interval = int(fps * CLIP_DURATION)
    saved_count = 0
    current_frame = 0 # 0초부터 시작
    
    # 젠슨 황 영상은 앞부분 인트로가 좀 있으니, 5분(9000프레임) 뒤부터 자르도록 스킵 가능
    # (필요하면 아래 줄 주석 해제하세요. 지금은 0초부터도 괜찮습니다.)
    # current_frame = 30 * 60 * 5 

    clip_idx = 0

    while cap.isOpened() and saved_count < MAX_CLIPS:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret: break
        
        clip_name = f"{folder_name}_{clip_idx:03d}.mp4"
        clip_path = os.path.join(save_dir, clip_name)
        
        height, width, _ = frame.shape
        out = cv2.VideoWriter(clip_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        for _ in range(frame_interval):
            ret, frame = cap.read()
            if not ret: break
            out.write(frame)
        out.release()
        
        # 파일이 정상적으로 생성됐는지 확인
        if os.path.exists(clip_path) and os.path.getsize(clip_path) > 1000:
            print(f"      👉 저장됨: {clip_name}")
            saved_count += 1
        
        clip_idx += 1
        current_frame += frame_interval

    cap.release()
    if os.path.exists(temp_filename):
        os.remove(temp_filename)
    
    print(f"   🎉 {folder_name} 완료! 총 {saved_count}개 생성.")

if __name__ == "__main__":
    for name, link in TARGET_URLS.items():
        process_one_video(name, link)
    print("\n🏁 모든 데이터셋 준비 완료!")