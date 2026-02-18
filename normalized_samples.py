import subprocess
from tqdm import tqdm
import os

def normalize_videos(input_folder, output_folder):
    files = [f for f in os.listdir(input_folder) if f.endswith('.mp4')]
    
    # 진행 상황을 한눈에 볼 수 있는 tqdm 적용
    for file in tqdm(files, desc="정규화 진행 중"):
        input_path = os.path.join(input_folder, file)
        output_path = os.path.join(output_folder, f"std_{file}")
        
        # CPU 점유율을 고려한 'slower' 프리셋 활용
        command = [
            'ffmpeg', '-i', input_path,
            '-vf', 'scale=1920:1080,fps=30',
            '-t', '10',
            '-c:v', 'libx264', '-crf', '18', '-preset', 'slower',
            '-y', output_path
        ]
        
        # 에러 로그를 확인하며 실행
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

print("준비 완료. 가상환경에서 스크립트를 실행하세요.")