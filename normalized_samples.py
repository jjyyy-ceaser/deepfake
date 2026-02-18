import os
import subprocess

input_dir = "./raw_samples"
output_dir = "./normalized_samples"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for filename in os.listdir(input_dir):
    if filename.endswith((".mp4", ".mov", ".mkv")):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"std_{filename}")
        
        # FFmpeg 명령어 실행
        cmd = [
            'ffmpeg', '-i', input_path,
            '-vf', 'scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,fps=30',
            '-t', '10',
            '-c:v', 'libx264', '-crf', '0', '-preset', 'veryslow',
            '-c:a', 'copy', output_path, '-y'
        ]
        
        print(f"Processing: {filename}...")
        subprocess.run(cmd)

print("정규화 완료!")