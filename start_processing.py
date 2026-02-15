import os
import subprocess
import shutil
from tqdm import tqdm

# ==========================================
# ⚙️ 1. 경로 설정 (0번, 1번, 2번 폴더 구조 반영)
# ==========================================
BASE_DIR = r"C:\Users\leejy\Desktop\test_experiment\dataset"
PURE_ROOT = os.path.join(BASE_DIR, "2_exp_train_pure") # 분할 기준점
SRC_1 = os.path.join(BASE_DIR, "1_generalization")     # Runway, Pika, FFPP

# 가공 설정
RES = "scale=-2:360" # 360p
CRF_VAL = "40"       # 고압축

def apply_quality(src, dst, mode):
    """FFmpeg를 사용하여 지정된 화질(Case 1~4)로 가공"""
    if mode == 'case1': # 원본 유지
        shutil.copy2(src, dst)
        return
    
    # 터미널에서 ffmpeg가 확인되었으므로 "ffmpeg" 명령어를 직접 사용합니다.
    cmd = ["ffmpeg", "-y", "-i", src]
    
    if mode == 'case2': # 저해상도만
        cmd += ["-vf", RES, "-crf", "23"] 
    elif mode == 'case3': # 고압축만
        cmd += ["-crf", CRF_VAL]
    elif mode == 'case4': # 최악 (저해상도 + 고압축)
        cmd += ["-vf", RES, "-crf", CRF_VAL]
    
    cmd += ["-c:v", "libx264", "-preset", "veryfast", dst]
    
    # 가공 과정을 보고 싶으시면 stderr=None으로 바꾸세요.
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# ==========================================
# 🏋️ 2. 학습용(Worst/Mixed) 가공 (2번/train 소스)
# ==========================================
def make_train_sets():
    print("1️⃣ 학습용 변형 데이터(Worst/Mixed) 생성 시작...")
    train_src = os.path.join(PURE_ROOT, "train")
    
    for var in ["worst", "mixed"]:
        dst_path = os.path.join(BASE_DIR, f"2_train_{var}")
        for cls in ["real", "fake"]:
            os.makedirs(os.path.join(dst_path, cls), exist_ok=True)
            files = sorted(os.listdir(os.path.join(train_src, cls)))
            for i, f in enumerate(tqdm(files, desc=f"{var}-{cls}")):
                s, d = os.path.join(train_src, cls, f), os.path.join(dst_path, cls, f)
                if var == "worst": apply_quality(s, d, 'case4')
                else: # mixed (50% pure, 50% worst)
                    if i % 2 == 0: shutil.copy2(s, d)
                    else: apply_quality(s, d, 'case4')

# ==========================================
# 🧪 3. 테스트 도메인(3~6번) 전수 가공 (2번/test 소스)
# ==========================================
def make_test_sets():
    print("\n2️⃣ 테스트 도메인 1~4(Case 1~4) 전수 가공 시작...")
    test_src = os.path.join(PURE_ROOT, "test")

    # {폴더명: (Real 소스, Fake 소스)}
    TEST_MAP = {
        "3_test_svd":    (os.path.join(test_src, "real"), os.path.join(test_src, "fake")),
        "4_test_runway": (os.path.join(test_src, "real"), os.path.join(SRC_1, "fake_runway")),
        "5_test_pika":   (os.path.join(test_src, "real"), os.path.join(SRC_1, "fake_pika")),
        "6_test_ffpp":   (os.path.join(SRC_1, "real_ffpp"), os.path.join(test_src, "fake"))
    }

    for folder, (r_src, f_src) in TEST_MAP.items():
        for case in ["case1", "case2", "case3", "case4"]:
            for cls, s_dir in [("real", r_src), ("fake", f_src)]:
                dst_dir = os.path.join(BASE_DIR, folder, case, cls)
                os.makedirs(dst_dir, exist_ok=True)
                
                # 상위 33개 영상 전수 가공 (통계적 일관성 확보)
                files = sorted([f for f in os.listdir(s_dir) if f.lower().endswith('.mp4')])[:33]
                for f in tqdm(files, desc=f"{folder}-{case}-{cls}", leave=False):
                    apply_quality(os.path.join(s_dir, f), os.path.join(dst_dir, f), case)

if __name__ == "__main__":
    make_train_sets()
    make_test_sets()
    print("\n✅ 모든 데이터 가공이 완료되었습니다. 2~6번 폴더를 확인하세요!")