import os
import glob

BASE_DIR = r"C:\Users\leejy\Desktop\test_experiment\dataset\final_dataset\raw\train"

def debug_identities():
    rf = glob.glob(os.path.join(BASE_DIR, "real", "*"))
    ff = glob.glob(os.path.join(BASE_DIR, "fake", "*"))
    
    real_ids = []
    for f in rf:
        # 공백이나 대소문자 문제를 확인하기 위해 strip()과 lower() 추가 권장
        identity = os.path.basename(f).split("--")[0].replace("real_", "").strip()
        real_ids.append(identity)
        
    fake_ids = []
    for f in ff:
        identity = os.path.basename(f).split("--")[0].strip()
        fake_ids.append(identity)
        
    print(f"📊 Real IDs (first 5): {real_ids[:5]}")
    print(f"📊 Fake IDs (first 5): {fake_ids[:5]}")
    
    intersection = set(real_ids).intersection(set(fake_ids))
    print(f"✅ 중복된 ID 수: {len(intersection)}")
    print(f"❌ 중복되지 않은 ID 수: {len(set(real_ids + fake_ids)) - len(intersection)}")

if __name__ == "__main__":
    debug_identities()