import os
import shutil
import random
from glob import glob
from tqdm import tqdm

BASE_DIR = "dataset"
PROCESSED_TRAIN = os.path.join(BASE_DIR, "processed_cases", "train")
FINAL_DIR = os.path.join(BASE_DIR, "final_datasets")

def clear_and_copy(src_files, dst_folder, prefix=""):
    os.makedirs(dst_folder, exist_ok=True)
    for f in src_files:
        basename = os.path.basename(f)
        shutil.copy2(f, os.path.join(dst_folder, f"{prefix}{basename}"))

def build_dataset_b_exclusive():
    print("🔨 Building DataSet B (Mixed - Mutually Exclusive)...")
    # 원본(Case 1)에서 파일 리스트 확보
    case1_root = os.path.join(PROCESSED_TRAIN, "case1_original")
    
    for label in ["real", "fake"]:
        # 파일 ID 리스트 가져오기
        src_path = os.path.join(case1_root, label)
        files = sorted(os.listdir(src_path))
        random.shuffle(files) # 랜덤 셔플
        
        # 4등분 (25%씩)
        chunk_size = len(files) // 4
        chunks = [files[i:i + chunk_size] for i in range(0, len(files), chunk_size)]
        # 남는 자투리는 마지막 청크에 병합
        if len(chunks) > 4: chunks[3].extend(chunks[4]); del chunks[4]

        # 각 청크를 서로 다른 Case에서 가져오기
        cases = ["case1_original", "case2_lowres", "case3_compress", "case4_mixed"]
        
        dst_path = os.path.join(FINAL_DIR, "dataset_B_mixed", label)
        os.makedirs(dst_path, exist_ok=True)

        for i, case_name in enumerate(cases):
            # i번째 청크는 i번째 Case 폴더에서 가져옴 -> 중복 절대 없음
            for fname in chunks[i]:
                src_file = os.path.join(PROCESSED_TRAIN, case_name, label, fname)
                # 파일명에 출처 Case 표기 (디버깅용)
                dst_file = os.path.join(dst_path, f"{case_name}_{fname}")
                shutil.copy2(src_file, dst_file)

def main():
    random.seed(42) # 재현성 확보
    
    # 1. DataSet A (Pure: Case 1 100%)
    print("🔨 Building DataSet A (Pure)...")
    for label in ["real", "fake"]:
        src = glob(os.path.join(PROCESSED_TRAIN, "case1_original", label, "*"))
        clear_and_copy(src, os.path.join(FINAL_DIR, "dataset_A_pure", label))

    # 2. DataSet C (Worst: Case 4 100%)
    print("🔨 Building DataSet C (Worst)...")
    for label in ["real", "fake"]:
        src = glob(os.path.join(PROCESSED_TRAIN, "case4_mixed", label, "*"))
        clear_and_copy(src, os.path.join(FINAL_DIR, "dataset_C_worst", label))

    # 3. DataSet B (Mixed: Exclusive 25%)
    build_dataset_b_exclusive()
    
    print("✨ 모든 데이터셋 구축 완료!")

if __name__ == "__main__":
    main()