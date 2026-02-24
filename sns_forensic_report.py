import os
import glob
import gc
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import KFold
from tqdm import tqdm


# =====================================================================
# 1. 환경 변수 및 설정
# =====================================================================
BASE_DIR = r"C:\Users\leejy\Desktop\test_experiment\dataset\test"
TRAIN_LIST_PATH = r"C:\Users\leejy\Desktop\test_experiment\dataset\train_list.txt"

MAIN_CASES = ["case1", "case4"]
PLATFORMS = {
    "Raw": "raw", 
    "YouTube": "youtube",
    "Instagram": "instagram",
    "Kakao_Normal": "kakao_normal",
    "Kakao_High": "kakao_high"
}
NUM_FOLDS = 5

# 대상 모델 타입 설정 ("spatial", "temporal", "videomae" 중 택 1)
MODEL_TYPE = "temporal"
BATCH_SIZE = 32 if MODEL_TYPE == "spatial" else 4

# =====================================================================
# 2. 메모리 정리 및 무결성 검증 로직
# =====================================================================
def clean_memory():
    """✨ GPU VRAM 및 시스템 RAM 캐시 강제 반환 ✨"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("    🧹 [메모리 환수 완료] VRAM/RAM 누수 방지 조치 적용됨.")

def validate_no_data_leakage(test_files, train_list_path):
    print("\n" + "="*70)
    print("🛡️ [단계 1] 데이터 오염(Data Leakage) 사전 블랙리스트 검증")
    print("="*70)
    
    if not os.path.exists(train_list_path):
        print("⚠️ 학습 리스트 파일이 존재하지 않아 무결성 검증을 건너뜁니다.")
        return
        
    with open(train_list_path, 'r', encoding='utf-8') as f:
        train_set = set(os.path.splitext(line.strip())[0] for line in f if line.strip())
        
    test_set = set(os.path.splitext(os.path.basename(p))[0] for p in test_files)
    leakage = train_set.intersection(test_set)
    
    if leakage:
        raise ValueError(f"🚨 [치명적 오류] 데이터 오염 감지! 중복: {list(leakage)[:5]} ... 실험 강제 중단.")
    print("✅ 무결성 확인: 학습 데이터와 평가 데이터가 100% 독립적입니다.")

# =====================================================================
# 3. 고속 배치 기반 DataLoader 및 추론 엔진
# =====================================================================
class VideoInferenceDataset(Dataset):
    def __init__(self, file_paths, model_type="spatial"):
        self.file_paths = file_paths
        self.model_type = model_type
        # TODO: 사용자의 train_*.py 파일 내 transform 로직 이식 필요
        
    def __len__(self):
        return len(self.file_paths)
        
    def __getitem__(self, idx):
        path = self.file_paths[idx]
        # 더미 텐서 (실제 CV2/Torchvision 변환 로직으로 교체)
        dummy_tensor = torch.zeros((3, 224, 224)) if self.model_type == "spatial" else torch.zeros((3, 16, 224, 224))
        return dummy_tensor, path

def run_batch_inference(file_paths, model, device="cuda"):
    dataset = VideoInferenceDataset(file_paths, model_type=MODEL_TYPE)
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=8,        # 병목 해소 핵심 1
        pin_memory=True,      # 병목 해소 핵심 2
        prefetch_factor=2,    # 병목 해소 핵심 3
        persistent_workers=True
    )
    
    predictions, confidences = [], []
    model.eval()
    
    with torch.no_grad():
        for inputs, paths in tqdm(loader, desc="추론 진행 중", leave=False):
            inputs = inputs.to(device)
            # TODO: 실제 추론 로직 (예: outputs = model(inputs); probs = torch.sigmoid(outputs))
            
            # 아래는 로직 중단 방지용 더미 데이터 생성기
            batch_preds = [1 if "fake" in p.lower() else 0 for p in paths]
            batch_confs = [0.85 if "fake" in p.lower() else 0.15 for p in paths]
            
            predictions.extend(batch_preds)
            confidences.extend(batch_confs)
            
    del loader, dataset  # 즉각적인 참조 해제
    return predictions, confidences

# =====================================================================
# 4. 단일 케이스-플랫폼 평가 및 DataFrame 적재
# =====================================================================
def evaluate_condition(main_case, platform_key, platform_folder, model, device="cuda"):
    target_path = os.path.join(BASE_DIR, main_case, platform_folder)
    real_files = sorted(glob.glob(os.path.join(target_path, "real", "*.*")))
    fake_files = sorted(glob.glob(os.path.join(target_path, "fake", "*.*")))
    
    if len(real_files) != len(fake_files) or len(real_files) == 0:
        raise ValueError(f"[{main_case} - {platform_key}] 데이터 1:1 쌍 불일치 혹은 폴더 비어있음.")
        
    print(f"\n▶ [{main_case} - {platform_key}] 데이터 적재 및 추론 시작...")
    
    r_preds, r_confs = run_batch_inference(real_files, model, device)
    f_preds, f_confs = run_batch_inference(fake_files, model, device)
    
    results = []
    for i in range(len(real_files)):
        results.append({"pair_id": i, "filename": os.path.basename(real_files[i]), "true_label": 0, "pred_label": r_preds[i], "confidence": r_confs[i]})
        results.append({"pair_id": i, "filename": os.path.basename(fake_files[i]), "true_label": 1, "pred_label": f_preds[i], "confidence": f_confs[i]})
        
    return pd.DataFrame(results)

def calculate_metrics(df):
    y_true, y_pred, y_prob = df["true_label"], df["pred_label"], df["confidence"]
    return {
        "Acc": accuracy_score(y_true, y_pred),
        "Pre": precision_score(y_true, y_pred, zero_division=0),
        "Rec": recall_score(y_true, y_pred, zero_division=0),
        "F1":  f1_score(y_true, y_pred, zero_division=0),
        "AUC": roc_auc_score(y_true, y_prob)
    }

# =====================================================================
# 5. 엄격한 변인 통제 기반 2D 다중 K-Fold 교차 분석
# =====================================================================
def run_matrix_kfold_analysis(case_data_dict):
    print("\n" + "="*80)
    print("📊 [단계 2] K-Fold (5-Fold) 변인 통제 매트릭스 분석 시작")
    print("="*80)
    
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    pair_ids = case_data_dict["case1"]["Raw"]['pair_id'].unique()
    
    fold_results = {mc: {pf: {m: [] for m in ["Acc", "Pre", "Rec", "F1", "AUC"]} for pf in PLATFORMS.keys()} for mc in MAIN_CASES}
    
    for fold, (_, test_idx) in enumerate(kf.split(pair_ids), 1):
        test_pairs = pair_ids[test_idx]
        
        for mc in MAIN_CASES:
            for pf in PLATFORMS.keys():
                df_target = case_data_dict[mc][pf]
                fold_data = df_target[df_target['pair_id'].isin(test_pairs)]
                
                # [변인 통제 필수 체크] 쌍(Pair) 데이터 누락 및 섞임 방지
                assert len(fold_data) == len(test_pairs) * 2, f"Fold {fold}: {mc}-{pf} 1:1 매칭 오류 발생!"
                
                res = calculate_metrics(fold_data)
                for m in res.keys():
                    fold_results[mc][pf][m].append(res[m])

    metrics_list = ["Acc", "AUC", "F1", "Pre", "Rec"]
    for m in metrics_list:
        print(f"\n[{m} 지표 (Mean ± Std)]")
        print(f"{'Platform':<15} | {'Case 1 (Baseline)':<20} | {'Case 4 (Extreme)':<20}")
        print("-" * 65)
        for pf in PLATFORMS.keys():
            c1_mean, c1_std = np.mean(fold_results["case1"][pf][m]), np.std(fold_results["case1"][pf][m])
            c4_mean, c4_std = np.mean(fold_results["case4"][pf][m]), np.std(fold_results["case4"][pf][m])
            print(f"{pf:<15} | {c1_mean:.4f} (±{c1_std:.4f}) | {c4_mean:.4f} (±{c4_std:.4f})")

# =====================================================================
# 6. 메인 실행 트리거
# =====================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    target_models = ["r3d_temporal_model", "swin_spatial_model"] # 예시
    
    # 1. 평가 시작 전 최초 1회 전체 데이터 무결성 검증 (Raw 기준)
    test_target_files = glob.glob(os.path.join(BASE_DIR, "case1", "raw", "real", "*.*")) + \
                        glob.glob(os.path.join(BASE_DIR, "case1", "raw", "fake", "*.*"))
    validate_no_data_leakage(test_target_files, TRAIN_LIST_PATH)
    
    for model_name in target_models:
        print(f"\n\n{'#'*80}")
        print(f"🚀 [타겟 모델 추론 시작] {model_name}")
        print(f"{'#'*80}")
        
        # TODO: 실제 가중치 로드 코드 삽입 위치
        dummy_model_instance = torch.nn.Linear(10, 2).to(device) 
        all_results = {mc: {} for mc in MAIN_CASES}
        
        for mc in MAIN_CASES:
            for pf_key, pf_folder in PLATFORMS.items():
                # 개별 조건 추론
                all_results[mc][pf_key] = evaluate_condition(mc, pf_key, pf_folder, dummy_model_instance, device)
                # 평가 후 메모리 즉각 환수
                clean_memory()
                
        # 종합 매트릭스 도출
        run_matrix_kfold_analysis(all_results)
        
        # 모델 완전 폐기 및 VRAM 반환 (다음 모델 평가 준비)
        del dummy_model_instance
        clean_memory()
        print(f"🏁 [{model_name}] 모델 평가 완전 종료.")