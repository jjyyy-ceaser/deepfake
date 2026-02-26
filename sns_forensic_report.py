import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
import os, gc, timm
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from transformers import VideoMAEForVideoClassification
from pathlib import Path

# 📌 체크 리스트 1: 데이터 로더 규격 준수
from data_loader import UnifiedDataset

# ==========================================
# ⚙️ 설정 및 환경 구축 (🔥 GPU 풀악셀 최적화 적용)
# ==========================================
TEST_DIR = Path(r"C:\Users\leejy\Desktop\test_experiment\dataset\test")
WEIGHTS_DIR = Path("./weights")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🚀 Inference 전용 Batch Size 및 Worker 튜닝 (학습 코드 세팅 이식)
SPATIAL_BATCH_SIZE = 64
TEMPORAL_BATCH_SIZE = 8
NUM_WORKERS = 4

FILTER_KEYWORD = "pure"

CASES = ["case1_original", "case4_mixed"]
# 유저가 폴더명을 직접 kakao_normal로 수정했으므로 그대로 유지
PLATFORMS = ["original", "youtube", "instagram", "kakao_normal", "kakao_high"]
GENERATORS = ["pika", "runway", "svd"] 

SPATIAL_MODELS = ["xception", "convnext", "swin"]
TEMPORAL_MODELS = ["r3d", "r2plus1d", "videomae"]

def clean_memory():
    """📌 체크 리스트 3: 엄격한 메모리 정리"""
    gc.collect()
    torch.cuda.empty_cache()

def get_model_and_transform(model_name):
    """📌 체크 리스트 4 & 5: 정규화 및 아키텍처 매칭"""
    if model_name in SPATIAL_MODELS:
        tf = transforms.Compose([
            transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        if model_name == "xception": model = timm.create_model('xception', num_classes=2)
        elif model_name == "convnext": model = timm.create_model('convnext_tiny', num_classes=2)
        elif model_name == "swin": model = timm.create_model('swin_tiny_patch4_window7_224', num_classes=2)
        return model, tf, 'spatial'
    else:
        if model_name == "videomae":
            tf = transforms.Compose([
                transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base", num_labels=2, ignore_mismatched_sizes=True)
            return model, tf, 'videomae'
        else:
            tf = transforms.Compose([
                transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor(),
                transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
            ])
            if model_name == "r3d": model = models.video.r3d_18(num_classes=2)
            else: model = models.video.r2plus1d_18(num_classes=2)
            return model, tf, 'temporal'

def main():
    clean_memory()
    all_weights = [f for f in os.listdir(WEIGHTS_DIR) if f.endswith('.pth') and FILTER_KEYWORD in f.lower()]
    model_list = [m for m in SPATIAL_MODELS + TEMPORAL_MODELS if any(m in w.lower() for w in all_weights)]

    raw_probs_list = []
    
    # 테스트 데이터 경로 수집
    all_samples = []
    for case in CASES:
        for platform in PLATFORMS:
            case_path = TEST_DIR / case
            r_path = case_path / "real" / platform
            if r_path.exists():
                all_samples += [{"path": str(v), "case": case, "platform": platform, "gen": "real", "label": 0} for v in r_path.glob("*.mp4")]
            for gen in GENERATORS:
                f_path = case_path / "fake" / gen / platform
                if f_path.exists():
                    all_samples += [{"path": str(v), "case": case, "platform": platform, "gen": gen, "label": 1} for v in f_path.glob("*.mp4")]
    if not all_samples: return

    # --- 추론 스캐닝 ---
    for m_name in model_list:
        print(f"\n🚀 모델 분석: {m_name.upper()}")
        model_arch, transform, m_type = get_model_and_transform(m_name)
        weights = sorted([w for w in all_weights if m_name in w.lower()])
        
        ds = UnifiedDataset([(s['path'], s['label']) for s in all_samples], model_type=m_type, transform=transform)
        
        # 🔥 [속도 최적화] 학습 코드(train_*.py)의 완벽한 튜닝값 이식
        current_batch = SPATIAL_BATCH_SIZE if m_type == 'spatial' else TEMPORAL_BATCH_SIZE
        loader = DataLoader(
            ds, 
            batch_size=current_batch, 
            shuffle=False, 
            num_workers=NUM_WORKERS,
            pin_memory=True,            # GPU VRAM으로의 다이렉트 고속도로 개통
            persistent_workers=True     # 폴드가 바뀌어도 CPU 워커를 죽이지 않고 재활용
        )

        for w_file in weights:
            print(f"  └─ {w_file}")
            model = model_arch
            model.load_state_dict(torch.load(WEIGHTS_DIR / w_file))
            model = model.to(DEVICE).eval()
            
            fold_probs = []
            with torch.no_grad():
                for x, y in tqdm(loader, desc="      Scanning", leave=False):
                    x = x.to(DEVICE)
                    with torch.amp.autocast('cuda'):
                        if m_name == "videomae": out = model(pixel_values=x).logits
                        else: out = model(x)
                        p = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                        fold_probs.extend(p)
            
            f_id = w_file.split('fold')[-1].split('.')[0]
            col_name = f"{m_name}_fold{f_id}"
            for i, prob in enumerate(fold_probs):
                if len(raw_probs_list) <= i:
                    raw_probs_list.append({**all_samples[i], col_name: prob})
                else:
                    raw_probs_list[i][col_name] = prob
            del model
            clean_memory()

    # --- 데이터 가공 및 리포트화 ---
    raw_df = pd.DataFrame(raw_probs_list)
    raw_df.to_csv("Inference_Raw_Backup.csv", index=False)

    # 📌 그룹 앙상블 (Total 제외, Spatial과 Temporal만 유지)
    ensemble_groups = {"Spatial_Ensemble": SPATIAL_MODELS, "Temporal_Ensemble": TEMPORAL_MODELS}
    for group_name, m_group in ensemble_groups.items():
        for f_idx in range(1, 6):
            cols = [f"{m}_fold{f_idx}" for m in m_group if f"{m}_fold{f_idx}" in raw_df.columns]
            if cols: raw_df[f"{group_name}_fold{f_idx}"] = raw_df[cols].mean(axis=1)

    all_targets = model_list + list(ensemble_groups.keys())
    all_flat_metrics = []

    print("\n📊 7대 체크리스트 기반 마스터 리포트 생성 중...")
    with pd.ExcelWriter("SNS_Forensic_Detailed_Report.xlsx", engine="openpyxl") as writer:
        for target in all_targets:
            fold_cols = sorted([c for c in raw_df.columns if c.startswith(f"{target}_fold")])
            if not fold_cols: continue
            
            # 해당 타겟(모델)의 앙상블 확률(Soft Voting) 계산
            raw_df[f"{target}_SV"] = raw_df[fold_cols].mean(axis=1)

            target_metrics = []
            for case in CASES:
                for platform in PLATFORMS:
                    for gen in GENERATORS:
                        subset = raw_df[(raw_df['case'] == case) & (raw_df['platform'] == platform) & ((raw_df['gen'] == gen) | (raw_df['label'] == 0))]
                        if subset.empty or len(subset['label'].unique()) < 2: continue 
                        
                        y_true = subset['label'].values
                        
                        def calc_metrics(y_prob):
                            y_pred = (y_prob > 0.5).astype(int)
                            try: 
                                auc = roc_auc_score(y_true, y_prob)
                            except ValueError: 
                                auc = np.nan
                            return {
                                "AUC": auc, "Accuracy": accuracy_score(y_true, y_pred),
                                "F1-Score": f1_score(y_true, y_pred, zero_division=0),
                                "Precision": precision_score(y_true, y_pred, zero_division=0),
                                "Recall": recall_score(y_true, y_pred, zero_division=0)
                            }

                        fold_results = {metric: [] for metric in ["AUC", "Accuracy", "F1-Score", "Precision", "Recall"]}
                        for f_col in fold_cols:
                            m_res = calc_metrics(subset[f_col].values)
                            f_num = f_col.split('fold')[-1]
                            
                            row_data = {"Model": target, "Case": case, "Platform": platform, "Generator": gen, "Eval_Type": f"Fold_{f_num}"}
                            row_data.update({k: (f"{v:.4f}" if not np.isnan(v) else "NaN") for k, v in m_res.items()})
                            target_metrics.append(row_data)
                            all_flat_metrics.append(row_data)

                            for k in fold_results.keys(): fold_results[k].append(m_res[k])

                        sv_res = calc_metrics(subset[f"{target}_SV"].values)
                        sv_row = {"Model": target, "Case": case, "Platform": platform, "Generator": gen, "Eval_Type": "Model_Ensemble (Soft Voting)"}
                        sv_row.update({k: (f"{v:.4f}" if not np.isnan(v) else "NaN") for k, v in sv_res.items()})
                        target_metrics.append(sv_row)
                        all_flat_metrics.append(sv_row)

                        stat_row = {"Model": target, "Case": case, "Platform": platform, "Generator": gen, "Eval_Type": "Folds_Mean ± SD"}
                        stat_row.update({k: f"{np.nanmean(fold_results[k]):.4f} ± {np.nanstd(fold_results[k], ddof=1):.4f}" for k in fold_results.keys()})
                        target_metrics.append(stat_row)
                        all_flat_metrics.append(stat_row)

            if target_metrics:
                m_df = pd.DataFrame(target_metrics)
                m_df['Platform_Gen'] = m_df['Platform'] + "_" + m_df['Generator']
                pivot = m_df.melt(id_vars=["Model", "Case", "Platform_Gen", "Eval_Type"], var_name="Metric", value_name="Score") \
                            .pivot_table(index=["Case", "Eval_Type", "Metric"], columns="Platform_Gen", values="Score", aggfunc='first')
                pivot.to_excel(writer, sheet_name=target[:31])

    pd.DataFrame(all_flat_metrics).to_csv("Experimental_Results_Detailed_Flat.csv", index=False, encoding='utf-8-sig')
    print("✅ 완료: 'Experimental_Results_Detailed_Flat.csv' 및 'SNS_Forensic_Detailed_Report.xlsx'")

if __name__ == "__main__":
    main()