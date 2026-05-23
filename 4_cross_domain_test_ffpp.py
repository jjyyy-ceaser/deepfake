import os
import warnings
import gc
from pathlib import Path

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

# ============================================================
# 기본 설정
# ============================================================
warnings.filterwarnings("ignore")

os.environ["HF_HOME"] = r"C:\hf_cache"
os.environ["TORCH_HOME"] = r"C:\torch_cache"

from utils import get_model, calculate_metrics_at_best_threshold
from data_loader import get_dataloader


# ============================================================
# FF++ Face2Face c23 external dataset evaluation
#
# 목적:
#   기존 SVD 데이터셋으로 학습한 5-fold 모델 weight를 그대로 사용하여
#   FaceForensics++ Face2Face c23 외부 데이터셋에서 보조 일반화 평가 수행
#
# 기존 코드 반영 기준:
#   - 공간 모델 2개: Xception, Swin-Tiny
#   - 시간 모델 2개: R3D-18, VideoMAE
#   - Hybrid 제외
#   - 기존 4_cross_domain_test.py 기준 batch_size=16 유지
#   - 기존 최적 프레임 수 유지:
#       Xception / Swin-Tiny = 1 frame
#       R3D-18 = 12 frames
#       VideoMAE = 16 frames
#   - data_loader.py 내부 num_workers=0, pin_memory=True 유지
#
# FF++ 때문에 바뀐 부분:
#   - SVD용 prepare_dataset() 미사용
#   - FF++ fake 파일명 008_990.mp4 -> target_id=008 기준으로 real과 매칭
#   - 결과 파일명을 ffpp_external_*.csv로 분리 저장
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_ROOT = Path(r"C:\Users\leejy\Desktop\test_experiment\dataset\final_dataset_v2")
WEIGHT_DIR = Path(r"C:\Users\leejy\Desktop\test_experiment\results\final_weights")
SAVE_DIR = Path(r"C:\Users\leejy\Desktop\test_experiment\results")

SAVE_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "xception": "xception",
    "swin_tiny_patch4_window7_224": "swin",
    "r3d_18": "r3d",
    "videomae_base": "videomae",
}

DOMAINS = {
    "FFPP_Raw": DATASET_ROOT / "raw_ffpp",
    "FFPP_Instagram": DATASET_ROOT / "instagram_ffpp",
    "FFPP_YouTube": DATASET_ROOT / "youtube_ffpp",
    "FFPP_Kakao_High": DATASET_ROOT / "kakao_high_ffpp",
    "FFPP_Kakao_Normal": DATASET_ROOT / "kakao_normal_ffpp",
}

TEST_BATCH_SIZE = 16


def clean_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_frame_count(model_name: str) -> int:
    name = model_name.lower()

    if "r3d" in name:
        return 12

    if "videomae" in name:
        return 16

    return 1


def get_fake_target_id(fake_path: Path) -> str:
    """
    FF++ Face2Face 파일명 기준:
      008_990.mp4 -> 008
      183_253.mp4 -> 183
      ffpp_008.mp4 -> 008
    """
    stem = fake_path.stem

    if stem.startswith("ffpp_"):
        return stem.replace("ffpp_", "", 1)

    if "_" in stem:
        return stem.split("_")[0]

    return stem


def validate_weight_files():
    print("\n============================================================")
    print("[가중치 파일 확인]")
    print("============================================================")

    missing = []

    for model_name, weight_prefix in MODELS.items():
        for fold in range(1, 6):
            weight_path = WEIGHT_DIR / f"{weight_prefix}_f{fold}.pth"

            if weight_path.exists():
                print(f"[OK] {weight_path.name}")
            else:
                print(f"[MISSING] {weight_path.name}")
                missing.append(str(weight_path))

    if missing:
        print("\n[WARN] 누락된 가중치가 있습니다. 해당 fold는 평가에서 제외됩니다.")
    else:
        print("\n모든 가중치 파일이 존재합니다.")


def prepare_ffpp_dataset(domain_path: Path):
    real_dir = domain_path / "real"
    fake_dir = domain_path / "fake"

    if not real_dir.exists():
        print(f"   [SKIP] real 폴더 없음: {real_dir}")
        return [], [], []

    if not fake_dir.exists():
        print(f"   [SKIP] fake 폴더 없음: {fake_dir}")
        return [], [], []

    real_files_all = sorted(real_dir.glob("*.mp4"))
    fake_files_all = sorted(fake_dir.glob("*.mp4"))

    real_by_id = {p.stem: p for p in real_files_all}

    matched_real = []
    matched_fake = []
    matched_ids = []
    unmatched_fake = []

    for fake_path in fake_files_all:
        target_id = get_fake_target_id(fake_path)

        if target_id in real_by_id:
            matched_real.append(real_by_id[target_id])
            matched_fake.append(fake_path)
            matched_ids.append(target_id)
        else:
            unmatched_fake.append((fake_path.name, target_id))

    if unmatched_fake:
        print(f"   [WARN] 매칭되지 않은 fake 수: {len(unmatched_fake)}")
        for fname, tid in unmatched_fake[:10]:
            print(f"      fake={fname}, target_id={tid}")

    files = [str(p) for p in matched_real] + [str(p) for p in matched_fake]
    labels = [0] * len(matched_real) + [1] * len(matched_fake)

    sample_infos = []

    for target_id, real_path in zip(matched_ids, matched_real):
        sample_infos.append({
            "target_id": target_id,
            "label_name": "real",
            "file_name": real_path.name,
            "file_path": str(real_path),
        })

    for target_id, fake_path in zip(matched_ids, matched_fake):
        sample_infos.append({
            "target_id": target_id,
            "label_name": "fake",
            "file_name": fake_path.name,
            "file_path": str(fake_path),
        })

    print(
        f"   데이터 확인 | real 전체={len(real_files_all)}, fake 전체={len(fake_files_all)}, "
        f"매칭 pair={len(matched_ids)}, 평가 샘플={len(files)}"
    )

    if len(matched_real) == 0 or len(matched_fake) == 0:
        return [], [], []

    return files, labels, sample_infos


def load_svd_trained_weight(model, weight_path: Path):
    state_dict = torch.load(str(weight_path), map_location=DEVICE)

    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "").replace("backbone.", "")
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=False)
    return model


def forward_model(model, model_name: str, bx):
    """
    기존 최종 학습/검증 코드와 맞추기 위해 VideoMAE는 pixel_values 명시.
    나머지 모델은 기존 cross-domain test처럼 model(bx) 사용.
    """
    if "videomae" in model_name.lower():
        return model(pixel_values=bx).logits

    outputs = model(bx)

    if hasattr(outputs, "logits"):
        outputs = outputs.logits

    return outputs


def evaluate_one_fold(model_name, weight_prefix, fold, domain_name, files, labels, sample_infos):
    weight_path = WEIGHT_DIR / f"{weight_prefix}_f{fold}.pth"

    if not weight_path.exists():
        print(f"      [SKIP] 가중치 없음: {weight_path}")
        return None, []

    frames = get_frame_count(model_name)

    test_loader = get_dataloader(
        files,
        labels,
        model_name=model_name,
        batch_size=TEST_BATCH_SIZE,
        mode="test",
        frames=frames,
    )

    model = get_model(model_name, device=DEVICE, num_classes=2)

    try:
        model = load_svd_trained_weight(model, weight_path)
        model.eval()
    except Exception as e:
        print(f"      [FAIL] Fold {fold} weight 로드 실패: {e}")
        del model
        clean_memory()
        return None, []

    trues = []
    probs = []

    with torch.no_grad():
        for bx, by in tqdm(
            test_loader,
            desc=f"      {model_name} | {domain_name} | fold {fold}",
            leave=False,
        ):
            bx = bx.to(DEVICE)

            outputs = forward_model(model, model_name, bx)
            prob = torch.softmax(outputs, dim=1)[:, 1]

            trues.extend(by.cpu().numpy())
            probs.extend(prob.cpu().numpy())

    del model
    clean_memory()

    trues = np.array(trues)
    probs = np.array(probs)

    if len(np.unique(trues)) < 2:
        print(f"      [FAIL] Fold {fold} 지표 계산 실패: label class가 1개뿐입니다.")
        return None, []

    try:
        auc = roc_auc_score(trues, probs)
        apcer, bpcer, eer, best_thresh = calculate_metrics_at_best_threshold(trues, probs)

        preds = (probs >= best_thresh).astype(int)

        acc = accuracy_score(trues, preds)
        pre = precision_score(trues, preds, zero_division=0)
        rec = recall_score(trues, preds, zero_division=0)

    except Exception as e:
        print(f"      [FAIL] Fold {fold} 지표 계산 실패: {e}")
        return None, []

    metric_row = {
        "Model": model_name,
        "Weight_Prefix": weight_prefix,
        "Domain": domain_name,
        "Fold": fold,
        "Frames": frames,
        "Batch_Size": TEST_BATCH_SIZE,
        "N": int(len(trues)),
        "N_Real": int((trues == 0).sum()),
        "N_Fake": int((trues == 1).sum()),
        "AUC": float(auc),
        "EER": float(eer),
        "APCER": float(apcer),
        "BPCER": float(bpcer),
        "ACC": float(acc),
        "Precision": float(pre),
        "Recall": float(rec),
        "Best_Threshold": float(best_thresh),
    }

    prediction_rows = []

    for i, info in enumerate(sample_infos):
        true_label = int(trues[i])
        prob_fake = float(probs[i])
        pred_label = int(prob_fake >= best_thresh)

        prediction_rows.append({
            "Model": model_name,
            "Weight_Prefix": weight_prefix,
            "Domain": domain_name,
            "Fold": fold,
            "Frames": frames,
            "Batch_Size": TEST_BATCH_SIZE,
            "target_id": info["target_id"],
            "file_name": info["file_name"],
            "file_path": info["file_path"],
            "label_name": info["label_name"],
            "true_label": true_label,
            "prob_fake": prob_fake,
            "best_threshold": float(best_thresh),
            "pred_label": pred_label,
            "correct": int(pred_label == true_label),
        })

    return metric_row, prediction_rows


def summarize_fold_results(fold_df: pd.DataFrame):
    metric_cols = [
        "AUC",
        "EER",
        "APCER",
        "BPCER",
        "ACC",
        "Precision",
        "Recall",
    ]

    summary_rows = []

    for (model, domain), g in fold_df.groupby(["Model", "Domain"]):
        row = {
            "Model": model,
            "Domain": domain,
            "N_Folds": int(len(g)),
            "Frames": int(g["Frames"].iloc[0]),
            "Batch_Size": int(g["Batch_Size"].iloc[0]),
            "N_Mean": float(g["N"].mean()),
            "N_Real_Mean": float(g["N_Real"].mean()),
            "N_Fake_Mean": float(g["N_Fake"].mean()),
        }

        for col in metric_cols:
            row[f"{col}_Mean"] = float(g[col].mean())
            row[f"{col}_Std"] = float(g[col].std(ddof=0))

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    model_order = {
        "xception": 0,
        "swin_tiny_patch4_window7_224": 1,
        "r3d_18": 2,
        "videomae_base": 3,
    }

    domain_order = {
        "FFPP_Raw": 0,
        "FFPP_Instagram": 1,
        "FFPP_YouTube": 2,
        "FFPP_Kakao_High": 3,
        "FFPP_Kakao_Normal": 4,
    }

    summary_df["Model_Order"] = summary_df["Model"].map(model_order)
    summary_df["Domain_Order"] = summary_df["Domain"].map(domain_order)

    summary_df = summary_df.sort_values(["Model_Order", "Domain_Order"])
    summary_df = summary_df.drop(columns=["Model_Order", "Domain_Order"])

    return summary_df


def main():
    print("============================================================")
    print("FF++ Face2Face c23 external dataset evaluation")
    print("공간 모델 2개: Xception, Swin-Tiny")
    print("시간 모델 2개: R3D-18, VideoMAE")
    print("Hybrid 제외")
    print(f"DEVICE      : {DEVICE}")
    print(f"DATASET_ROOT: {DATASET_ROOT}")
    print(f"WEIGHT_DIR  : {WEIGHT_DIR}")
    print(f"SAVE_DIR    : {SAVE_DIR}")
    print(f"TEST_BATCH  : {TEST_BATCH_SIZE}")
    print("============================================================")

    validate_weight_files()

    fold_rows = []
    prediction_rows_all = []

    for model_name, weight_prefix in MODELS.items():
        frames = get_frame_count(model_name)

        print("\n============================================================")
        print(f"모델 평가 시작: {model_name}")
        print(f"사용 프레임 수: {frames}")
        print("============================================================")

        for domain_name, domain_path in DOMAINS.items():
            print(f"\n   도메인: {domain_name}")
            print(f"   경로  : {domain_path}")

            if not domain_path.exists():
                print("   [SKIP] 도메인 경로 없음")
                continue

            files, labels, sample_infos = prepare_ffpp_dataset(domain_path)

            if len(files) == 0:
                print("   [SKIP] 평가 데이터 없음")
                continue

            for fold in range(1, 6):
                metric_row, pred_rows = evaluate_one_fold(
                    model_name=model_name,
                    weight_prefix=weight_prefix,
                    fold=fold,
                    domain_name=domain_name,
                    files=files,
                    labels=labels,
                    sample_infos=sample_infos,
                )

                if metric_row is not None:
                    fold_rows.append(metric_row)
                    prediction_rows_all.extend(pred_rows)

                    print(
                        f"      Fold {fold} | "
                        f"AUC={metric_row['AUC']:.4f}, "
                        f"EER={metric_row['EER']:.4f}, "
                        f"APCER={metric_row['APCER']:.4f}, "
                        f"BPCER={metric_row['BPCER']:.4f}, "
                        f"ACC={metric_row['ACC']:.4f}"
                    )

    if not fold_rows:
        print("\n평가 결과가 없습니다. 데이터 경로, 가중치 파일명, 환경을 확인하세요.")
        return

    fold_df = pd.DataFrame(fold_rows)

    fold_csv = SAVE_DIR / "ffpp_external_fold_metrics.csv"
    fold_df.to_csv(fold_csv, index=False, encoding="utf-8-sig")

    summary_df = summarize_fold_results(fold_df)

    summary_csv = SAVE_DIR / "ffpp_external_metrics_report.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    pred_csv = None
    if prediction_rows_all:
        pred_df = pd.DataFrame(prediction_rows_all)
        pred_csv = SAVE_DIR / "ffpp_external_predictions.csv"
        pred_df.to_csv(pred_csv, index=False, encoding="utf-8-sig")

    print("\n============================================================")
    print("FF++ 외부 데이터셋 평가 완료")
    print(f"Fold별 결과 저장  : {fold_csv}")
    print(f"요약 결과 저장    : {summary_csv}")

    if pred_csv is not None:
        print(f"Prediction 저장   : {pred_csv}")

    print("============================================================")

    try:
        pivot = summary_df.pivot(index="Model", columns="Domain", values="AUC_Mean")
        print("\n[도메인별 AUC 평균 요약]")
        print(pivot.to_string())
    except Exception:
        pass


if __name__ == "__main__":
    main()