import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
import itertools
import os

# ⚙️ 경로 설정 (3_final_train.py 결과 연동)
RESULT_CSV = r"C:\Users\leejy\Desktop\test_experiment\results\final_weights\final_training_summary.csv"
SAVE_DIR = r"C:\Users\leejy\Desktop\test_experiment\results\statistics"
os.makedirs(SAVE_DIR, exist_ok=True)

def run_analysis():
    print("🔬 [10.1] 논문용 통계 검정 및 성능표 생성 시작")
    
    if not os.path.exists(RESULT_CSV):
        print(f"❌ 결과 파일을 찾을 수 없습니다: {RESULT_CSV}")
        return

    df = pd.read_csv(RESULT_CSV)
    
    # 1. [성능 요약표] 논문 삽입용 Mean ± Std 계산
    summary = df.groupby('model').agg({
        'best_auc': ['mean', 'std'],
        'eer': ['mean', 'std'],
        'apcer': ['mean', 'std'],
        'best_thresh': ['mean'] # 최적 임계값 평균 확인
    }).reset_index()
    
    # 컬럼명 정리
    summary.columns = ['Model', 'AUC_Mean', 'AUC_Std', 'EER_Mean', 'EER_Std', 'APCER_Mean', 'APCER_Std', 'Avg_Threshold']
    summary_path = os.path.join(SAVE_DIR, "table1_performance_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"   📊 성능 요약표 저장됨: {summary_path}")
    print(summary)

    # 2. [윌콕슨 검정] 모델 우위 증명 (ISO/IEC 표준 EER 기준)
    print("\n   🧬 모델 간 유의성 검정 (Wilcoxon Signed-rank Test on EER)")
    models = df['model'].unique()
    pairwise_results = []

    for m1, m2 in itertools.combinations(models, 2):
        scores1 = df[df['model'] == m1]['eer'].values
        scores2 = df[df['model'] == m2]['eer'].values
        
        # 샘플 수가 적으므로(n=5) exact method 사용
        if len(scores1) == 5:
            stat, p_val = wilcoxon(scores1, scores2)
            
            # 논문 서술용 판정 (p < 0.05)
            significance = "**Significant**" if p_val < 0.05 else "Not Significant"
            winner = m1 if scores1.mean() < scores2.mean() else m2 # EER은 낮을수록 좋음
            
            pairwise_results.append({
                "Model A": m1, "Model B": m2, 
                "p-value": f"{p_val:.4f}", 
                "Result": significance,
                "Winner": winner
            })

    stats_path = os.path.join(SAVE_DIR, "table2_significance_test.csv")
    pd.DataFrame(pairwise_results).to_csv(stats_path, index=False)
    print(f"   🧬 검정 결과 리포트 저장됨: {stats_path}")

if __name__ == "__main__":
    run_analysis()