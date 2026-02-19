import pandas as pd
import os

# 📂 경로 설정
BASE_DIR = r"C:\Users\leejy\Desktop\test_experiment\dataset\sns_analysis"
INPUT_CSV = os.path.join(BASE_DIR, "final_forensic_report.csv")
OUTPUT_EXCEL = os.path.join(BASE_DIR, "SNS_Distortion_Matrix_Final.xlsx")

def generate_summary_matrix():
    # 1. 데이터 로드
    if not os.path.exists(INPUT_CSV):
        print(f"❌ 분석 데이터가 없습니다: {INPUT_CSV}")
        print("👉 먼저 'sns_forensic_report.py'를 실행하여 Raw Data를 만드세요.")
        return

    df = pd.read_csv(INPUT_CSV)
    print(f"✅ 분석된 샘플 수: {len(df)}개")

    # 2. 집계 규칙 (프로토콜 기준)
    # ⚠️ 수정됨: 'Resolution' -> 'Dist_Res' (CSV 파일의 컬럼명과 일치시킴)
    agg_rules = {
        'Est_CRF': 'mean',              # 추정 CRF
        'Dist_Res': lambda x: x.mode()[0] if not x.mode().empty else "N/A", # 출력 해상도 (수정됨)
        'Codec': lambda x: x.mode()[0] if not x.mode().empty else "N/A",    # 코덱
        'Box_Sequence': lambda x: x.mode()[0] if not x.mode().empty else "N/A", # Box Sequence
        'Blockiness': 'mean',           # 기만적 아티팩트
        'FPS_Diff': 'mean',             # 프레임 변동
        'Bitrate_Loss(%)': 'mean'       # 비트레이트 손실률
    }

    # 3. 매트릭스 생성
    # agg_rules의 순서대로 컬럼이 생성됩니다.
    matrix_df = df.groupby('Platform').agg(agg_rules)

    # 4. 수치 다듬기
    matrix_df['Est_CRF'] = matrix_df['Est_CRF'].round(1)
    matrix_df['Blockiness'] = matrix_df['Blockiness'].round(3)
    matrix_df['FPS_Diff'] = matrix_df['FPS_Diff'].round(2)
    matrix_df['Bitrate_Loss(%)'] = matrix_df['Bitrate_Loss(%)'].round(1)

    # 5. 컬럼명 한글화 (보고서용)
    # 순서가 agg_rules와 일치해야 합니다.
    matrix_df.columns = [
        '추정 CRF (압축강도)', 
        '출력 해상도', 
        '코덱 (Codec)', 
        'Box Sequence (지문)', 
        'Blockiness (공간왜곡)', 
        'FPS 변동 (시간왜곡)', 
        '비트레이트 손실(%)'
    ]

    # 6. 엑셀 저장
    matrix_df.to_excel(OUTPUT_EXCEL)
    
    print("\n" + "="*60)
    print("🎉 [최종 마스터] SNS 왜곡 특성 매트릭스 생성 완료!")
    print(f"💾 파일 위치: {OUTPUT_EXCEL}")
    print("="*60)
    print(matrix_df)

if __name__ == "__main__":
    generate_summary_matrix()