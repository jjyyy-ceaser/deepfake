import pandas as pd
from datasets import load_dataset

# 데이터셋 ID (이게 정확해야 함)
DATASET_ID = "luchaoqi/TalkingHeadBench"

print(f"🕵️ '{DATASET_ID}'의 내부 지시서(Parquet)를 조회합니다...")

try:
    # 1. 영상은 안 받고 '정보'만 로드 (streaming=True)
    # trust_remote_code=True는 데이터셋 스크립트가 있을 경우 필수
    ds = load_dataset(DATASET_ID, split="train", streaming=True, trust_remote_code=True)
    
    print("\n✅ 지시서 로드 성공! 첫 번째 데이터의 내용을 공개합니다:\n")
    
    # 2. 첫 번째 데이터 1개만 꺼내서 내용 확인
    first_item = next(iter(ds))
    
    # 보기 좋게 출력
    for key, value in first_item.items():
        # 너무 긴 내용은 잘라서 출력
        str_val = str(value)
        if len(str_val) > 100: str_val = str_val[:100] + "..."
        print(f"🔹 {key}: {str_val}")

    print("\n" + "="*50)
    
    # 3. 핵심 정보가 있는지 분석
    keys = first_item.keys()
    if 'start_time' in keys or 'timestamp' in keys or 'clips' in keys:
        print("🎉 찾았다! 이 안에 '시간 정보(Timestamp)'가 들어있습니다.")
        print("이제 이 정보를 이용해 원본을 자르면 됩니다.")
    else:
        print("⚠️ 시간 정보가 안 보입니다. 이 데이터셋은 '이미 잘린 영상'을 제공하거나")
        print("별도의 메타데이터 파일이 다른 곳(GitHub 등)에 있을 수 있습니다.")

except Exception as e:
    print(f"\n❌ 로드 실패: {e}")
    print("이유: 데이터셋 접근 권한이 없거나, 인터넷 연결 문제, 또는 로그인이 필요할 수 있습니다.")