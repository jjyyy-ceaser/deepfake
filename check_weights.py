import os
import torch
import timm
import torch.nn as nn

def check_xception():
    path = r"C:\Users\leejy\Desktop\test_experiment\weights\xception_ffpp.pth"
    print(f"🔍 경로 확인: {path}")
    
    if not os.path.exists(path):
        print("❌ [파일 없음] 경로가 틀렸거나 파일이 해당 위치에 없습니다.")
        return

    try:
        print("⏳ 가중치 파일 정밀 분석 중...")
        # weights_only=False로 설정하여 구조 파악
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        
        # 1. 딕셔너리 구조 파악
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        print(f"✅ 파일 로드 성공 (레이어 수: {len(state_dict)})")
        
        # 2. 모델 생성
        model = timm.create_model('xception', pretrained=False, num_classes=2)
        
        # 3. 레이어 이름 매칭 시도 (module. 및 backbone. 접두어 제거)
        new_state_dict = {}
        matched_count = 0
        
        print("🛠️ 접두어 제거 및 매칭 작업 시작...")
        for k, v in state_dict.items():
            # 발견된 'backbone.' 및 'module.' 접두어 제거
            name = k.replace('module.', '').replace('backbone.', '')
            
            if name in model.state_dict():
                new_state_dict[name] = v
                matched_count += 1
        
        print(f"📊 매칭 결과: 전체 {len(state_dict)}개 중 {matched_count}개 레이어 이름 일치")
        
        # 4. 결과 판정
        if matched_count == 0:
            print("\n❌ 여전히 일치하는 레이어가 없습니다. 파일과 모델의 키를 직접 대조해야 합니다.")
        elif matched_count < 100:
            print(f"\n⚠️ 일부 레이어({matched_count}개)만 일치합니다. 모델 구조가 다를 수 있습니다.")
        else:
            # 매칭된 가중치로 로드 시도
            msg = model.load_state_dict(new_state_dict, strict=False)
            print(f"\n🎉 [최종 성공] {matched_count}개의 레이어를 성공적으로 매칭했습니다!")
            print("💡 조치: utils.py의 로드 로직에도 '.replace(\'backbone.\', \'\')' 코드를 추가해야 합니다.")

    except Exception as e:
        print(f"❌ [에러] {e}")

if __name__ == "__main__":
    check_xception()