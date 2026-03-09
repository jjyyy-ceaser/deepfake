import os
import torch
import torch.nn as nn
import timm
import torchvision.models as models
from transformers import VideoMAEForVideoClassification
from sklearn.metrics import confusion_matrix, roc_curve
import numpy as np

# ---------------------------------------------------------
# [Hybrid Model] ConvNeXt + GRU (Rev.18 Specification)
# ---------------------------------------------------------
class HybridModel(nn.Module):
    def __init__(self, backbone_name='convnext_tiny', hidden_size=512, num_classes=2, dropout_rate=0.5):
        super(HybridModel, self).__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)

        # 🔧 [설계안 9-2] ConvNeXt 백본 동결
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        with torch.no_grad():
            feat_dim = self.backbone(torch.randn(1, 3, 224, 224)).shape[1]
        self.gru = nn.GRU(input_size=feat_dim, hidden_size=hidden_size, num_layers=1, batch_first=True)
        
        # 🔧 [설계안 9-2] GRU 직교 초기화
        for name, param in self.gru.named_parameters():
            if 'weight' in name: nn.init.orthogonal_(param)
            elif 'bias' in name: nn.init.constant_(param, 0)
        
        self.fc = nn.Sequential(nn.Dropout(p=dropout_rate), nn.Linear(hidden_size, num_classes))

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w) 
        features = self.backbone(x).reshape(b, t, -1)
        _, h_n = self.gru(features)
        return self.fc(h_n[-1])

# ---------------------------------------------------------
# 📊 [10절] ISO/IEC 포렌식 지표 산출 (최적 임계값)
# ---------------------------------------------------------
def calculate_iso_metrics(trues, bins, probs):
    """(기존) 고정 임계값 0.5 기준 지표 산출"""
    try:
        tn, fp, fn, tp = confusion_matrix(trues, bins).ravel()
        apcer = fn / (tp + fn) if (tp + fn) > 0 else 0.0
        bpcer = fp / (tn + fp) if (tn + fp) > 0 else 0.0
        fpr, tpr, _ = roc_curve(trues, probs)
        eer = fpr[np.nanargmin(np.absolute((1 - tpr) - fpr))]
    except: apcer, bpcer, eer = 0.0, 0.0, 0.5
    return apcer, bpcer, eer

def calculate_metrics_at_best_threshold(trues, probs):
    """[설계안 9-4] EER이 발생하는 최적 임계값 자동 탐색"""
    try:
        fpr, tpr, thresholds = roc_curve(trues, probs)
        fnr = 1 - tpr
        eer_idx = np.nanargmin(np.absolute(fnr - fpr))
        
        eer = fpr[eer_idx]
        best_thresh = thresholds[eer_idx]
        
        bins_opt = (np.array(probs) >= best_thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(trues, bins_opt).ravel()
        
        apcer = fn / (tp + fn) if (tp + fn) > 0 else 0.0
        bpcer = fp / (tn + fp) if (tn + fp) > 0 else 0.0
        
        return apcer, bpcer, eer, best_thresh
    except:
        return 0.0, 0.0, 0.5, 0.5

# ---------------------------------------------------------
# 🏭 Model Factory
# ---------------------------------------------------------
def get_model(model_name, device, num_classes=2, **kwargs):
    name = model_name.lower()
    dropout_rate = kwargs.get('dropout', 0.5)
    
    if 'xception' in name:
        model = timm.create_model('xception', pretrained=False, num_classes=num_classes, drop_rate=dropout_rate)
        weights_path = r"C:\Users\leejy\Desktop\test_experiment\weights\xception_ffpp.pth"
        if os.path.exists(weights_path):
            checkpoint = torch.load(weights_path, map_location='cpu')
            state_dict = checkpoint.get('state_dict', checkpoint)
            new_state_dict = {k.replace('backbone.', '').replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict, strict=False)
        
        # 하위 2개 블록 동결
        for i, child in enumerate(model.children()):
            if i < 2:
                for p in child.parameters(): p.requires_grad = False
                
    elif 'swin' in name:
        model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=num_classes, drop_path_rate=0.2)
        
    elif 'r3d' in name:
        # 🚨 [R3D 수정] 인터넷 다운로드 차단 및 로컬 파일 강제 로드
        model = models.video.r3d_18(weights=None) # 빈 모델 생성
        
        # 재용 씨의 로컬 경로 하드코딩
        r3d_path = r"C:\Users\leejy\Desktop\test_experiment\weights\hub\checkpoints\r3d_18-b3b3357e.pth"
        
        if os.path.exists(r3d_path):
            state_dict = torch.load(r3d_path, map_location='cpu')
            # fc 레이어는 클래스 수가 다르므로(400 vs 2) 제외하고 로드
            model.load_state_dict({k: v for k, v in state_dict.items() if 'fc' not in k}, strict=False)
            print(f"✅ [R3D-18] 로컬 KINETICS 가중치 로드 완료: {r3d_path}")
        else:
            print("⚠️ [R3D-18] 지정된 경로에 가중치가 없습니다. 랜덤 초기화 상태로 시작합니다.")
            
        # 분류기 재설정 (2 Class)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif 'videomae' in name:
        model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics", num_labels=num_classes, ignore_mismatched_sizes=True)
        
    elif 'hybrid' in name:
        model = HybridModel(dropout_rate=dropout_rate)
        for p in model.backbone.parameters(): p.requires_grad = False
        
    return model.to(device)