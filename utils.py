import torch
import torch.nn as nn
import timm
import torchvision.models as models
from transformers import VideoMAEForVideoClassification

# ---------------------------------------------------------
# [New] Hybrid Model: ConvNeXt + GRU (Rev.9 Specification)
# ---------------------------------------------------------
class HybridModel(nn.Module):
    def __init__(self, backbone_name='convnext_tiny', hidden_size=512, num_classes=2, dropout_rate=0.5):
        super(HybridModel, self).__init__()
        # 1. Backbone: ConvNeXt-Tiny (Pretrained)
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        
        # 2. Get Feature Dimension dynamically
        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            feat_dim = self.backbone(dummy).shape[1]
            
        # 3. Temporal Modeling: GRU
        self.gru = nn.GRU(input_size=feat_dim, hidden_size=hidden_size, num_layers=1, batch_first=True)
        
        # 4. Classifier Head
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        # x: (Batch, Frames, Channels, Height, Width)
        b, f, c, h, w = x.shape
        
        # ⚡ [수정] view -> reshape (메모리 불연속성 문제 해결)
        x = x.reshape(b * f, c, h, w) 
        
        features = self.backbone(x) # (B*F, 768)
        
        # Reshape for RNN
        features = features.reshape(b, f, -1) # (B, F, 768)
        
        # Temporal Analysis
        _, h_n = self.gru(features) # h_n: (1, B, 512)
        
        # Final Classification
        return self.fc(h_n[-1])

# ---------------------------------------------------------
# Model Factory
# ---------------------------------------------------------
def get_model(model_name, device, num_classes=2, dropout_rate=0.0):
    name = model_name.lower()
    model = None
    
    if 'xception' in name:
        model = timm.create_model('xception', pretrained=True, num_classes=num_classes)
        if dropout_rate > 0:
            model.fc = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(model.fc.in_features, num_classes)
            )
            
    elif 'swin' in name:
        # drop_path_rate=0.2 인자를 추가하여 사양서의 Stochastic Depth 준수
        model = timm.create_model(
            'swin_tiny_patch4_window7_224', 
            pretrained=True, 
            num_classes=num_classes,
            drop_path_rate=0.2  # 👈 이 부분이 추가되어야 합니다.
        )
        
    elif 'r3d' in name:
        # Fixed: R3D-18 Baseline (Kinetics400)
        weights = models.video.R3D_18_Weights.KINETICS400_V1
        model = models.video.r3d_18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif 'videomae' in name:
        model = VideoMAEForVideoClassification.from_pretrained(
            "MCG-NJU/videomae-base-finetuned-kinetics", 
            num_labels=num_classes, 
            ignore_mismatched_sizes=True
        )
        
    elif 'hybrid' in name:
        model = HybridModel(dropout_rate=dropout_rate)
        
    else:
        raise ValueError(f"Unknown model name: {model_name}")
        
    return model.to(device)