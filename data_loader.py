import os
import glob
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import DeepfakeDataset

def get_transforms(model_name):
    """[설계안 6절] Lanczos4 보간법 강제 적용"""
    name = model_name.lower()
    size = (112, 112) if "r3d" in name else (224, 224)
    
    if "r3d" in name:
        mean, std = [0.432, 0.394, 0.376], [0.228, 0.221, 0.216]
    else:
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    
    return transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

def get_dataloader(files, labels, model_name, batch_size, mode='train', frames=16):
    ds = DeepfakeDataset(
        file_paths=files, 
        labels=labels, 
        model_name=model_name, 
        mode=mode,
        transform=get_transforms(model_name),
        window_size=frames
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=(mode == 'train'), num_workers=0, pin_memory=True)

def prepare_dataset(base_dir):
    """
    [수정] Real(000~299)와 Fake(svd_000~299)의 ID 매칭
    - 목적: 같은 번호를 가진 Real과 Fake가 같은 Fold에 들어가도록 강제함 (Leakage 방지)
    """
    real_dir = os.path.join(base_dir, "real")
    fake_dir = os.path.join(base_dir, "fake")
    
    real_files = sorted(glob.glob(os.path.join(real_dir, "*.mp4")))
    fake_files = sorted(glob.glob(os.path.join(fake_dir, "*.mp4")))
    
    files = real_files + fake_files
    labels = [0] * len(real_files) + [1] * len(fake_files)
    
    # 그룹 ID 생성: "svd_001.mp4" -> "001", "001.mp4" -> "001"
    groups = []
    for p in files:
        name = os.path.basename(p)
        # 'svd_' 접두사 제거하여 숫자 ID만 추출
        clean_id = name.replace("svd_", "").split(".")[0]
        groups.append(clean_id)
            
    return files, labels, groups