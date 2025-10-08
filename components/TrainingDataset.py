# ---NOTES---
# ---IMPORTS---
from torch.utils.data import Dataset
import torch
import scipy.io as sio
import os
from PIL import Image
import torchvision.transforms as transforms
import re
# -------------

class TrainingDataset(Dataset):
    
    def __init__(self, img_dir, gt_dir):
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
])
        self.img_files = sorted(
            [f for f in os.listdir(img_dir) if f.endswith('.jpg')],
            key=lambda x: int(re.search(r'\d+', x).group())
)
        
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        
        gt_path = os.path.join(self.gt_dir, 'GT_' + self.img_files[idx].replace('.jpg', '.mat'))
        
        img = Image.open(img_path).convert('RGB')
        
        mat = sio.loadmat(gt_path)
        points = mat['image_info'][0][0][0][0][0]
        count = points.shape[0]
        
        if self.transform:
            img = self.transform(img)
        
        return img, torch.tensor([count], dtype=torch.float32)
            
            
            
                
        
    
    
    