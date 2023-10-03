import os
import torch
import torch.nn.functional as F
from torchvision.transforms import (
    Compose,
    Lambda,
)

import src.shared.augmentation as AUG

barebone_transform = Compose([
    #Lambda(lambda t: AUG.random_execution(t, 0.1, AUG.gaussian_blur)),    
    # Lambda(lambda t: AUG.random_fork(t, 0.8, 
    #                    lambda e: AUG.resize_bilinear(e, 12), 
    #                    lambda e: AUG.resize_bicubic(e, 12))),
    Lambda(lambda t: AUG.random_scale(t)),
    Lambda(lambda t: AUG.max_scale(t)),
    Lambda(lambda t: AUG.center(t)),
    Lambda(lambda t: AUG.random_execution(t, 0.5, AUG.reflect)),
    Lambda(lambda t: AUG.random_execution(t, 0.25, AUG.split_nth)),        
    Lambda(lambda t: AUG.random_shift(t)),
    #Lambda(lambda t: AUG.random_execution(t, 0.25, AUG.perlin_noise)),
    #Lambda(lambda t: AUG.random_execution(t, 0.25, AUG.fractal_noise)),            
])

blend_transform  = Compose([    
    Lambda(lambda t: AUG.random_execution(t, 0.1, AUG.gaussian_blur)),
    Lambda(lambda t: AUG.random_scale(t)),
    Lambda(lambda t: AUG.max_scale(t)),
    Lambda(lambda t: AUG.center(t)),
    Lambda(lambda t: AUG.random_execution(t, 0.5, AUG.reflect)),
    Lambda(lambda t: AUG.random_execution(t, 0.25, AUG.split_nth)),        
    Lambda(lambda t: AUG.random_shift(t)),
    Lambda(lambda t: AUG.random_execution(t, 0.1, AUG.perlin_noise)),
    Lambda(lambda t: AUG.random_execution(t, 0.1, AUG.fractal_noise)),
])

class BlendDataset(torch.utils.data.Dataset):
    def __init__(self, pth_root, device=None, transform=None):
        self.pth_root = pth_root        
        self.transform = transform
        self.records = torch.stack([
            torch.load(pth_root + f)
            for f in os.listdir(pth_root)
            if f.endswith('.pth')
        ]).float()
        if device:
            self.records = self.records.to(device)
        self._n = len(self.records)
        
    def __len__(self):        
        return self._n ** 2
    
    def __getitem__(self, idx):
        t = self.records[idx // self._n]
        r = self.records[idx % self._n]        
        if self.transform:
            t = self.transform(t)
            r = self.transform(r)        
        return AUG.blend_two(t, r)[0]