import os
import torch
import torch.nn.functional as F

from torch.distributions import Uniform
from torchvision.transforms import (
    Compose,
    Lambda,
    GaussianBlur,
)

def create_transform(size, scale=(0.9, 1.1), blur=(3,1)):
    return Compose([    
        GaussianBlur(blur[0], sigma=blur[1]),
        Lambda(lambda t: t *  torch.tensor([1.,1.,1.])[:, None, None]),
        Lambda(lambda t: t *  Uniform(*scale).sample((3, 1, 1))), 
        Lambda(lambda t: t - t.mean(dim=(-1, -2), keepdim=True)), # center
        Lambda(lambda t: t / torch.max(t.abs())), # scale
        Lambda(lambda t: t if t.dim() == 4 else t[None]), 
        Lambda(lambda t: F.interpolate(t, size=size, mode='bilinear')),
        Lambda(lambda t: t if t.dim() == 3 else t[0]),
    ])

class GridDataset(torch.utils.data.Dataset):
    def __init__(self, grid_root, transform=None):
        self.grid_root = grid_root        
        self.transform = transform
        self.records = torch.stack([
            torch.load(grid_root + f)
            for f in os.listdir(grid_root)
            if f.endswith('.pth')
        ])        
        
    def __len__(self):        
        return len(self.records)
    
    def __getitem__(self, idx):
        if self.transform:
            return self.transform(self.records[idx])[0]
        return self.records[idx][0]