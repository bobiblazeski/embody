import os
from random import uniform
import torch
import trimesh
from torchvision.transforms import v2

from render import (load_trimesh, prepare_patch, render_mesh)

transform = v2.Compose([
    v2.ToImageTensor(),
    v2.ConvertImageDtype(torch.float32),
    #v2.Resize(size=(224, 224), antialias=True),
    v2.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.), ratio=(0.9, 1.),antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class CoarseRenderDataset(torch.utils.data.Dataset):
    
    def __init__(self, stl_root, coarse_root=None, random_scale=0.025, transform=transform):        
        self.stl_root = stl_root
        self.coarse_root = coarse_root
        self.rs = random_scale
        self.transform = transform
        self.meshes = [            
            load_trimesh(stl_root+f, as_trimesh=False)
            for f in sorted(os.listdir(stl_root))
            if f.endswith('.stl')
        ]
        self.coarse = self.load_patches(coarse_root)                    

    def load_patches(self, root):
        list = [torch.load(root+f) for f in sorted(os.listdir(root))]        
        if list[0].dim() == 3:            
            return torch.stack(list).cpu()
        return torch.cat(list, dim=0).cpu()

    def __len__(self):        
        return len(self.meshes)

    def render_mesh(self, v, f, scale):         
        return render_mesh(trimesh.Trimesh(vertices=v * scale,  faces=f))[0]

    def render_patch(self, patch):        
        return render_mesh(prepare_patch(patch))[0]
    
    def __getitem__(self, idx):
        scale = torch.Tensor([uniform(1-self.rs, 1+self.rs) for _ in range(3)])        
        stl_img = self.render_mesh(*self.meshes[idx], scale)
        coarse = self.coarse[idx] * scale[:, None, None]        
        coarse_img = self.render_patch(coarse)
        return {
            'stl_img': self.transform(stl_img.copy()),
            'coarse':  coarse, 
            'coarse_img':self.transform(coarse_img.copy()), 
        }