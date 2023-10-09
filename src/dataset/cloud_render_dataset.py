import os
from random import uniform
import torch

from kaolin.ops.mesh import sample_points
import nvdiffrast.torch as dr

import src.shared.util as UT
import src.shared.phong as PH
import src.shared.transforms as TR

class CloudRenderDataset(torch.utils.data.Dataset):
    
    def __init__(self, stl_root, n_samples, device, resolution=None,
            coarse_root=None, fine_root=None, scale=True,
            flip=[1., 1., -1.], material=0.8, random_scale=0.1):
        self.glctx = dr.RasterizeGLContext()
        self.stl_root = stl_root
        self.coarse_root = coarse_root
        self.fine_root = fine_root
        self.n_samples = n_samples
        self.device = device
        self.scale = scale
        self.resolution = resolution        
        self.material = material
        self.rs = random_scale
        self.records = [
            UT.load_mesh(stl_root+f, flip=flip, batch=True) + (f,)
            for f in sorted(os.listdir(stl_root))
            if f.endswith('.stl')
        ]
        if coarse_root:
            self.coarse = torch.cat([
                torch.load(coarse_root+f)
                for f in sorted(os.listdir(coarse_root))
            ], dim=0).cpu()
        if fine_root:
            self.fine = torch.cat([
                torch.load(fine_root+f)
                for f in sorted(os.listdir(fine_root))
            ], dim=0).cpu()
        
    def __len__(self):        
        return len(self.records)    
    
    def __getitem__(self, idx):
        scale = torch.Tensor([uniform(1-self.rs, 1+self.rs) for _ in range(3)])

        vertices, faces, _ = self.records[idx]
        if self.scale:
            vertices = vertices * scale
        else:
            scale = torch.ones_like(scale)

        vertices, faces = [f.to(self.device) for f in (vertices, faces)]
        
        res = {}
        if self.resolution:
            material = self.material + torch.zeros_like(vertices)
            samples = PH.sample_all(1, self.device)
            img = PH.render_diffuse(self.glctx, vertices, faces, material, 
                              resolution=self.resolution, samples=samples)
            img = TR.channel_first(img)                          
            res['img'] = img[0].cpu()
            res['colors'] = samples[0][0].cpu()
            res['directions'] = samples[1][0].cpu()
            res['views'] = samples[2][0].cpu()        
        if self.n_samples:
            cloud, _  = sample_points(vertices, faces, self.n_samples)
            res['cloud'] = cloud[0].cpu()
        if self.coarse_root:
            res['coarse'] =  self.coarse[idx] * scale[:, None, None]
        if self.fine_root:
            res['fine'] =  self.fine[idx] * scale[:, None, None]
        return res