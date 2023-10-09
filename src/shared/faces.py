# pyright: reportMissingImports=false
import torch

from trimesh.util import triangle_strips_to_faces

def create_strips(n, m):
    res = []
    for i in range(n-1):
        strip = []
        for j in range(m):            
            strip.append(j+(i+1)*m)
            strip.append(j+i*m)
            #strip.append(j+(i+1)*m)
        res.append(strip)
    return res

def make_faces(n, m=None, device=None):
    strips = create_strips(n, m or n)    
    return torch.tensor(triangle_strips_to_faces(strips),
                        dtype=torch.int32, device=device)
