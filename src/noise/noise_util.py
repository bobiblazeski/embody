from math import log2
import torch
import torch.nn.functional as F

import src.shared.util as U
import src.shared.normals as NRM


def strength(n, exp=None):
    if exp:
        t = 1 / (exp**torch.arange(n, dtype=torch.float32))
    else:
        t = 1/ (torch.arange(n, dtype=torch.float32)+1)        
    return t / t.sum()

def rescale(t, size, mode='bilinear'):
    y = F.interpolate(t, size, mode=mode, align_corners=True)
    faces = U.make_faces(y.size(-1))
    
    yn = NRM.vertex_normals(U.to_vertices(y), faces)
    yn = U.to_patch(yn)
    y = torch.cat([y, yn], dim=1)
    y = F.interpolate(y, t.size(-1), mode=mode, align_corners=True)
    return torch.chunk(y, 2, dim=1)

def normalize_batch(x):
    x = x - x.mean(dim=(1, 2, 3), keepdim=True)
    return x / x.std(dim=(1, 2, 3), keepdim=True)

def normalize_cloud(cloud,  mean, std):
    return (cloud - mean.squeeze(dim=-1)) / std.squeeze(dim=-1)   

def fractal_noise(x, mode='bicubic', exp=3):
    b, c, _, size = x.shape
    resolutions = [2**i for i in range(1, int(log2(size)))]    
    noise = 0    
    for i, res in enumerate(resolutions):
        octave = torch.randn(b, c, res, res, device=x.device)
        octave = F.interpolate(octave, (size, size), mode=mode)
        noise += octave.div(exp**i)        
    return normalize_batch(noise)

def noisify(clear, signal_ratio, noise_fn=fractal_noise):
    noise = noise_fn(clear)
    noisy = clear * signal_ratio + noise * (1- signal_ratio)
    return noisy, noise

def denoise(noisy, noise, signal_ratio):
    return (noisy - (noise * (1-signal_ratio))) / signal_ratio

