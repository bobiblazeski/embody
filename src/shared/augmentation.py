import torch
import torch.nn.functional as F

from torchvision.transforms import (    
    GaussianBlur,
)
from random import random 
from torch.distributions import Uniform
from perlin_numpy import generate_perlin_noise_3d,  generate_fractal_noise_3d

def blend_two(t, r, r_ratio=0.05):
    return (1-r_ratio) * t +  r_ratio * r

def center(t):
    return t - t.mean(dim=(-1, -2), keepdim=True)

def gaussian_blur(t):
    return GaussianBlur(3)(t)

def max_scale(t):
    return t / torch.max(t.abs())

def random_nth(t, n):
    su, sv = torch.randint(n, (2,)).tolist()
    return t[..., su::n, sv::n]

def resize_nearest(t, size):
    return F.interpolate(t, size)

def resize_bilinear(t, size):
    return F.interpolate(t, size, mode='bilinear', align_corners=True)

def resize_bicubic(t, size):
    return F.interpolate(t, size, mode='bicubic', align_corners=True)

def split_nth(t):
    return random_nth(random_split(t), 2)

def random_scale(t, scale=(0.9, 1.1)):
    shape = (3, 1, 1) if t.dim() == 3 else (t.size(0), 3, 1, 1)    
    return t * Uniform(*scale).sample(shape).to(t.device)

def random_shift(t, scale=(-0.1, +0.1)):
    shape = (3, 1, 1) if t.dim() == 3 else (t.size(0), 3, 1, 1)    
    return t + Uniform(*scale).sample(shape).to(t.device)

def random_execution(t, p, aug):
    return aug(t) if random() < p else t

def random_fork(t, p, aug0, aug1):
    return aug0(t) if random() < p else aug1(t)

def reflect(t):    
    return F.pad(t, (t.size(-1)-1, -(t.size(-1)-1), 0, 0), mode='reflect')

def perlin_noise(t, noise_ratio=0.05, res=(1, 4, 4)):    
    shape = (3, t.size(-1), t.size(-1))
    noise = generate_perlin_noise_3d(shape, res)
    noise = torch.tensor(noise, device=t.device, dtype=torch.float32)
    return (1-noise_ratio) * t + noise_ratio * noise

def fractal_noise(t, noise_ratio=0.05, res=(1, 2, 2), octaves=3):
    shape = (4, t.size(-1), t.size(-1))
    noise = generate_fractal_noise_3d(shape, res, octaves=octaves)
    noise = torch.tensor(noise, device=t.device, dtype=torch.float32)[:-1]
    return (1-noise_ratio) * t + noise_ratio * noise

def random_split(t):
    mn = t.size(-1)    
    
    u0 = t[:, :, :-1, :]
    u1 = t[:, :, 1:, :]    
    b, _, m, n = u0.shape
    pu = torch.rand(b, 1, m, n, device=t.device)
    um = u0* pu + u1 * (1-pu)
    uexp = F.interpolate(t, (mn+mn-1, mn), mode='nearest')
    uexp[:, :, 1::2, :] = um

    v0 = uexp[:, :, :, :-1]
    v1 = uexp[:, :, :, 1:]        
    b, _, m, n = v0.shape
    pv = torch.rand(b, 1, m, n, device=t.device)
    vm = v0* pv + v1 * (1-pv)
    uvexp = F.interpolate(uexp, (mn+mn-1, mn+mn-1), mode='nearest')  
    uvexp[:, :, :, 1::2] = vm
    
    uvexp = F.interpolate(uvexp, 2*mn, mode='bilinear')    
    return uvexp

