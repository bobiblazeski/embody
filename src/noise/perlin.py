import math
import torch
import torch.nn as nn

from noise.noise_util import strength

def interp(t):
    return 3 * t**2 - 2 * t ** 3
    
def perlin_noise(width, height, scale=10):
    gx, gy = torch.randn(2, width + 1, height + 1, 1, 1)
    xs = torch.linspace(0, 1, scale + 1)[:-1, None]
    ys = torch.linspace(0, 1, scale + 1)[None, :-1]

    wx, wy = 1 - interp(xs), 1 - interp(ys)
    
    dots = wx * wy * (gx[:-1, :-1] * xs + gy[:-1, :-1] * ys)
    dots += (1 - wx) * wy * (-gx[1:, :-1] * (1 - xs) + gy[1:, :-1] * ys)
    dots += wx * (1 - wy) * (gx[:-1, 1:] * xs - gy[:-1, 1:] * (1 - ys))
    dots += (1 - wx) * (1 - wy) * (-gx[1:, 1:] * (1 - xs) - gy[1:, 1:] * (1 - ys))    
    return dots.permute(0, 2, 1, 3).contiguous().view(width * scale, height * scale)

def perlin_fractal(size, normalized=True, exp=None, cut_last=1):
    scales = [2**i for i in range(int(math.log2(size)), 0, -1)]
    scales = scales[:-cut_last]    
    strengths = strength(len(scales), exp=exp)    
    noise = 0 
    for scale, s in zip(scales, strengths):
        p =  perlin_noise(size // scale, size // scale, scale)
        noise += s * p
    if normalized:
        noise -= noise.mean()
        noise /= noise.abs().max()
    return noise

class Perlin(nn.Module):
    def __init__(self, width, height, scale=10):
        super().__init__()
        self.width = width
        self.height = height
        self.scale = scale
        self.gx = nn.Parameter(torch.randn(3, width + 1, height + 1, 1, 1))
        self.gy = nn.Parameter(torch.randn(3, width + 1, height + 1, 1, 1))

        xs = torch.linspace(0, 1, scale + 1)[:-1, None]
        ys = torch.linspace(0, 1, scale + 1)[None, :-1]
        self.register_buffer('xs', xs)
        self.register_buffer('ys', ys)
        self.register_buffer('wx', 1 - self.interp(xs))
        self.register_buffer('wy', 1 - self.interp(ys))

    def interp(_, t):
        return 3 * t**2 - 2 * t ** 3
        
    def forward(self):
        gx, gy, xs, ys, wx, wy = self.gx, self.gy, self.xs, self.ys, self.wx, self.wy
        dots = wx * wy * (gx[:, :-1, :-1] * xs + gy[:, :-1, :-1] * ys)
        dots += (1 - wx) * wy * (-gx[:, 1:, :-1] * (1 - xs) + gy[:, 1:, :-1] * ys)
        dots += wx * (1 - wy) * (gx[:, :-1, 1:] * xs - gy[:, :-1, 1:] * (1 - ys))
        dots += (1 - wx) * (1 - wy) * (-gx[:, 1:, 1:] * (1 - xs) - gy[:, 1:, 1:] * (1 - ys))        
        res = dots.permute(0, 1, 3, 2, 4).contiguous()
        return res.view(3, self.width * self.scale, self.height * self.scale)