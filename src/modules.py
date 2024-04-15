import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
                    
class DS2d(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, 
                 padding_mode='replicate', **kwargs):
        super().__init__()
        self.append(nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, 
            padding=padding, padding_mode=padding_mode, **kwargs))
        self.append(nn.Conv2d(in_ch, out_ch, 1))

class Conv(nn.Sequential):
    def __init__(self, in_ch, out_ch, **kwargs):
        super().__init__()
        self.append(DS2d(in_ch, out_ch, **kwargs))
        self.append(nn.BatchNorm2d(out_ch))
        self.append(nn.SiLU(True))

class TwoConv(nn.Sequential):
    def __init__(self, in_ch, out_ch, mid_ch=None, **kwargs):
        super().__init__()
        mid_ch = mid_ch or out_ch
        self.append(Conv(in_ch, mid_ch, **kwargs))
        self.append(Conv(mid_ch, out_ch, **kwargs))
        
class Residual(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None, **kwargs):     
        super().__init__()
        self.block = TwoConv(in_ch, out_ch, **kwargs)
        self.proj = (nn.Identity() if in_ch == out_ch
                     else DS2d(in_ch, out_ch, **kwargs))
        
    def forward(self, x):
        return self.block(x) + self.proj(x)

class Sinusoidal(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)        
        return embeddings

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered
