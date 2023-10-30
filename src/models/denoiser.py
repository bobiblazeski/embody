import math
import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward, DWTInverse 
import torch.nn.functional as F
from src.models.blocks import (
    toOut,
    LinearAttention,
    Residual,   
)

class CoarseBlock(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, time_emb_dim):
        super().__init__()        
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, mid_ch),
            nn.ReLU(True),
            nn.Unflatten(1, (mid_ch, 1, 1)),
        )         
        self.head = Residual(2*in_ch, mid_ch)        
        self.attn = LinearAttention(mid_ch)
        self.tail = Residual(mid_ch, mid_ch)
        self.noise = toOut(mid_ch, out_ch)        
        
    def forward(self, x, lvl, t):
        x = F.interpolate(x, lvl.shape[-2:])
        y = torch.cat([x, lvl], dim=1)
        y = self.head(y)
        t = self.time_mlp(t)
        y = self.attn(y)
        y = self.tail(y+t)
        noise = self.noise(y) + lvl[:, :3]        
        return y, noise
        
class FineBlock(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, time_emb_dim):
        super().__init__()        
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, in_ch),
            nn.ReLU(True),
            nn.Unflatten(1, (in_ch, 1, 1)),
        )
        self.head = Residual(in_ch, mid_ch)   
        self.tail = Residual(mid_ch, mid_ch)                        
        self.noise = toOut(mid_ch, out_ch)        
        
    def forward(self, y, lvl, t):
        t = self.time_mlp(t)
        b, c, d, m, n = lvl.shape        
        y = F.interpolate(y, (m, n))
        y = torch.cat([y, lvl.view(b, c*d, m, n)], dim=1)        
        y = self.head(y+t)        
        y = self.tail(y)
        lvl = lvl[:, :3]         
        noise = self.noise(y).view_as(lvl) + lvl
        return y, noise
        

class SinusoidalPositionEmbeddings(nn.Module):
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
        
class Denoiser(nn.Module):
    def __init__(self, input_ch=3, output_ch=3, channels=(32, 64, 128, 256),
                 time_emb_dim=32, wave='db2', levels=3, padding_mode='zero'):
        super().__init__()      
        self.fwt = DWTForward(J=levels, mode=padding_mode, wave=wave)
        self.rwt = DWTInverse(mode=padding_mode, wave=wave)
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(True),
        )
        self.head = CoarseBlock(input_ch, channels[0], output_ch, time_emb_dim)
        self.blocks = nn.ModuleList([
            FineBlock(in_ch+input_ch*3, out_ch, output_ch*3, time_emb_dim)
            for in_ch, out_ch in zip(channels, channels[1:])
        ])
       
    def forward(self, x, timestep):        
        coarse, fines = self.fwt(x)
        t = self.time_mlp(timestep.view(-1))
        y, coarse_noise = self.head(x, coarse, t)
        noises = []
        for block, fine in zip(self.blocks, fines[::-1]):
            y, noise = block(y, fine, t)
            noises.insert(0, noise)        
        noise = self.rwt((coarse_noise, noises))
        return noise