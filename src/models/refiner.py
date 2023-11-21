import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from pytorch_wavelets import DWTForward, DWTInverse 

from src.models.blocks import DSBlock

        
class Refiner(nn.Module):
    def __init__(self, ch_no, in_ch=3, wave='db2', mode='zero'):
        super().__init__()
        self.in_ch = in_ch
        self.net  = DSBlock(4*in_ch, ch_no, 4*in_ch)        
        
        self.xfm = DWTForward(J=1, mode=mode, wave=wave)
        self.ifm = DWTInverse(mode=mode, wave=wave)

    def split(self, x):
        coarse, fine = self.xfm(x)
        fine = rearrange(fine[0], 'b c d w h -> b (c d) w h')         
        return torch.cat([coarse, fine], dim=1)

    def join(self, y):
        b, _, m, n = y.shape
        coarse, fine = y[:, :self.in_ch], y[:, self.in_ch:]
        fine = [fine.view(b, self.in_ch, 3, m, n)]
        return self.ifm((coarse, fine))
        
    def forward(self, x):
        x = self.split(x)        
        x = self.net(x)
        x = self.join(x)
        return x
    