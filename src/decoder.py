import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modules import (DS2d, Residual)

class Decoder(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, e_sizes, mean=None):
        super().__init__()
        self.register_buffer('mean', mean or torch.ones(1, 3, in_ch, in_ch))
        self.head = nn.ModuleList(
            nn.Sequential(
                nn.Linear(e_sz, e_sz//2),            
                nn.ReLU(True),            
                nn.Linear(e_sz//2, 3*in_ch**2),                        
                nn.Unflatten(1, (3, in_ch, in_ch)),
            ) for e_sz in e_sizes)
        self.tail = nn.Sequential(
            Residual(3 * (len(e_sizes)+1), out_ch),
            Residual(out_ch, mid_ch // 2),
            Residual(mid_ch // 2, mid_ch),
            Residual(mid_ch, mid_ch // 2),
            Residual(mid_ch // 2, out_ch),
            DS2d(out_ch, 3),
        )

    def forward(self, emb):
        x = self.mean.repeat(emb.size(0), 1, 1, 1)
        emb = [emb[:,i] for i in range(emb.size(1))] 
        emb = [h(e) for h, e in zip(self.head, emb)]        
        emb = torch.cat([x] + emb, dim=1)                
        return self.tail(emb) + x