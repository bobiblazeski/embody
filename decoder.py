import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import src.shared.util as U

class Head(nn.Module):    

    def __init__(self, p_sz, emb_sz, mid_sz, dropout=0.05):
        super().__init__()
        self.p_sz = p_sz
        self.proj_emb = nn.Linear(emb_sz, mid_sz, bias=False)
        self.proj_key = nn.Linear(3*p_sz**2, mid_sz, bias=False)
        
        self.net = nn.Sequential(            
            nn.Dropout(dropout),
            nn.ReLU(True),
            nn.Linear(2*mid_sz, 3*p_sz**2, bias=False),
        ) 

    def forward(self, x, emb):                
        emb = self.proj_emb(emb)
        u, v = x.size(-2) // self.p_sz, x.size(-1) // self.p_sz
        y = rearrange(x, 'b c (u m) (v n) -> b (u v) (c m n)',
                      m=self.p_sz, n=self.p_sz)        
        y = self.proj_key(y)        
        emb = emb.expand(-1, y.size(1), -1)
        y = y.expand(emb.size(0), -1, -1)        
        y = torch.cat([emb, y], dim=-1)        
        y = self.net(y)
        return rearrange(y, 'b (u v) (c m n) -> b c (u m) (v n)',
                         c=3, u=u, v=v, m=self.p_sz, n=self.p_sz)
        

class MultiHead(nn.Module):
    def __init__(self, p_sz, e_sz, mid_sz, sizes):
        super().__init__()
        self.p_sz = p_sz        
        self.sizes = sizes
        self.heads = nn.ModuleList([
            Head(p_sz, e_sz, mid_sz) 
            for _ in sizes])

    def forward(self, emb, mean):        
        layers = [mean]        
        for i, (head, sz) in enumerate(zip(self.heads, self.sizes)):                        
            o = F.interpolate(mean, self.p_sz * sz, mode='bilinear')
            y = head(o, emb)
            layers.append(y)
        return U.join(layers)

class Decoder(nn.Module):
    def __init__(self, p_sz, e_sz, mid_sz, blocks):
        super().__init__()
        self.blocks = nn.ModuleList([
            MultiHead(p_sz, e_sz, mid_sz, sizes)
            for sizes  in blocks
        ])
    
    def forward(self, emb, mean):
        emb = emb.mean(dim=1, keepdim=True)        
        res = []
        for block in self.blocks:
            mean = block(emb, mean)
            res.append(mean)
        return res