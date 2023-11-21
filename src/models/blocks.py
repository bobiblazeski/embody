import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def toOut(in_ch, out_ch):
    return nn.Sequential(
        DepthwiseSeparable2d(in_ch, in_ch),
        nn.ReLU(True),
        DepthwiseSeparable2d(in_ch, out_ch),        
    )

class Residual(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            DepthwiseSeparable2d(in_ch, out_ch),
            nn.BatchNorm2d(out_ch),         
            nn.ReLU(True),
            DepthwiseSeparable2d(out_ch, out_ch),
        )
        self.proj = nn.Identity() if in_ch == out_ch else DepthwiseSeparable2d(in_ch, out_ch)
        
    def forward(self, x):        
        return self.proj(x) + self.block(x) 

class DepthwiseSeparable2d(nn.Module):
    def __init__(self, in_ch, out_ch, bias=False, kernel_size=3, stride=1, padding=1, padding_mode='replicate'):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, stride, padding, padding_mode=padding_mode, groups=in_ch, bias=bias)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
    
class DSBlock(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.main = nn.Sequential(*[            
            DepthwiseSeparable2d(in_ch, mid_ch,  kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            DepthwiseSeparable2d(mid_ch, mid_ch,  kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            DepthwiseSeparable2d(mid_ch, out_ch,  kernel_size=3, stride=1, padding=1),  
        ])        
        
    def forward(self, x):                
        return self.main(x)    
       
class DoubleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()        
        self.conv = DepthwiseSeparable2d(in_channels, out_channels)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0)
        return self.conv(x)


class NonLocalBlock(nn.Module):
    def __init__(self, channels, num_groups):
        super(NonLocalBlock, self).__init__()
        self.in_channels = channels

        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=channels,
                               eps=1e-6, affine=True)
        self.q = nn.Conv2d(channels, channels, 1, 1, 0)
        self.k = nn.Conv2d(channels, channels, 1, 1, 0)
        self.v = nn.Conv2d(channels, channels, 1, 1, 0)
        self.proj_out = nn.Conv2d(channels, channels, 1, 1, 0)

    def forward(self, x):
        h_ = self.gn(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape

        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)
        k = k.reshape(b, c, h*w)
        v = v.reshape(b, c, h*w)

        attn = torch.bmm(q, k)
        attn = attn * (int(c)**(-0.5))
        attn = F.softmax(attn, dim=2)
        attn = attn.permute(0, 2, 1)

        A = torch.bmm(v, attn)
        A = A.reshape(b, c, h, w)

        return x + A


class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)
        q = q * self.scale

        k = k.softmax(dim = -1)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)
    
class PositionalEncoding(nn.Module):
    def __init__(self, max_time_steps: int, embedding_size: int, n: int = 10000) -> None:
        super().__init__()

        i = torch.arange(embedding_size // 2)
        k = torch.arange(max_time_steps).unsqueeze(dim=1)

        self.pos_embeddings = torch.zeros(max_time_steps, embedding_size, requires_grad=False)
        self.pos_embeddings[:, 0::2] = torch.sin(k / (n ** (2 * i / embedding_size)))
        self.pos_embeddings[:, 1::2] = torch.cos(k / (n ** (2 * i / embedding_size)))

    def forward(self, t):
        return self.pos_embeddings[t, :]    











