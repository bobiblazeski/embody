import torch
from pytorch_wavelets import DWTForward, DWTInverse 

from src.modules.to_rgb import ToRGB
from src.modules.styled_conv import StyledConv

class Refiner(torch.nn.Module):
    def __init__(self, in_channels, out_channels, style_dim):
        super().__init__()
        self.conv1 = StyledConv(in_channels, out_channels, 3, style_dim)
        self.conv2 = StyledConv(out_channels, out_channels, 3, style_dim)
        self.toRGB = ToRGB(out_channels, style_dim)

    def forward(self, x, style):
        y = self.conv1(x, style)
        y = self.conv2(y, style)
        y = self.toRGB(y, style)
        return x + y

class RefinerDWT(torch.nn.Module):
    def __init__(self, channels, style_dim, wave='db1'):
        super().__init__()
        self.conv1 = StyledConv(12, channels, 3, style_dim)
        self.conv2 = StyledConv(channels, channels, 3, style_dim)
        self.conv3 = StyledConv(channels, 12, 3, style_dim)
        
        self.xfm = DWTForward(J=1, mode='zero', wave=wave)
        self.ifm = DWTInverse(mode='zero', wave=wave)

    def split(self, x):
        coarse, fine = self.xfm(x)
        b, f, c, m, n = fine[0].shape
        fine = fine[0].view(b, f*c, m, n)        
        return torch.cat([coarse, fine], dim=1)

    def join(self, y):
        b, _, m, n = y.shape
        coarse, fine = y[:, :3], y[:, 3:]
        fine = [fine.view(b, 3, 3, m, n)]
        return self.ifm((coarse, fine))
        
    def forward(self, x, style):
        y = self.split(x)

        y = self.conv1(y, style)
        y = self.conv2(y, style)
        y = self.conv3(y, style)
        y = self.join(y)
        return y    