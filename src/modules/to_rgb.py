import torch
import torch.nn as nn


from src.shared.util import make_kernel
from src.modules.modulated_conv2d import ModulatedConv2d

class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()        
        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style):
        out = self.conv(input, style)
        out = out + self.bias        
        return out