import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modules.equal_linear import EqualLinear

class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        demodulate=True,        
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel        

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate
        self.new_demodulation = True

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, '            
        )

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape

        if style.dim() > 2:
            style = F.interpolate(style, size=(input.size(2), input.size(3)), mode='bilinear', align_corners=False)
            style = self.modulation(style).unsqueeze(1)
            if self.demodulate:
                style = style * torch.rsqrt(style.pow(2).mean([2], keepdim=True) + 1e-8)
            input = input * style
            weight = self.scale * self.weight
            weight = weight.repeat(batch, 1, 1, 1, 1)
        else:
            style = style.view(batch, style.size(1))
            style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
            if self.new_demodulation:
                style = style[:, 0, :, :, :]
                if self.demodulate:
                    style = style * torch.rsqrt(style.pow(2).mean([1], keepdim=True) + 1e-8)
                input = input * style
                weight = self.scale * self.weight
                weight = weight.repeat(batch, 1, 1, 1, 1)
            else:
                weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )
        

        
        input = input.view(1, batch * in_channel, height, width)
        out = F.conv2d(input, weight, padding=self.padding, groups=batch)
        _, _, height, width = out.shape
        out = out.view(batch, self.out_channel, height, width)

        return out