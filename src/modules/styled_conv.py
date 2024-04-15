import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modules.modulated_conv2d import ModulatedConv2d

class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))
        self.fixed_noise = None
        self.image_size = None

    def forward(self, image, noise=None):
        if self.image_size is None:
            self.image_size = image.shape

        if noise is None and self.fixed_noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()
        elif self.fixed_noise is not None:
            noise = self.fixed_noise
            # to avoid error when generating thumbnails in demo
            if image.size(2) != noise.size(2) or image.size(3) != noise.size(3):
                noise = F.interpolate(noise, image.shape[2:], mode="nearest")
        else:
            pass  # use the passed noise

        return image + self.weight * noise
    
class StyledConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,        
        demodulate=True,
        use_noise=True,        
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,            
            demodulate=demodulate,
        )

        self.use_noise = use_noise
        self.noise = NoiseInjection()        
        self.activate = nn.LeakyReLU(0.2)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        if self.use_noise:
            out = self.noise(out, noise=noise)        
        out = self.activate(out)
        return out