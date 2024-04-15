import torch
import torch.nn as nn
from torchvision.ops.misc import Conv2dNormActivation as conv
from torchvision.transforms import GaussianBlur

class EncodeLayer(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv0 = conv(in_ch, out_ch, 3, 1, 1, activation_layer=nn.LeakyReLU)
        self.conv1 = conv(out_ch, out_ch, 3, 2, 1, activation_layer=nn.LeakyReLU)
        self.blur = GaussianBlur(3, 0.5)

    def forward(self, x):
        y = self.conv0(x)
        y = self.blur(y)
        y = self.conv1(y)
        return y

class Encoder(torch.nn.Module):
    def __init__(self, layers, style_dim):
        super().__init__()
        self.extractor = nn.Sequential(*[
            EncodeLayer(ic, oc)
            for ic, oc
            in zip(layers, layers[1:])
        ])
        self.encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(layers[-1], style_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(style_dim, style_dim),
        )      

    def forward(self, x):
        y = self.extractor(x)        
        y = self.encoder(y)
        return y
