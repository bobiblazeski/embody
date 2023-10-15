import torch.nn as nn
from src.models.blocks import (
    DepthwiseSeparable2d,
    DoubleBlock,
    ResidualStack,   
  
)

class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super().__init__()
        self.net = nn.Sequential(
            DepthwiseSeparable2d(in_channels, num_hiddens,  kernel_size=3,
                                 stride=1, padding=1),
            ResidualStack(in_channels=num_hiddens,
                          num_hiddens=num_hiddens,
                          num_residual_layers=num_residual_layers,
                          num_residual_hiddens=num_residual_hiddens),
            DoubleBlock(num_hiddens, num_hiddens//2),
            nn.ReLU(True),
            DoubleBlock(num_hiddens//2, 3),            
        )        

    def forward(self, x):        
        return self.net(x)