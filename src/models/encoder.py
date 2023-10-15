import torch.nn as nn
from src.models.blocks import (  
  DepthwiseSeparable2d,
  ResidualStack,
)

class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, embedding_dim):
        super().__init__()
        self.net = nn.Sequential(
            DepthwiseSeparable2d(in_channels, num_hiddens//2,  kernel_size=4,
                                 stride=2, padding=1),
            nn.ReLU(True),
            DepthwiseSeparable2d(num_hiddens//2, num_hiddens,  kernel_size=4,
                                 stride=2, padding=1),
            nn.ReLU(True),
            DepthwiseSeparable2d(num_hiddens, num_hiddens,  kernel_size=3,
                                 stride=1, padding=1),
            ResidualStack(in_channels=num_hiddens,
                          num_hiddens=num_hiddens,
                          num_residual_layers=num_residual_layers,
                          num_residual_hiddens=num_residual_hiddens),
            DepthwiseSeparable2d(num_hiddens, embedding_dim,  kernel_size=1,
                                 stride=1, padding=0),            
        )        

    def forward(self, x):        
        return self.net(x)