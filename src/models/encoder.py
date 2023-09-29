from collections import OrderedDict
import torch.nn as nn
from src.models.blocks import ResidualBlock, NonLocalBlock, DownSampleBlock

class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()                
        channels = [args.image_channels] + args.encoder_channels
        head = nn.Sequential(*[            
            nn.Sequential(
                ResidualBlock(in_c, out_c, args.num_groups),
                DownSampleBlock(out_c),
            )
            for in_c, out_c in 
            zip(channels, channels[1:])
        ])
        tail = nn.Sequential(            
            NonLocalBlock(channels[-1], args.num_groups),
            ResidualBlock(channels[-1], channels[-1], args.num_groups), 
            nn.Conv2d(channels[-1], args.latent_dim, 3, 1, 1),
            nn.Conv2d(args.latent_dim, args.latent_dim, 1),        
        )
        self.model = nn.Sequential(OrderedDict({
            'head': head,
            'tail': tail,
        }))

    def forward(self, x):
        return self.model(x)
