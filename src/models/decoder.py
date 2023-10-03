from collections import OrderedDict
import torch.nn as nn
from src.models.blocks import ResidualBlock, NonLocalBlock, UpSampleBlock


class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        channels = args.decoder_channels
        head = nn.Sequential(
            nn.Conv2d(args.latent_dim, args.latent_dim, 1),
            nn.Conv2d(args.latent_dim, channels[0], 3, 1, 1),
            NonLocalBlock(channels[0], args.num_groups),            
        )
        tail = nn.Sequential(*[            
            (nn.Sequential(
                UpSampleBlock(in_c),
                ResidualBlock(in_c, out_c, args.num_groups),
             ) if args.decoder_upsample
             else ResidualBlock(in_c, out_c, args.num_groups))
            for in_c, out_c in 
            zip(channels, channels[1:])
        ])
        tail.append(nn.Conv2d(channels[-1], args.image_channels, 3, 1, 1))
        self.model = nn.Sequential(OrderedDict({
            'head': head,
            'tail': tail,
        }))

    def forward(self, x):
        return self.model(x)
