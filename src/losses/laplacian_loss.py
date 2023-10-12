from collections import OrderedDict

import torch
import torch.nn as nn

class LaplacianLoss(nn.Module):
    def __init__(self, channels=3, padd=True, normalize=True):
        super().__init__()
        self.padd = padd
        arr = [('padd', nn.ReplicationPad2d(1))] if padd else []
        arr.append(('conv', nn.Conv2d(channels, channels, 3, 
            stride=1, padding=0, bias=False, groups=channels)),)
        self.seq = nn.Sequential(OrderedDict(arr))
        self.seq.requires_grad_(False)
        self.weights_init(normalize)

    def forward(self, x, reduced=False):
        if not self.padd and x.size(-1) < 3:
            return torch.tensor(0).to(x.device)
        seq = self.seq(x)
        return seq.abs().mean() if reduced else seq

    def weights_init(self, normalize):
        w = torch.tensor([[ 1.,   4., 1.],
                          [ 4., -20., 4.],
                          [ 1.,   4., 1.],])
        if normalize:
            w = w.div(20.)
        for _, f in self.named_parameters():           
            f.data.copy_(w)
