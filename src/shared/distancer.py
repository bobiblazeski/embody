import torch
import torch.nn as nn

class MeanDistancer(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.up = self.__create_selector(0, 1)
        self.left = self.__create_selector(1, 0)
        self.right = self.__create_selector(1, 2)
        self.down = self.__create_selector(2, 1)

    def __create_selector(_, m, n):
        conv = nn.Conv2d(3, 1, 3, groups=1, bias=False, padding=1, 
                        padding_mode='replicate')
        conv.requires_grad_(False)
        conv.weight.data.zero_()
        conv.weight.data[:, 0, m, n] = -0.25
        conv.weight.data[:, 0, 1, 1] = +0.25
        return conv

    def forward(self, t):
        ut, dt, lt, rt = [torch.abs(f(t)) for f 
            in [self.up, self.down, self.left, self.right]]      
        return ut + dt + lt + rt
    
class NeighborDistancer(nn.Module):
    def __init__(self):
        super().__init__()
        self.directions = nn.ModuleList([
            self.__create_selector(i, j)
            for i in range(3) 
            for j in  range(3) 
            if (i, j) != (1, 1)
        ])        

    def __create_selector(_, m, n):        
        conv = nn.Conv2d(3, 3, 3, groups=3, bias=False, padding=1, 
                        padding_mode='replicate')
        conv.requires_grad_(False)
        conv.weight.data.zero_()
        conv.weight.data[:, 0, m, n] = -1.
        conv.weight.data[:, 0, 1, 1] = +1.
        return conv

    def forward(self, t):
        return torch.cat([
            torch.abs(f(t)) for f 
            in self.directions
        ], dim=1)
    