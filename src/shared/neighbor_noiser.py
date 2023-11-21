import torch
import torch.nn as nn

class NeighborNoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = self.__create_selector(0, 1)
        self.left = self.__create_selector(1, 0)
        self.right = self.__create_selector(1, 2)
        self.down = self.__create_selector(2, 1)

    def __create_selector(_, m, n):
        conv = nn.Conv2d(3, 3, 3, groups=3, bias=False, padding=1, 
                        padding_mode='replicate')
        conv.requires_grad_(False)
        conv.weight.data.zero_()
        conv.weight.data[:, 0, m, n] = 1.   
        return conv

    def forward(self, t):
        up_t, down_t, left_t, right_t = [f(t) for f 
            in [self.up, self.down, self.left, self.right]]
        up_r, down_r, left_r, right_r = [torch.rand_like(t, device=t.device)
            for t in [up_t, down_t, left_t, right_t]]
        sum_r = up_r + down_r + left_r + right_r
        up_r, down_r, left_r, right_r = [r / sum_r 
            for r in [up_r, down_r, left_r, right_r]]
        res = up_t * up_r + down_t * down_r + left_t * left_r + right_t * right_r
        return res