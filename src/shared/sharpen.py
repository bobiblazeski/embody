
from collections import OrderedDict

import torch
import torch.nn as nn

class Sharpen(nn.Module):
    def __init__(self, mask=('sharpen_low', '5x5'), channels=3, padd=True):
        super().__init__()        
        kernel = torch.tensor(kernels[mask[0]][mask[1]])       
        self.padd = padd
        arr = [('padd', nn.ReplicationPad2d(kernel.size(-1) // 2))] if padd else []
        arr.append(('conv', nn.Conv2d(channels, channels, kernel.size(-1), 
            stride=1, padding=0, bias=False, groups=channels)),)
        self.seq = nn.Sequential(OrderedDict(arr))
        self.seq.requires_grad_(False)
        self.weights_init(kernel)

    def forward(self, x):
        return self.seq(x)

    def weights_init(self, kernel):
        for _, f in self.named_parameters():
            f.data.copy_(kernel)

sharpen = {
    '3x3_2': [
        [ 0.00,  0.25, 0.00],
        [ 0.25, -2.00, 0.25],
        [ 0.00,  0.25, 0.00],
    ],
    '3x3_3': [
        [ 0.0,  0.5, 0.0],
        [ 0.5, -3.0, 0.5],
        [ 0.0,  0.5, 0.0],
    ],
    '3x3_5': [
        [ 0.,  1., 0.],
        [ 1., -5., 1.],
        [ 0.,  1., 0.],
    ],
    '3x3_7': [
        [ 0.5,  1.0, 0.5],
        [ 1.0, -7.0, 1.0],
        [ 0.5,  1.0, 0.5],
    ],
    '3x3_9': [
        [ 1.,  1., 1.],
        [ 1., -9., 1.],
        [ 1.,  1., 1.],
    ],
    '5x5': [
        [ 1.,  1.,  1., 1., 1.],
        [ 1.,  1.,  1., 1., 1.],
        [ 1.,  1.,-25., 1., 1.],
        [ 1.,  1.,  1., 1., 1.],
        [ 1.,  1.,  1., 1., 1.],
    ],
    '7x7': [
        [ 1.,  1.,  1.,  1., 1., 1., 1.],
        [ 1.,  1.,  1.,  1., 1., 1., 1.],
        [ 1.,  1.,  1.,  1., 1., 1., 1.],
        [ 1.,  1.,  1.,-49., 1., 1., 1.],
        [ 1.,  1.,  1.,  1., 1., 1., 1.],
        [ 1.,  1.,  1.,  1., 1., 1., 1.],
        [ 1.,  1.,  1.,  1., 1., 1., 1.],
    ],
}
sharpen_low = {
    '3x3': [
        [ -1.,  -1., -1.],
        [ -1.,  16., -1.],
        [ -1.,  -1., -1.],
    ],
    '5x5': [
        [ -1, -3, -4, -3, -1],
        [ -3,  0,  6,  0, -3],
        [ -4,  6, 40,  6, -4],
        [ -3,  0,  6,  0, -3],
        [ -1, -3, -4, -3, -1],
    ],
    '7x7': [
        [ -2, -3, -4, -6, -4, -3, -2],
        [ -3, -5, -4, -3, -4, -5, -3],
        [ -4, -4,  9, 20,  9, -4, -4],
        [ -6, -3, 20, 72, 20, -3, -7],
        [ -4, -4,  9, 20,  9, -4, -4],
        [ -3, -5, -4, -3, -4, -5, -3],
        [ -2, -3, -4, -6, -4, -3, -2],
    ],
}

sobel_horizontal = {
    '3x3': [
        [ 1, 2, 1],
        [ 0, 0, 0],
        [-1,-2,-1],
    ],
    '5x5': [
        [ 1,  4,  7,  4, 1],
        [ 2, 10, 17, 10, 2],
        [ 0,  0,  0,  0, 0],
        [-2,-10,-17,-10,-2],
        [-1, -4, -7, -4,-1],
    ],
    '7x7': [
        [  1,   4,   9,  13,   9,   4,   1],
        [  3,  11,  26,  34,  26,  11,   3],
        [  3,  13,  30,  40,  30,  13,   3],
        [  0,   0,   0,   0,   0,   0,   0],
        [ -3, -13, -30, -40, -30, -13,  -3],
        [ -3, -11, -26, -34, -26, -11,  -3],
        [ -1,  -4,  -9, -13,  -9,  -4,  -1],
    ],    
}
sobel_vertical = {
    '3x3': [
        [ 1,  0, -1],
        [ 2,  0, -2],
        [ 1,  0, -1],
    ],
    '5x5': [
        [  1,   2,   0,  -2,  -1],
        [  4,  10,   0, -10,  -4],
        [  7,  17,   0, -17,  -7],
        [  4,  10,   0, -10,  -4],
        [  1,   2,   0,  -2,  -1],
    ],
    '7x7': [
       [  1,   3,   3,   0,  -3,  -3,  -1],
       [  4,  11,  13,   0, -13, -11,  -4],
       [  9,  26,  30,   0, -30, -26,  -9],
       [ 13,  34,  40,   0, -40, -34, -13],
       [  9,  26,  30,   0, -30, -26,  -9],
       [  4,  11,  13,   0, -13, -11,  -4],
       [  1,   3,   3,   0,  -3,  -3,  -1],
    ],   
}

laplacian = {
    '3x3': [
        [ -1.,  -1., -1.],
        [ -1.,   8., -1.],
        [ -1.,  -1., -1.],
    ],
    '5x5': [
        [ -1, -3, -4, -3, -1],
        [ -3,  0,  6,  0, -3],
        [ -4,  6, 20,  6, -4],
        [ -3,  0,  6,  0, -3],
        [ -1, -3, -4, -3, -1],
    ],
    '7x7': [
        [ -2, -3, -4, -6, -4, -3, -2],
        [ -3, -5, -4, -3, -4, -5, -3],
        [ -4, -4,  9, 20,  9, -4, -4],
        [ -6, -3, 20, 36, 20, -3, -7],
        [ -4, -4,  9, 20,  9, -4, -4],
        [ -3, -5, -4, -3, -4, -5, -3],
        [ -2, -3, -4, -6, -4, -3, -2],
    ],
}

custom = {
    '3x3_5': [
        [ 0.,  1., 0.],
        [ 1., -5., 1.],
        [ 0.,  1., 0.],
    ],
    
    '3x3_21': [
        [ 1.,   4., 1.],
        [ 4., -21., 4.],
        [ 1.,   4., 1.],
    ],
    '5x5_232': [
        [ 1.,  4.,    7.,  4., 1.],
        [ 4., 16.,   26., 16., 4.],
        [ 7., 26., -232., 26., 7.],
        [ 4., 16.,   26., 16., 4.],
        [ 1.,  4.,    7.,  4., 1.],
    ],    
}

kernels = {
    'sharpen': sharpen,
    'sharpen_low': sharpen_low,
    'sobel_horizontal': sobel_horizontal,
    'sobel_vertical': sobel_vertical,
    'laplacian':laplacian,
    'custom': custom,
    
}