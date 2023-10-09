# pyright: reportMissingImports=false
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride):
        super(ConvBlock,self).__init__()
        self.add_module('conv',nn.Conv2d(in_channel ,out_channel,kernel_size=ker_size,stride=stride,padding=padd)),
        self.add_module('norm',nn.BatchNorm2d(out_channel)),
        #self.add_module('LeakyRelu',nn.LeakyReLU(0.2, inplace=False))
        #self.add_module('SiLU',nn.SiLU(inplace=True))
        self.add_module('ReLU',nn.ReLU(inplace=False))

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('Norm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        
class Generator(nn.Module):
    def __init__(self, nfc=128, nc_im=3, num_layer=5):
        super(Generator, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        self.num_layer = num_layer
        self.head = ConvBlock(nc_im, nfc, 3, 0, 1)
        self.body = nn.Sequential()
        for i in range(num_layer-2):            
            block = ConvBlock(nfc, nfc, 3,0,1)
            self.body.add_module('block%d'%(i+1),block)
        self.tail = nn.Sequential(
            nn.Conv2d(nfc, nc_im, kernel_size=3, stride=1, padding=0),
            #nn.Tanh()
        )
    def forward(self, x):
        _, _, sz_h, sz_v = x.shape
        size = (sz_h+2*self.num_layer, sz_v+2*self.num_layer)
        x = F.interpolate(x, size=size,mode='bilinear')
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)        
        return x
