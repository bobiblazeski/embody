import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class ResNet50Features(nn.Module):
    layers =  (0, 1, 2, 3, 4)
    def __init__(self, weights=None, file=None):
        super().__init__()
        net = torchvision.models.resnet50(weights=weights)
        if file:
            net.load_state_dict(torch.load(file))        
        self.net = nn.Sequential(
            nn.Sequential(
               net.conv1,
               net.bn1,
               nn.ReLU(inplace=False),
               net.maxpool,
            ),
            net.layer1, 
            net.layer2,
            net.layer3,
            net.layer4,
        )
        # for param in self.net.parameters():
        #     param.requires_grad_(False)

    def forward(self, x, layers=None, pool=True):
        layers = layers or ResNet50Features.layers
        res, max_layer = [], max(layers) if layers else -1 
        #with torch.no_grad():            
        for i, l in enumerate(self.net):
            if i > max_layer: break
            x = l(x)
            if i in layers: res.append(x.clone())
        if pool:
            res = [F.adaptive_avg_pool2d(f, (2, 2)).flatten(1) for f in res]
        return res

class Vgg16Features(nn.Module):
    def __init__(self, weights=None, file=None):
        super().__init__()
        net = torchvision.models.vgg16(weights=weights).features
        if file: 
            net.load_state_dict(torch.load(file))
        # for param in net.parameters():
        #     param.requires_grad_(False)
        self.net = nn.ModuleList([
            nn.ReLU() if isinstance(l, nn.ReLU) else l
            for l in net.children()
        ])

    def forward(self, x, layers=()):
        res, max_layer = [], max(layers) if layers else -1 
        #with torch.no_grad():            
        for i, l in enumerate(self.net):
            if i > max_layer: break
            x = l(x)
            if i in layers: res.append(x)
        return res

class AlexNetFeatures(nn.Module):
    def __init__(self, weights=None, file=None):
        super().__init__()
        self.net = torchvision.models.alexnet(weights=weights).features
        if file: 
            self.net.load_state_dict(torch.load(file))
        for param in self.net.parameters():
            param.requires_grad_(False)

    def forward(self, x, layers=()):
        res, max_layer = [], max(layers) if layers else -1 
        #with torch.no_grad():            
        for i, l in enumerate(self.net):
            if i > max_layer: break
            x = l(x)
            if i in layers: res.append(x)
        return res

class FeatureDecoder(nn.Module):
    def __init__(self, channels, middle, size,  mode='bilinear'):
        super().__init__()
        self.size = size
        self.mode = mode
        self.channels_no = sum(channels)+3
        self.net = nn.Sequential(
            nn.Conv2d(self.channels_no, middle, 3, 1, 1),
            nn.LeakyReLU(0.05),
            nn.Conv2d(middle, middle, 3, 1, 1),
            nn.LeakyReLU(0.05),
            nn.Conv2d(middle, 3, 3, 1, 1),
            nn.LeakyReLU(0.05),
        )

    def scale(self, t):
        return F.interpolate(t, size=self.size, mode=self.mode)

    def cat(self, l):
        return torch.cat([self.scale(e) for e in l], dim=1)

    def  forward(self, x, l):        
        y = self.cat([x] + l)
        y = self.net(y)
        return x + y