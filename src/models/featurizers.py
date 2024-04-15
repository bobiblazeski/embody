import torch
import torch.nn as nn
import torchvision

class Vgg16Features(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.layers = hp.vgg16_layers
        net = torchvision.models.vgg16(weights=hp.vgg16_weights).features
        if hp.vgg16_file: 
            net.load_state_dict(torch.load(hp.vgg16_file))        
        self.net = nn.ModuleList([
            nn.ReLU() if isinstance(l, nn.ReLU) else l
            for l in net.children()
        ])
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        ) if hp.vgg16_pool else nn.Identity()        
        for param in self.parameters():
            param.requires_grad_(hp.vgg16_requires_grad)

    def forward(self, x):
        layers =  self.layers        
        res, max_layer = [], max(layers) if layers else -1
        with torch.no_grad():
            for i, l in enumerate(self.net):
                if i > max_layer: break
                x = l(x)
                if i in layers: res.append(self.pool(x))        
        return res