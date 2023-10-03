from PIL import Image
import torch
from torchvision.transforms import (    
    Compose,
    GaussianBlur,   
    Normalize,
    Lambda,
    Resize, 
    ToTensor,    
)
from src.shared.util import scale

# Image net settings
img_res =  160 
img_mean = (0.485, 0.456, 0.406)
img_std = (0.229, 0.224, 0.225)

def channel_first(t):
    t = t[..., :3]
    if t.dim() == 3:
        return t.permute(2, 0, 1)
    return t.permute(0, 3, 1, 2)

def channel_last(t):    
    if t.dim() == 3:
        return t.permute(1, 2, 0)
    return t.permute(0, 2, 3, 1)    
    
render_as_image = Compose([ 
    Lambda(channel_first),    
    Resize((img_res, img_res), antialias=True), 
    Normalize(img_mean, img_std),           
])

image_transform = Compose([   
    ToTensor(),    
    Resize((img_res, img_res), antialias=True), 
    Normalize(img_mean, img_std),    
])

def load_image(file):
    img = Image.open(file)
    return image_transform(img)

def center_maxscale(t, patch=True):
    if patch:
        t = t - t.mean(dim=(-1, -2), keepdim=True) # center
        t = t / t.abs().amax(dim=(-3, -2, -1), keepdim=True)
    else:
        t = t - t.mean(dim=-2, keepdim=True) # center
        t = t / t.abs().amax(dim=(-2, -1), keepdim=True)
    return t


def patch_transform(t, size, mode, blur=None):
    t = t * torch.tensor([1.,-1.,1.])[:, None, None].to(t.device)    
    t = center_maxscale(t)
    if blur:
        kernel, sigma = blur        
        t = GaussianBlur(kernel, sigma)(t)
    t = scale(t, size, mode)
    return t