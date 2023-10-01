# pyright: reportMissingImports=false
import math
import trimesh
import torch
import torch.nn.functional as F

from src.shared.faces import make_faces

def scale(t, size, mode):    
    if t.dim() == 3:
        return F.interpolate(t[None], size=size, mode=mode)[0]
    return F.interpolate(t, size=size, mode=mode)


def to_vertices(t):
    dims = len(t.shape)
    if dims == 3:
        return t.permute(2, 1, 0).reshape(-1, 3)
    elif dims == 4:
        bs = t.size(0)
        return t.permute(0, 3, 2, 1).reshape(bs, -1, 3)
    raise Exception("No of dimension must be 3 or 4 not", dims)

def to_cylinder(t):
    dims = len(t.shape)
    if dims == 2:
        n = int(math.sqrt(t.size(0)))    
        return t.reshape(n, n, 3).permute(2, 1, 0)
    elif dims == 3:
        n = int(math.sqrt(t.size(1)))
        bs = t.size(0)
        return t.reshape(bs, n, n, 3).permute(0, 3, 2, 1)
    raise Exception("No of dimension must be 2 or 3 not", dims)

def load_mesh(filename, device=None, flip=[1., 1., -1.], batch=False):
    '''Load, Scale, center & permute mesh'''
    mesh = trimesh.load_mesh(filename)
    faces = torch.tensor(mesh.faces).int()
    v = torch.tensor(mesh.vertices).float()
    v = v * torch.tensor([flip])    
    v = v - v.mean(dim=0, keepdim=True)
    v = v / v.abs().max()
    v = torch.stack([v[:, 1], v[:, 2], v[:, 0]], dim=1)
    if batch: 
        v = v[None]
    return (v.to(device), faces.to(device)) if device else (v, faces)

def export_stl(t, file, faces=None, flip=None):
    if faces is None:
        _, m, n = t.size()
        v = to_vertices(t.detach().cpu())
        faces = make_faces(m, n)
    else:
        v, faces = t.detach().cpu(), faces.cpu()    
    if flip is not None:
        v *= torch.tensor(flip, device=v.device)
    mesh1 = trimesh.Trimesh(vertices=v, faces=faces)
    file = file if file.endswith('.stl') else file + '.stl'
    mesh1.export(file, file_type='stl'); 

def export_offsets(offsets, location, prefix):
    for i in range(len(offsets)-1):    
        joined = join(offsets[:-i] if i else offsets)
        export_stl(joined[0], location + prefix + str(joined.size(-1)))

def split(t, sizes=None):
    res, size = [], t.size(-1)
    if sizes is None:
        sizes = [2**i for i in range(int(math.log2(size))+1)]
    for sz in sizes:
        layer =  F.adaptive_avg_pool2d(t, sz)        
        res.append(layer)
        t = t - F.interpolate(layer, size, mode='nearest')
    return res  

def join(layers):
    res, size = 0, layers[-1].size(-1)
    for l in layers:
        res = res + F.interpolate(l, size, mode='nearest')
    return res

def noise(t, ratio=0.05):
    mean= t.mean(dim=(-2, -1), keepdim=True)
    std = t.std(dim=(-2, -1), keepdim=True)
    noise = mean + torch.randn_like(t) * std
    return ratio * noise + (1-ratio) *t
    
def steps(offsets):
    res, device  = [], offsets[0].device
    step =torch.tensor([0])[None, :, None, None].to(device)
    for o in offsets:        
        step = o + F.interpolate(step, o.size(-1))
        res.append(step)
    return res

def add_fourier(t, terms, base=1e4, input=True, inverse=True, divide=True):
    freq = base**torch.arange(0, terms, device=t.device).float() 
    if divide:
        freq = freq / terms
    if inverse: 
        freq = 1.0 / freq
    freq = freq.repeat(t.size(1))[None, :, None, None]    
    expanded = t.repeat(1, terms, 1, 1) * freq    
    encodings = [torch.sin(expanded), torch.cos(expanded)]
    if input: 
        encodings.insert(0, t)    
    return torch.cat(encodings, dim=1)

def is_custom_kernel_supported():
    version_str = str(torch.version.cuda).split(".")
    major = version_str[0]
    minor = version_str[1]
    return int(major) >= 10 and int(minor) >= 1    

def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)
    if k.dim() == 1:
        k = k[None, :] * k[:, None]
    k /= k.sum()
    return k
