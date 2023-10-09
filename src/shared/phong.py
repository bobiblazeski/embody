import math
import torch
import torch.nn.functional as F
from kaolin.render.camera import Camera
import nvdiffrast.torch as dr 
from src.shared.transforms import img_res

import src.shared.normals as NRM


def sample_light_colors(n, ratio=0.9, high=10000):    
    return torch.randint(int(ratio*high), high, (n, 1, 3)).div(high)

def sample_light_directions(n, ratio=0.9, standard=(0.0, 0.0, -1.0)):
    direction_std = torch.tensor(standard)[None, :, None]
    direction_rnd = F.normalize(torch.randn(n, 3, 1), dim=0)
    
    directions = ratio * direction_std + (1-ratio) * direction_rnd
    return F.normalize(directions, dim=-1)

def sample_views(n, x=(-0.05, 0.05), y=(-0.05, 0.05), z=(1.5, 1.6)):
    eye = torch.stack([
        torch.FloatTensor(n).uniform_(*x),
        torch.FloatTensor(n).uniform_(*y),
        torch.FloatTensor(n).uniform_(*z),
    ]).t()
    camera = Camera.from_args(eye=eye,
                           at=torch.tensor([0., 0., 0.]),
                           up=torch.tensor([0., 1., 0.]),
                           fov=math.pi * 45 / 180,
                           width=512, height=512)
    proj = camera.view_projection_matrix()
    return proj[:, None, ...]

def sample_all(n, device=None):
    res = [
        sample_light_colors(n),
        sample_light_directions(n),
        sample_views(n),
    ]
    if device:
        res = [f.to(device) for f in res]
    return res

def to_homogenous(v):
    ones = torch.ones(v.size(0), v.size(1), 1, device=v.device)
    return  torch.cat([v, ones], dim=-1)

def render_diffuse(glctx, v, faces, material, samples=None, resolution=(img_res, img_res)):
    vh = to_homogenous(v)
    if samples is None:
        samples = sample_all(v.size(0), v.device)
    light_color, light_directions, views = samples
    
    vertices_clip = (views @  vh[..., None]).squeeze(-1)
    vhc = vertices_clip[..., :3]
    vertex_normals = NRM.vertex_normals(vhc, faces)#.detach() #???? detach
    
    diffuse_strength = (vertex_normals @ light_directions)
    diffuse_strength = diffuse_strength.clip(0)
        
    lighting = diffuse_strength * light_color * material 
    
    rast, _ = dr.rasterize(glctx, vertices_clip, faces, resolution=resolution)
    out, _ = dr.interpolate(lighting, rast, faces)
    color  = dr.antialias(out, rast, vh, faces)
    return color