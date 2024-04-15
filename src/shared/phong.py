import math
import torch
import torch.nn.functional as F
from kaolin.render.camera import Camera
import nvdiffrast.torch as dr 
from src.shared.transforms import img_res

import src.shared.normals as NRM

def sample_light_colors(n, ratio, high):    
    return torch.randint(int(ratio*high), high, (n, 1, 3)).div(high)

def sample_light_directions(n, ratio, standard):
    direction_std = torch.tensor(standard)[None, :, None]
    direction_rnd = F.normalize(torch.randn(n, 3, 1), dim=0)
    
    directions = ratio * direction_std + (1-ratio) * direction_rnd
    return F.normalize(directions, dim=-1)

def sample_views(n, x, y, z):
    eye = torch.stack([
        torch.FloatTensor(n).uniform_(*x),
        torch.FloatTensor(n).uniform_(*y),
        torch.FloatTensor(n).uniform_(*z),
    ]).t()
    camera = Camera.from_args(eye=eye,
                           at=torch.tensor([0., 0., 0.]),
                           up=torch.tensor([0., 1., 0.]),
                           fov=math.pi * 45 / 180,
                           width=256, height=256)
    proj = camera.view_projection_matrix()
    return proj[:, None, ...]

def sample_all(n, d, device=None):
    res = [
        sample_light_colors(n, d.color_ratio, d.color_high),
        sample_light_directions(n, d.direction_ratio, d.direction_standard),
        sample_views(n, d.eye_x, d.eye_y, d.eye_z),
    ]    
    return [f.to(device) for f in res] if device else res

def to_homogenous(v):
    ones = torch.ones(v.size(0), v.size(1), 1, device=v.device)
    return  torch.cat([v, ones], dim=-1)

def render_diffuse(glctx, v, faces, material, hp, samples=None, return_samples=False):
    vh = to_homogenous(v)
    samples = samples or sample_all(v.size(0), hp, v.device)
    light_color, light_directions, views = samples
    
    vertices_clip = (views @  vh[..., None]).squeeze(-1)
    vhc = vertices_clip[..., :3]
    vertex_normals = NRM.vertex_normals(vhc, faces)#.detach() #???? detach
    
    diffuse_strength = (vertex_normals @ light_directions)
    diffuse_strength = diffuse_strength.clip(0)
        
    lighting = diffuse_strength * light_color * material + hp.min_light
    
    rast, _ = dr.rasterize(glctx, vertices_clip, faces, resolution=(hp.rndr_sz, hp.rndr_sz))
    out, _ = dr.interpolate(lighting, rast, faces)
    color  = dr.antialias(out, rast, vh, faces)
    return (color, samples) if return_samples else color