import math
import torch
import torch.nn.functional as F


from numpy import random, mgrid, dstack
from scipy.spatial import cKDTree

from noise.noise_util import strength

def worley_noise(width, height, density):
    points = [[random.randint(0, height), random.randint(0, width)] for _ in range(density)]  # Generates Points(y, x)   
    coord = dstack(mgrid[0:height, 0:width])  # Makes array with coordinates as values
    tree = cKDTree(points)  # Build Tree
    distances = tree.query(coord, workers=-1)[0]  # Calculate distances (workers=-1: Uses all CPU Cores)
    return torch.tensor(distances, dtype=torch.float32)
    
def worley_fractal(size, normalized=True, exp=None, cut_last=1):
    scales = [2**i for i in range(int(math.log2(size)), 0, -1)]
    scales = scales[:-cut_last]
    strengths = strength(len(scales), exp=exp)
    noise = 0    
    for scale, s in zip(scales, strengths):
        n = worley_noise(size, size, scale)
        noise += n * s
    if normalized:
        noise = noise-noise.mean()
        noise /= noise.abs().max()
    return noise
