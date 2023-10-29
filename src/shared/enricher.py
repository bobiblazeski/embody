import torch
import torch.nn as nn

from src.shared.normals import vertex_normals
from src.shared.faces import make_faces
import src.shared.util as U
from src.shared.distancer import (MeanDistancer, NeighborDistancer)

class Enricher(nn.Module):
    def __init__(self, terms=4,  normals=True, distances=True):
        super().__init__()
        self.terms = terms
        self.normals = normals
        self.distances = distances
        self.mean_distancer = MeanDistancer()
        self.neighbor_distancer = NeighborDistancer()        

    def adjusted_normals(self, t, faces):
        if faces is None:
            faces = make_faces(t.size(-1), device=t.device)
        vertices = U.to_vertices(t)
        normals =  U.to_patch(vertex_normals(vertices, faces))
        distances = self.mean_distancer(t)
        return normals * distances    

    def forward(self, x, faces=None):
        res = [x]
        if self.terms:
            res.append(U.add_fourier(x, self.terms, input=False))
        if self.normals:            
            res.append(self.adjusted_normals(x, faces))
        if self.distances:            
            res.append(self.neighbor_distancer(x))        
        return torch.cat(res, dim=1) if len(res) > 1 else res[0]
