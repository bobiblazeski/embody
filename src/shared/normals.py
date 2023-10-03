import torch

def _face_normals(verts, faces):    
    fi = torch.transpose(faces, 0, 1).long()
    verts = torch.transpose(verts, -2, -1)    
    v = [verts.index_select(-1, fi[0]),
         verts.index_select(-1, fi[1]),
         verts.index_select(-1, fi[2])]        
    c = torch.cross(v[1] - v[0], v[2] - v[0])    
    n = c / torch.norm(c, dim=-2, keepdim=True)
    return n

_safe_acos = lambda x: torch.acos(x.clamp(min=-1, max=1))    

def _vertex_normals(verts, faces, face_normals):        
    fi = torch.transpose(faces, 0, 1).long()
    verts = torch.transpose(verts, 0, 1)
    normals = torch.zeros_like(verts)

    v = [verts.index_select(1, fi[0]),
         verts.index_select(1, fi[1]),
         verts.index_select(1, fi[2])]

    for i in range(3):
        d0 = v[(i + 1) % 3] - v[i]
        d0 = d0 / torch.norm(d0)
        d1 = v[(i + 2) % 3] - v[i]
        d1 = d1 / torch.norm(d1)
        d = torch.sum(d0*d1, 0)
        face_angle = _safe_acos(torch.sum(d0*d1, 0))
        nn =  face_normals * face_angle
        for j in range(3):
            normals[j].index_add_(0, fi[i], nn[j])
    return (normals / torch.norm(normals, dim=0)).transpose(0, 1)

def vertex_normals(vertices, faces):
    assert vertices.dim() == 3 and faces.dim() == 2, \
        f'{vertices.dim()=} != 3  or {faces.dim()=} != 2'
    face_normals = _face_normals(vertices, faces)
    return torch.stack([
        _vertex_normals(v, faces, fn)
        for v, fn in zip(vertices, face_normals)
    ])
