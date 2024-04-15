# https://github.com/NVIDIAGameWorks/kaolin/blob/5406915c3a020596384b1a36679ff68afd04d455/kaolin/ops/mesh/trianglemesh.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Subdivide(nn.Module):
    def __init__(self):
        super().__init__()        
        weight = torch.tensor([
            [ 0.025,  0.100, 0.025],
            [ 0.100,  0.500, 0.100],
            [ 0.025,  0.100, 0.025],
        ])[None, None].expand(3, -1, -1, -1)
        self.register_buffer('weight', weight)

    def smooth(self, x, interpolate=False):
        if interpolate:
            x = F.interpolate(x,  x.size(-1) + 2, mode='bilinear', align_corners=True)  
        return F.conv2d(x, self.weight, padding=0, groups=3,)
        
    def forward(self, x, steps=1):        
        for _ in range(steps):
            size =  x.size(-1) * 2 + 2
            x = F.interpolate(x, size, mode='bilinear', align_corners=True)        
            x = self.smooth(x)
        return x
    
def _get_adj_verts(edges_ex2, v):
    """Get sparse adjacency matrix for vertices given edges"""
    adj_sparse_idx = torch.cat([edges_ex2, torch.flip(edges_ex2, [1])])
    adj_sparse_idx = torch.unique(adj_sparse_idx, dim=0)

    values = torch.ones(
        adj_sparse_idx.shape[0], device=edges_ex2.device).float()
    adj_sparse = torch.sparse.FloatTensor(
        adj_sparse_idx.t(), values, torch.Size([v, v]))
    return adj_sparse


def _get_alpha(n):
    """Compute weight alpha based on number of neighboring vertices following Loop Subdivision"""
    n = n.float()
    alpha = (5.0 / 8 - (3.0 / 8 + 1.0 / 4 * torch.cos(2 * math.pi / n)) ** 2) / n
    alpha[n == 3] = 3 / 16

    return alpha
    
def subdivide_trianglemesh(vertices, faces, iterations, alpha=None):
    r"""Subdivide triangular meshes following the scheme of Loop subdivision proposed in 
    `Smooth Subdivision Surfaces Based on Triangles`_. 
    If the smoothing factor alpha is not given, this function performs exactly as Loop subdivision.
    Elsewise the vertex position is updated using the given per-vertex alpha value, which is 
    differentiable and the alpha carries over to subsequent subdivision iterations. Higher alpha leads
    to smoother surfaces, and a vertex with alpha = 0 will not change from its initial position 
    during the subdivision. Thus, alpha can be learnable to preserve sharp geometric features in contrast to 
    the original Loop subdivision.
    For more details and example usage in learning, see `Deep Marching Tetrahedra\: a Hybrid 
    Representation for High-Resolution 3D Shape Synthesis`_ NeurIPS 2021.

    Args:
        vertices (torch.Tensor): batched vertices of triangle meshes, of shape
                                 :math:`(\text{batch_size}, \text{num_vertices}, 3)`.
        faces (torch.LongTensor): unbatched triangle mesh faces, of shape
                              :math:`(\text{num_faces}, 3)`.
        iterations (int): number of subdivision iterations.
        alpha (optional, torch.Tensor): batched per-vertex smoothing factor, alpha, of shape
                            :math:`(\text{batch_size}, \text{num_vertices})`.

    Returns:
        (torch.Tensor, torch.LongTensor): 
            - batched vertices of triangle meshes, of shape
                                 :math:`(\text{batch_size}, \text{new_num_vertices}, 3)`.
            - unbatched triangle mesh faces, of shape
                              :math:`(\text{num_faces} \cdot 4^\text{iterations}, 3)`.

    Example:
        >>> vertices = torch.tensor([[[0, 0, 0],
        ...                           [1, 0, 0],
        ...                           [0, 1, 0],
        ...                           [0, 0, 1]]], dtype=torch.float)
        >>> faces = torch.tensor([[0, 1, 2],[0, 1, 3],[0, 2, 3],[1, 2, 3]], dtype=torch.long)
        >>> alpha = torch.tensor([[0, 0, 0, 0]], dtype=torch.float)
        >>> new_vertices, new_faces = subdivide_trianglemesh(vertices, faces, 1, alpha)
        >>> new_vertices
        tensor([[[0.0000, 0.0000, 0.0000],
                 [1.0000, 0.0000, 0.0000],
                 [0.0000, 1.0000, 0.0000],
                 [0.0000, 0.0000, 1.0000],
                 [0.3750, 0.1250, 0.1250],
                 [0.1250, 0.3750, 0.1250],
                 [0.1250, 0.1250, 0.3750],
                 [0.3750, 0.3750, 0.1250],
                 [0.3750, 0.1250, 0.3750],
                 [0.1250, 0.3750, 0.3750]]])
        >>> new_faces
        tensor([[1, 7, 4],
                [0, 4, 5],
                [2, 5, 7],
                [5, 4, 7],
                [1, 8, 4],
                [0, 4, 6],
                [3, 6, 8],
                [6, 4, 8],
                [2, 9, 5],
                [0, 5, 6],
                [3, 6, 9],
                [6, 5, 9],
                [2, 9, 7],
                [1, 7, 8],
                [3, 8, 9],
                [8, 7, 9]])
                
    .. _Smooth Subdivision Surfaces Based on Triangles:
            https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/thesis-10.pdf    

    .. _Deep Marching Tetrahedra\: a Hybrid Representation for High-Resolution 3D Shape Synthesis:
            https://arxiv.org/abs/2111.04276
    """
    init_alpha = alpha
    for i in range(iterations):
        device = vertices.device
        b, v, f = vertices.shape[0], vertices.shape[1], faces.shape[0]

        edges_fx3x2 = faces[:, [[0, 1], [1, 2], [2, 0]]]
        edges_fx3x2_sorted, _ = torch.sort(edges_fx3x2.reshape(edges_fx3x2.shape[0] * edges_fx3x2.shape[1], 2), -1)
        all_edges_face_idx = torch.arange(edges_fx3x2.shape[0], device=device).unsqueeze(-1).expand(-1, 3).reshape(-1)
        edges_ex2, inverse_indices, counts = torch.unique(
            edges_fx3x2_sorted, dim=0, return_counts=True, return_inverse=True)

        # To compute updated vertex positions, first compute alpha for each vertex
        # TODO(cfujitsang): unify _get_adj_verts with adjacency_matrix
        adj_sparse = _get_adj_verts(edges_ex2, v)
        n = torch.sparse.sum(adj_sparse, 0).to_dense().view(-1, 1)
        if init_alpha is None:
            alpha = (_get_alpha(n) * n).unsqueeze(0)
        if alpha.dim() == 2:
            alpha = alpha.unsqueeze(-1)

        adj_verts_sum = torch.bmm(adj_sparse.unsqueeze(0), vertices)
        vertices_new = (1 - alpha) * vertices + alpha / n * adj_verts_sum

        e = edges_ex2.shape[0]
        edge_points = torch.zeros((b, e, 3), device=device)  # new point for every edge
        edges_fx3 = inverse_indices.reshape(f, 3) + v
        alpha_points = torch.zeros((b, e, 1), device=device)

        mask_e = (counts == 2)

        # edge points on boundary is computed as midpoint
        if torch.sum(~mask_e) > 0:
            edge_points[:, ~mask_e] += torch.mean(vertices[:,
                                                  edges_ex2[~mask_e].reshape(-1), :].reshape(b, -1, 2, 3), 2)
            alpha_points[:, ~mask_e] += torch.mean(alpha[:, edges_ex2[~mask_e].reshape(-1), :].reshape(b, -1, 2, 1), 2)

        counts_f = counts[inverse_indices]
        mask_f = (counts_f == 2)
        group = inverse_indices[mask_f]
        _, indices = torch.sort(group)
        edges_grouped = all_edges_face_idx[mask_f][indices]
        edges_face_idx = torch.stack([edges_grouped[::2], edges_grouped[1::2]], dim=-1)
        e_ = edges_face_idx.shape[0]
        edges_face = faces[edges_face_idx.reshape(-1), :].reshape(-1, 2, 3)
        edges_vert = vertices[:, edges_face.reshape(-1), :].reshape(b, e_, 6, 3)
        edges_vert = torch.cat([edges_vert, vertices[:, edges_ex2[mask_e].reshape(-1),
                               :].reshape(b, -1, 2, 3)], 2).mean(2)

        alpha_vert = alpha[:, edges_face.reshape(-1), :].reshape(b, e_, 6, 1)
        alpha_vert = torch.cat([alpha_vert, alpha[:, edges_ex2[mask_e].reshape(-1),
                               :].reshape(b, -1, 2, 1)], 2).mean(2)

        edge_points[:, mask_e] += edges_vert
        alpha_points[:, mask_e] += alpha_vert

        alpha = torch.cat([alpha, alpha_points], 1)
        vertices = torch.cat([vertices_new, edge_points], 1)
        faces = torch.cat([faces, edges_fx3], 1)
        faces = faces[:, [[1, 4, 3], [0, 3, 5], [2, 5, 4], [5, 3, 4]]].reshape(-1, 3)
    return vertices, faces