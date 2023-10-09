import torch
import torch.nn as nn
from einops import rearrange

class Quantizer(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """    
    def __init__(self, num_codebook_vectors, latent_dim, beta=0.25):
        super().__init__()
        self.n_e = num_codebook_vectors
        self.e_dim = latent_dim
        self.beta = beta        
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)    
        self.re_embed = num_codebook_vectors          

    def forward(self, z):                
        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)                
        # compute loss for embedding        
        loss = self.beta * torch.mean((z_q.detach()-z)**2) + \
               torch.mean((z_q - z.detach()) ** 2)        
        # preserve gradients
        z_q = z + (z_q - z).detach()
        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
        return z_q, min_encoding_indices, loss