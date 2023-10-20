import torch
import torch.nn as nn
from src.models.quantizer import Quantizer
from src.models.decoder import Decoder

class Codec(nn.Module):
    def __init__(self, embedding_dim, num_embeddings, num_hiddens, 
                 num_residual_layers, num_residual_hiddens, noise_alpha=5):
        super().__init__()
        self.noise_alpha = noise_alpha
        self.mag_norm = noise_alpha / (num_embeddings * embedding_dim)
        self._decoder = Decoder(embedding_dim,
                                num_hiddens, 
                                num_residual_layers, 
                                num_residual_hiddens)
        self._quantizer = Quantizer(num_embeddings, embedding_dim)

    def forward(self, x, add_noise=False):
        quantized, vq_loss, perplexity, encodings = self._quantizer(x)
        if add_noise:
            noise = torch.zeros_like(quantized).uniform_(-self.mag_norm, self.mag_norm)
            quantized = quantized + noise
        x_recon = self._decoder(quantized)
        return x_recon, (vq_loss, perplexity, quantized, encodings)    
