import torch.nn as nn
from src.models.quantizer import Quantizer
from src.models.decoder import Decoder

class Codec(nn.Module):
    def __init__(self, embedding_dim, num_embeddings, num_hiddens, 
                 num_residual_layers, num_residual_hiddens):
        super().__init__()
        self._decoder = Decoder(embedding_dim,
                                num_hiddens, 
                                num_residual_layers, 
                                num_residual_hiddens)
        self._quantizer = Quantizer(num_embeddings, embedding_dim)

    def forward(self, x):  
        quantized, vq_loss, perplexity, encodings = self._quantizer(x)
        x_recon = self._decoder(quantized)
        return x_recon, (vq_loss, perplexity, quantized, encodings)