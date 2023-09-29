import torch
import torch.nn as nn
from src.models.encoder import Encoder
from src.models.decoder import Decoder
from src.models.codebook import Codebook


class VQGAN(nn.Module):
    def __init__(self, args):
        super(VQGAN, self).__init__()
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)
        self.codebook = Codebook(args)

    def forward(self, imgs):
        encoded_images = self.encoder(imgs)        
        codebook_mapping, codebook_indices, q_loss = self.codebook(encoded_images)        
        decoded_images = self.decoder(codebook_mapping)

        return decoded_images, codebook_indices, q_loss

    def encode(self, imgs):
        encoded_images = self.encoder(imgs)        
        codebook_mapping, codebook_indices, q_loss = self.codebook(encoded_images)
        return codebook_mapping, codebook_indices, q_loss

    def decode(self, z):        
        decoded_images = self.decoder(z)
        return decoded_images

    def calculate_lambda(self, perceptual_loss, gan_loss):
        last_layer = self.decoder.model.tail[-1]
        last_layer_weight = last_layer.weight
        perceptual_loss_grads = torch.autograd.grad(perceptual_loss, last_layer_weight, retain_graph=True)[0]
        gan_loss_grads = torch.autograd.grad(gan_loss, last_layer_weight, retain_graph=True)[0]

        λ = torch.norm(perceptual_loss_grads) / (torch.norm(gan_loss_grads) + 1e-4)
        λ = torch.clamp(λ, 0, 1e4).detach()
        return 0.8 * λ

    @staticmethod
    def adopt_weight(disc_factor, i, threshold, value=0.):
        if i < threshold:
            disc_factor = value
        return disc_factor

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path))








