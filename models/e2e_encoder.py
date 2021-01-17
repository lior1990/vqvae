import torch.nn as nn
from models.encoder import Encoder


class E2EEncoder(nn.Module):
    def __init__(self, h_dim, res_h_dim, n_res_layers, embedding_dim, n_dimension_changes):
        super(E2EEncoder, self).__init__()
        # encode image into continuous latent space
        self.encoder = Encoder(3, h_dim, n_res_layers, res_h_dim, n_dimension_changes)
        self.pre_quantization_conv = nn.Conv2d(h_dim, embedding_dim, kernel_size=1, stride=1)

    def forward(self, x):
        z_e = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)
        return z_e
