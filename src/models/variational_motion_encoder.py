"""
This encodes 3-D motion. (B, 3)

Returns:
    Motion Latents (B, 128)
"""

import torch
import torch.nn as nn


class VariationalMotionEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU()
        )
        self.fc_mean = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

    def forward(self, x):
        h = self.encoder(x)
        mean = self.fc_mean(h)
        logvar = self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std  # Reparameterization trick
        return z, mean, logvar
