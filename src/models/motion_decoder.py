"""
Accepts Waveform Latents. (B, 128)

Returns:
    3-D Motion (B, 3)
"""

import torch
import torch.nn as nn


class MotionDecoder(nn.Module):

    def __init__(self, latent_dim=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64, 3)
        )

    def forward(self, waveform_latent):
        return self.model(waveform_latent)
