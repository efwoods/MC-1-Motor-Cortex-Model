import torch
import torch.nn as nn


class MotionDecoder(nn.Module):
    """
    Accepts Waveform Latents and recreates Motion
    """

    def __init__(self, latent_dim=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64, 3)
        )

    def forward(self, waveform_latent):
        return self.model(waveform_latent)
