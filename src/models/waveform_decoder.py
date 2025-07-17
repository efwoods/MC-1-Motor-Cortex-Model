import torch
import torch.nn as nn


class WaveformDecoder(nn.Module):
    """
    Decodes motion_latents into synthetic waveforms.
    """

    def __init__(self, latent_dim=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 20 * 64),
            nn.Unflatten(1, (20, 64)),
            nn.LayerNorm((20, 64)),
        )

    def forward(self, motion_latent):
        return self.model(motion_latent)
