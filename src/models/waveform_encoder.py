import torch
import torch.nn as nn


class WaveformEncoder(nn.Module):
    """
    Accepts the Synthetic or Raw Waveform and creates a Waveform Latent
    """

    def __init__(self, latent_dim=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(20 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, waveform):
        return self.model(waveform)
