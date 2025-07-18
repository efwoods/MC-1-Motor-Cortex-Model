import torch
import torch.nn as nn


class WaveformEncoder(nn.Module):
    """
    Accepts the Synthetic or Raw Waveform and creates a Waveform Latent
    """

    def __init__(self, latent_dim=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(20 * 64, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, waveform):
        return self.model(waveform.transpose(1, 2))
