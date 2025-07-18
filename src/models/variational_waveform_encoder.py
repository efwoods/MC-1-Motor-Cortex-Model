"""
Accepts a Synthetic or Raw Waveform. (B, (20, 64))

Returns:
    Waveform Latents (B, 128)
"""

import torch
import torch.nn as nn


class VariationalWaveformEncoder(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128, latent_dim=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,  # 64
            hidden_size=hidden_dim,  # 128
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.fc_mean = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)

    def forward(self, waveform):  # waveform: [B, 20, 64]
        lstm_out, (hn, cn) = self.lstm(waveform)  # lstm_out: [B, 20, 2*H]
        h_last = lstm_out[:, -1, :]  # Use the last time step's output

        mean = self.fc_mean(h_last)
        logvar = self.fc_logvar(h_last)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std  # Reparameterization

        return z, mean, logvar
