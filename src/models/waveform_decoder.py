"""
Accepts motion_latents (B, 128)

Returns:
    Synthetic waveforms (B, (20, 64))
"""

import torch
import torch.nn as nn


class WaveformDecoder(nn.Module):
    """
    Decodes waveform (ECoG) signal from latent representation using temporal modeling.
    Output shape: (batch, time=20, channels=64)
    """

    def __init__(self, latent_dim=64, time_steps=20, channels=64, hidden_dim=256):
        super().__init__()
        self.time_steps = time_steps
        self.channels = channels

        # Expand latent to initialize GRU hidden state
        self.latent_to_hidden = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim)
        )

        # GRU generates temporal dynamics
        self.gru = nn.GRU(
            input_size=channels, hidden_size=hidden_dim, num_layers=2, batch_first=True
        )

        # Map GRU output to ECoG signal at each time step
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, channels),
            nn.Tanh(),  # Or ReLU if your data is not [-1, 1] normalized
        )

        # Optional post-smoothing
        self.temporal_smoothing = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
            groups=channels,  # depth-wise
        )

    def forward(self, latent):
        """
        latent: (batch, latent_dim)
        returns: (batch, time_steps, channels)
        """
        batch_size = latent.size(0)

        # Expand latent vector into GRU hidden state
        h0 = self.latent_to_hidden(latent)  # (batch, hidden_dim)
        h0 = h0.unsqueeze(0).repeat(2, 1, 1)  # (num_layers, batch, hidden_dim)

        # Feed dummy zero input to GRU
        z = torch.zeros(batch_size, self.time_steps, self.channels).to(latent.device)

        # GRU forward
        out, _ = self.gru(z, h0)  # (batch, time_steps, hidden_dim)

        # Decode ECoG signal
        out = self.output_layer(out)  # (batch, time_steps, channels)

        # Apply optional smoothing
        # Convert (batch, time, channels) → (batch, channels, time) → conv1d → back
        out = out.transpose(1, 2)
        out = self.temporal_smoothing(out)
        out = out.transpose(1, 2)

        return out
