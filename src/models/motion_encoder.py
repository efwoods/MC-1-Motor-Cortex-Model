import torch
import torch.nn as nn


class MotionEncoder(nn.Module):
    """
    The motion encoder accepts physical motion and encodes to a motion_latent.
    """

    def __init__(self, latent_dim=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, motion_vec):  # Shape (S, 3)
        return self.model(motion_vec)
