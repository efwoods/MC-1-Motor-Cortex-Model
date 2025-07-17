import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd


class MotionECoGDataset(Dataset):
    def __init__(self, motion_csv, ecog_csv, motion_stride=1, ecog_window=20):
        self.motion_df = pd.read_csv(motion_csv)
        self.ecog_df = pd.read_csv(ecog_csv)

        assert (
            len(self.ecog_df) >= len(self.motion_df) * ecog_window
        ), "ECoG data must be longer than motion data × window"

        self.motion_stride = motion_stride
        self.ecog_window = ecog_window

        # Drop index columns if present
        if "Unnamed: 0" in self.motion_df.columns:
            self.motion_df = self.motion_df.drop(columns=["Unnamed: 0"])
        if "Unnamed: 0" in self.ecog_df.columns:
            self.ecog_df = self.ecog_df.drop(columns=["Unnamed: 0"])

        # Assume first three motion columns are X, Y, Z
        self.motion_xyz = self.motion_df.iloc[:, :3].values.astype(np.float32)
        self.ecog = self.ecog_df.values.astype(np.float32)

        self.total_samples = len(self.motion_xyz)

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # Get motion vector (X, Y, Z)
        motion_vec = self.motion_xyz[idx]

        # Get corresponding ecog window
        ecog_start = idx * self.ecog_window
        ecog_end = ecog_start + self.ecog_window
        ecog_window = self.ecog[ecog_start:ecog_end]

        return {
            "motion": torch.tensor(motion_vec),  # Shape (3,)
            "ecog": torch.tensor(ecog_window),  # Shape (20, 64)
        }
