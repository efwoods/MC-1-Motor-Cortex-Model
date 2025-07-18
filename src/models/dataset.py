import torch
from torch.utils.data import Dataset
import numpy as np


class MotionECoGDataset(Dataset):
    def __init__(self, motion_np_path, ecog_np_path, motion_stride=1, ecog_window=20):
        self.motion_data = np.load(motion_np_path).astype(np.float32)  # Shape: (N, 3)
        self.ecog_data = np.load(ecog_np_path).astype(
            np.float32
        )  # Shape: (N * ecog_window, C)

        assert (
            self.ecog_data.shape[0] >= self.motion_data.shape[0] * ecog_window
        ), f"ECoG data ({self.ecog_data.shape[0]}) must be at least motion data ({self.motion_data.shape[0]}) * ecog_window ({ecog_window})"

        self.motion_stride = motion_stride
        self.ecog_window = ecog_window
        self.total_samples = len(self.motion_data)

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # Get motion vector (X, Y, Z)
        motion_vec = self.motion_data[idx]  # Shape: (3,)

        # Get corresponding ecog window
        ecog_start = idx * self.ecog_window
        ecog_end = ecog_start + self.ecog_window
        ecog_window = self.ecog_data[ecog_start:ecog_end]  # Shape: (ecog_window, C)

        return {
            "motion": torch.tensor(motion_vec, dtype=torch.float32),  # (3,)
            "ecog": torch.tensor(ecog_window, dtype=torch.float32),  # (ecog_window, C)
        }
