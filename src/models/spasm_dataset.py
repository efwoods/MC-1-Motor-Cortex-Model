"""
This is a dataset where there are random spasms for a duration.

A default spasm lasts for 50 samples at 50 Hz or a single second
and has a random noise of physical movement added up to a maximum numerical value of 0.2
(or 0.2 from the original physical position... broadcast to each dimension).

The units of this dataset are unknown.
For hypothetical purposes consider this to be
2.4 inches from the origin of the physical position
given an armspan of a rhesus macaque monkey of
2 feet and estimated judgments from observing the data.
Do not use this context to convert any units.
"""

import torch


class SpasmDataset(torch.utils.data.Dataset):
    def __init__(self, spasm_data, spasm_indices, ecog_synth_spasms=None):
        self.spasm_data = spasm_data
        self.spasm_indices = spasm_indices
        self.ecog_synth_spasms = ecog_synth_spasms

    def __len__(self):
        return len(self.spasm_data)

    def __getitem__(self, idx):
        return {"spasm": self.spasm_data[idx]}
