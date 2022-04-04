import os
from typing import Callable

import torch
from torch.utils.data import Dataset


class JITSimulation(Dataset):
    def __init__(
        self, simulation_fn: Callable, simulation_params: dict, n_batches: int
    ) -> None:
        super().__init__()
        self.simulation_fn = simulation_fn
        self.simulation_params = simulation_params
        self.n_batches = n_batches

    def __len__(self):
        return self.n_batches

    def __getitem__(self, idx):
        return self.simulation_fn(self.simulation_params)
