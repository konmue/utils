import pandas as pd
import torch
from torch.utils.data import Dataset


class FilePerBatchData(Dataset):
    def __init__(self, data_dir: str, n_batches: int) -> None:
        super().__init__()

        self.folder_path = data_dir
        self.n_batches = n_batches

    def __len__(self):
        return self.n_batches

    def __getitem__(self, index):
        x = pd.read_parquet(f"{self.data_dir}_{index}.parquet")
        return torch.from_numpy(x.values).float()
