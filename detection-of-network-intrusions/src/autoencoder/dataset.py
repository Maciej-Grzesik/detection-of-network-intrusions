import numpy as np
import torch
from torch.utils.data import Dataset


class CoresIotDataset(Dataset):
    def __init__(self, X: np.ndarray):
        self.X = torch.from_numpy(X).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        x = self.X[idx]
        return x, x
