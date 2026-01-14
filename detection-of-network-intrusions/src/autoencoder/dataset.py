import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Union


class AutoencoderDataset(Dataset):
    def __init__(self, X: Union[np.ndarray, "np.memmap"]):
        arr = np.asarray(X)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32, copy=False)
        self.X = torch.from_numpy(arr)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        x = self.X[idx]
        return x, x


class CoresIotDataset(AutoencoderDataset):
    pass
