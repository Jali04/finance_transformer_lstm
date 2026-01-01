# src/dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset

class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int):
        self.X, self.y = X, y
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        xw = self.X[idx: idx + self.seq_len]
        yw = self.y[idx + self.seq_len]
        return (
            torch.tensor(xw, dtype=torch.float32),
            torch.tensor([yw], dtype=torch.float32)
        )