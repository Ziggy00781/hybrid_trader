import torch
from torch.utils.data import Dataset

class PatchTSTDataset(Dataset):
    def __init__(self, X, y, window):
        self.X = X
        self.y = y
        self.window = window
        self.n = len(X) - window

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x_win = self.X[idx : idx + self.window]      # (window, features)
        y_val = self.y[idx + self.window]            # scalar
        return x_win, y_val