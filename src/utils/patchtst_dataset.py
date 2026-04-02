import torch
from torch.utils.data import Dataset
import numpy as np

class PatchTSTDataset(Dataset):
    """
    Improved PatchTST Dataset for mathematical features.
    Supports sequence_length and horizon for multi-hour predictions.
    """
    def __init__(self, data: np.ndarray, sequence_length: int = 512, horizon: int = 96):
        """
        Args:
            data: numpy array of shape (num_samples, num_features)
            sequence_length: how many past timesteps to use as input
            horizon: how many steps ahead to predict
        """
        self.data = torch.tensor(data, dtype=torch.float32)
        self.sequence_length = sequence_length
        self.horizon = horizon
        self.n = len(data) - sequence_length - horizon + 1

        if self.n <= 0:
            raise ValueError(f"Dataset too small for sequence_length={sequence_length} and horizon={horizon}")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # Input window: past sequence_length timesteps
        x = self.data[idx : idx + self.sequence_length]           # (sequence_length, features)
        
        # Target: value horizon steps ahead (we predict the return or close price)
        y = self.data[idx + self.sequence_length + self.horizon - 1, 3]  # column 3 = 'close' 
        
        # Optional: predict normalized return instead of raw price
        # y = (self.data[idx + self.sequence_length + self.horizon - 1, 3] - 
        #      self.data[idx + self.sequence_length - 1, 3]) / self.data[idx + self.sequence_length - 1, 3]

        return x, y.unsqueeze(0)   # return as (1,) tensor for consistency