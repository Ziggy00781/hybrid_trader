import torch
import numpy as np
from pathlib import Path

blob = torch.load(Path("data/processed/patchtst/train_raw.pt"))
y = blob["y"].numpy()
print("min:", y.min(), "max:", y.max(), "mean:", y.mean(), "std:", y.std())
print("any nan:", np.isnan(y).any(), "any inf:", np.isinf(y).any())