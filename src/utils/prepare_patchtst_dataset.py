import pandas as pd
import numpy as np
import torch
from pathlib import Path

INPUT_WINDOW = 1024
HORIZON = 3

DATA_PATH = Path("data/raw/merged_resampled/btc_multi_exchange_5m_resampled.parquet")
OUT_DIR = Path("data/processed/patchtst")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"[prep] Loading {DATA_PATH} ...")
df = pd.read_parquet(DATA_PATH)

# Ensure datetime index with UTC
if not isinstance(df.index, pd.DatetimeIndex):
    raise ValueError("[prep] DataFrame index must be DatetimeIndex")
df.index = pd.to_datetime(df.index, utc=True)

print(f"[prep] Raw rows: {len(df)}")
if len(df) <= HORIZON:
    raise ValueError(f"[prep] Not enough rows ({len(df)}) for HORIZON={HORIZON}")

# -----------------------------
# Compute 15m future log-return
# -----------------------------
if "binance_close" not in df.columns:
    raise KeyError("[prep] 'binance_close' column not found in dataframe")

df["future_close"] = df["binance_close"].shift(-HORIZON)

# Avoid division by zero / negative / weird values
eps = 1e-12
base = df["binance_close"].clip(lower=eps)
future = df["future_close"].clip(lower=eps)

df["target"] = np.log(future / base)

# -----------------------------
# CLEAN ALL FEATURES BEFORE SPLITTING
# -----------------------------
print("[prep] Cleaning NaN/inf rows across ALL features...")

before = len(df)

# Replace infs with NaN
df = df.replace([np.inf, -np.inf], np.nan)

# Drop ANY row with NaN in ANY column
df = df.dropna()

after = len(df)
print(f"[prep] Dropped {before - after} rows with NaN/inf in features or target")

if len(df) <= INPUT_WINDOW:
    raise ValueError("[prep] Not enough rows after cleaning to support windowing")

# -----------------------------
# Feature matrix
# -----------------------------
FEATURE_COLS = [c for c in df.columns if c not in ["future_close", "target"]]
if not FEATURE_COLS:
    raise ValueError("[prep] No feature columns found after excluding target/future_close")

X = df[FEATURE_COLS].values.astype(np.float32)
y = df["target"].values.astype(np.float32)

# Clip extreme targets to stabilize training
y = np.clip(y, -0.1, 0.1)

print(f"[prep] X shape: {X.shape}, y shape: {y.shape}")
print(f"[prep] y stats: min={y.min():.6f}, max={y.max():.6f}, mean={y.mean():.6f}, std={y.std():.6f}")

# -----------------------------
# Time splits (ensure test has enough rows)
# -----------------------------
train_mask = df.index < "2024-01-01"
val_mask   = (df.index >= "2024-01-01") & (df.index < "2024-04-01")
test_mask  = df.index >= "2024-04-01"

def _count(mask, name):
    n = int(mask.sum())
    print(f"[prep] {name} rows: {n}")
    return n

n_train = _count(train_mask, "train")
n_val   = _count(val_mask, "val")
n_test  = _count(test_mask, "test")

if n_train < INPUT_WINDOW or n_val < INPUT_WINDOW or n_test < INPUT_WINDOW:
    raise ValueError(
        f"[prep] Split too small for windowing: "
        f"train={n_train}, val={n_val}, test={n_test}, window={INPUT_WINDOW}"
    )

X_train, y_train = X[train_mask], y[train_mask]
X_val,   y_val   = X[val_mask],   y[val_mask]
X_test,  y_test  = X[test_mask],  y[test_mask]

# -----------------------------
# Normalize using training stats
# -----------------------------
mean = X_train.mean(axis=0)
std  = X_train.std(axis=0) + 1e-8

X_train = (X_train - mean) / std
X_val   = (X_val   - mean) / std
X_test  = (X_test  - mean) / std

print(f"[prep] Normalized shapes: "
      f"train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

# -----------------------------
# Save normalized arrays only
# -----------------------------
torch.save(
    {
        "X": torch.tensor(X_train),
        "y": torch.tensor(y_train),
        "mean": torch.tensor(mean),
        "std": torch.tensor(std),
    },
    OUT_DIR / "train_raw.pt",
)

torch.save(
    {
        "X": torch.tensor(X_val),
        "y": torch.tensor(y_val),
    },
    OUT_DIR / "val_raw.pt",
)

torch.save(
    {
        "X": torch.tensor(X_test),
        "y": torch.tensor(y_test),
    },
    OUT_DIR / "test_raw.pt",
)

print("[prep] Saved normalized raw datasets (no windows).")
print("[prep] Use PatchTSTDataset to generate windows lazily.")