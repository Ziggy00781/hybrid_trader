import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from src.models.patchtst import PatchTST
from src.utils.patchtst_dataset import PatchTSTDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PARQUET = Path("data/raw/merged_resampled/btc_multi_exchange_5m_resampled.parquet")
DATA_DIR = Path("data/processed/patchtst")
MODEL_PATH = Path("models/patchtst/patchtst_best.pt")

INPUT_WINDOW = 1024
BATCH_SIZE = 64

def load_split(name):
    blob = torch.load(DATA_DIR / f"{name}_raw.pt")
    return blob["X"].float(), blob["y"].float()

def main():
    print("[viz] Loading raw price data...")
    df = pd.read_parquet(DATA_PARQUET)
    df.index = pd.to_datetime(df.index, utc=True)

    # --- use the SAME split logic as prepare_patchtst_dataset.py ---
    train_mask = df.index < "2024-01-01"
    val_mask   = (df.index >= "2024-01-01") & (df.index < "2024-04-01")
    test_mask  = df.index >= "2024-04-01"

    df_test = df[test_mask].copy()
    price = df_test["binance_close"].values.astype(float)

    print(f"[viz] Test price rows: {len(price)}")

    print("[viz] Loading test split tensors...")
    X_test, y_test = load_split("test")
    print(f"[viz] X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    ds = PatchTSTDataset(X_test, y_test, INPUT_WINDOW)
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    num_features = X_test.shape[-1]

    print("[viz] Loading model...")
    model = PatchTST(
        c_in=num_features,
        c_out=1,
        seq_len=INPUT_WINDOW,
        pred_len=1,
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    preds = []

    print("[viz] Running inference...")
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(DEVICE)
            out = model(xb).squeeze(-1)
            preds.append(out.cpu().numpy())

    preds = np.concatenate(preds)
    print(f"[viz] preds shape: {preds.shape}")

    # Convert predictions to direction (+1, 0, -1)
    direction = np.sign(preds)

    # Align price with windowed predictions:
    # first INPUT_WINDOW points are only used as history
    if len(price) <= INPUT_WINDOW:
        raise ValueError(f"Not enough test price points ({len(price)}) for window={INPUT_WINDOW}")

    price_for_windows = price[INPUT_WINDOW : INPUT_WINDOW + len(direction)]
    n = len(price_for_windows)
    direction = direction[:n]

    print(f"[viz] Aligned price points: {n}, aligned preds: {len(direction)}")

    # Scale direction to overlay on price
    scale = price_for_windows.std() * 0.5
    scaled_pred = price_for_windows + direction * scale

    print("[viz] Saving price vs prediction plot...")
    plt.figure(figsize=(18, 7))
    plt.plot(price_for_windows, label="Price", color="blue", alpha=0.7)
    plt.plot(scaled_pred, label="AI Prediction (scaled)", color="orange", alpha=0.7)
    plt.title("BTC Price vs AI Predicted Direction (Test Period)")
    plt.legend()
    plt.grid(True)

    out_path = Path("price_vs_prediction.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"[viz] Saved: {out_path.resolve()}")

if __name__ == "__main__":
    main()