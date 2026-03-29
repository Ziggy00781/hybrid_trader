import torch
import matplotlib
matplotlib.use("Agg")  # <-- IMPORTANT: headless backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from src.models.patchtst import PatchTST
from src.utils.patchtst_dataset import PatchTSTDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = Path("data/processed/patchtst")
MODEL_PATH = Path("models/patchtst/patchtst_best.pt")

INPUT_WINDOW = 1024
BATCH_SIZE = 64

def load_split(name):
    blob = torch.load(DATA_DIR / f"{name}_raw.pt")
    return blob["X"].float(), blob["y"].float()

def main():
    print("[viz] Loading test split...")
    X_test, y_test = load_split("test")

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
    trues = []

    print("[viz] Running inference...")
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            out = model(xb).squeeze(-1)

            preds.append(out.cpu().numpy())
            trues.append(yb.cpu().numpy())

    preds = np.concatenate(preds)
    trues = np.concatenate(trues)

    print("[viz] Saving plot to predictions.png...")
    plt.figure(figsize=(16, 6))
    plt.plot(trues, label="Actual", alpha=0.7)
    plt.plot(preds, label="Predicted", alpha=0.7)
    plt.title("PatchTST Predictions vs Actual Returns")
    plt.legend()
    plt.grid(True)

    out_path = Path("predictions.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"[viz] Saved: {out_path.resolve()}")

if __name__ == "__main__":
    main()