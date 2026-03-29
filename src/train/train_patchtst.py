import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import numpy as np
import time

from src.models.patchtst import PatchTST  # adjust if needed
from src.utils.patchtst_dataset import PatchTSTDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path("data/processed/patchtst")
MODEL_DIR = Path("models/patchtst")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

INPUT_WINDOW = 1024
BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-4  # unchanged (you asked to skip fix #3)

LOG_PATH = Path("logs")
LOG_PATH.mkdir(parents=True, exist_ok=True)

def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def load_raw_split(split: str):
    blob = torch.load(DATA_DIR / f"{split}_raw.pt")
    X = blob["X"].float()
    y = blob["y"].float()
    return X, y

def make_loader(X, y, shuffle):
    ds = PatchTSTDataset(X, y, INPUT_WINDOW)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, drop_last=True)
    
def main():
    log(f"Using device: {DEVICE}")

    # -----------------------------
    # Load data
    # -----------------------------
    X_train, y_train = load_raw_split("train")
    X_val,   y_val   = load_raw_split("val")
    X_test,  y_test  = load_raw_split("test")

    num_features = X_train.shape[-1]
    log(f"num_features = {num_features}")

    train_loader = make_loader(X_train, y_train, shuffle=True)
    val_loader   = make_loader(X_val,   y_val,   shuffle=False)
    test_loader  = make_loader(X_test,  y_test,  shuffle=False)

    log(f"Train batches: {len(train_loader)}")
    log(f"Val batches:   {len(val_loader)}")
    log(f"Test batches:  {len(test_loader)}")

    # -----------------------------
    # Model
    # -----------------------------
    model = PatchTST(
        c_in=num_features,
        c_out=1,
        seq_len=INPUT_WINDOW,
        pred_len=1,
    )

    # Dual-GPU via DataParallel (easy path)
    if torch.cuda.device_count() > 1 and DEVICE.type == "cuda":
        log(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
        model = nn.DataParallel(model)

    model = model.to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")
    best_path = MODEL_DIR / "patchtst_best.pt"

    # -----------------------------
    # Training loop
    # -----------------------------
    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()
        model.train()
        train_losses = []

        for batch_idx, (xb, yb) in enumerate(train_loader):
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            preds = model(xb).squeeze(-1)

            # NaN guard on predictions/targets
            if torch.isnan(preds).any() or torch.isnan(yb).any():
                log(f"Epoch {epoch} batch {batch_idx}: NaN in preds/targets — skipping batch")
                continue

            loss = criterion(preds, yb)

            if torch.isnan(loss):
                log(f"Epoch {epoch} batch {batch_idx}: NaN loss — skipping batch")
                continue

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_losses.append(loss.item())

        epoch_train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        duration = time.time() - start_time
        log(f"Epoch {epoch} train_loss={epoch_train_loss:.6f} | duration={duration:.1f}s")

        # -----------------------------
        # Validation
        # -----------------------------
        model.eval()
        val_losses = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)

                preds = model(xb).squeeze(-1)

                if torch.isnan(preds).any() or torch.isnan(yb).any():
                    log(f"Epoch {epoch}: NaN in val preds/targets — skipping batch")
                    continue

                loss = criterion(preds, yb)
                if torch.isnan(loss):
                    log(f"Epoch {epoch}: NaN val loss — skipping batch")
                    continue

                val_losses.append(loss.item())

        epoch_val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        log(f"Epoch {epoch} val_loss={epoch_val_loss:.6f}")

        # Save best model only if val_loss is finite and improved
        if np.isfinite(epoch_val_loss) and epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), best_path)
            log(f"Epoch {epoch}: New best val_loss={best_val_loss:.6f} — model saved")

    # -----------------------------
    # Final test evaluation
    # -----------------------------
    log("Running final test evaluation...")

    if not best_path.exists():
        log("No best model found — skipping test evaluation")
        return

    # Rebuild model for loading (handle DataParallel)
    model = PatchTST(
        c_in=num_features,
        c_out=1,
        seq_len=INPUT_WINDOW,
        pred_len=1,
    )
    if torch.cuda.device_count() > 1 and DEVICE.type == "cuda":
        model = nn.DataParallel(model)
    model = model.to(DEVICE)

    state_dict = torch.load(best_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    test_losses = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            preds = model(xb).squeeze(-1)

            if torch.isnan(preds).any() or torch.isnan(yb).any():
                log("NaN in test preds/targets — skipping batch")
                continue

            loss = criterion(preds, yb)
            if torch.isnan(loss):
                log("NaN test loss — skipping batch")
                continue

            test_losses.append(loss.item())

    if test_losses:
        test_loss = float(np.mean(test_losses))
        log(f"Final test_loss={test_loss:.6f}")
    else:
        log("No valid test batches — test_loss undefined")

if __name__ == "__main__":
    main()