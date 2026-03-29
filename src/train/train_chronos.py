import torch
from torch.utils.data import DataLoader
from chronos import ChronosForecastingModel
from pathlib import Path
import pandas as pd

from src.models.chronos_dataset import ChronosDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train_chronos(
    data_path="data/raw/btcusdt_5m.parquet",
    model_name="amazon/chronos-base",
    window=256,
    horizon=12,
    batch_size=32,
    epochs=5,
    lr=1e-4,
):
    print("Loading data...")
    df = pd.read_parquet(data_path)

    dataset = ChronosDataset(df, window=window, horizon=horizon)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    print(f"Dataset size: {len(dataset)} sequences")

    # Load Chronos model
    model = ChronosForecastingModel.from_pretrained(model_name)
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0

        for x, y in loader:
            x = x.to(DEVICE)  # (B, window, 1)
            y = y.to(DEVICE)  # (B, horizon)

            # Chronos expects shape (B, window)
            x_in = x.squeeze(-1)

            # Forward pass
            preds = model(x_in, prediction_length=horizon)

            # preds shape: (B, horizon)
            loss = torch.nn.functional.mse_loss(preds, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.6f}")

    # Save fine‑tuned model
    save_path = Path("models/chronos_btcusdt")
    save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_path)

    print(f"Saved fine‑tuned Chronos model to {save_path}")


if __name__ == "__main__":
    train_chronos()