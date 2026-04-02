import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import logging
import joblib
import warnings
from sklearn.preprocessing import StandardScaler

# Aggressive suppression of annoying warnings
warnings.filterwarnings("ignore", message="triton not found")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", module="torch.utils.flop_counter")

from src.features.ta_regime_features import build_mathematical_features
from src.utils.patchtst_dataset import PatchTSTDataset
from src.models.patchtst import PatchTST

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


class PatchTSTLightning(pl.LightningModule):
    def __init__(self, model, learning_rate=8e-5, horizon=96):
        super().__init__()
        self.model = model
        self.lr = learning_rate
        self.horizon = horizon
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}


def prepare_data_for_patchtst(df: pd.DataFrame, sequence_length=512, horizon=96, batch_size=256):
    logger.info("Building mathematical features...")
    feats = build_mathematical_features(df)

    price_cols = ['open', 'high', 'low', 'close', 'volume']
    price_data = df[price_cols].loc[feats.index]

    combined = pd.concat([price_data, feats.drop(columns=['regime'], errors='ignore')], axis=1)
    combined = combined.ffill().bfill().fillna(0.0)

    logger.info(f"Prepared raw dataset: {combined.shape[0]:,} samples × {combined.shape[1]} features")

    scaler_path = Path("data/processed/patchtst/scaler.pkl")
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
    else:
        scaler = StandardScaler()
        scaler.fit(combined.values)
        Path("data/processed/patchtst").mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_path)

    scaled_values = scaler.transform(combined.values)

    dataset = PatchTSTDataset(scaled_values, sequence_length=sequence_length, horizon=horizon)
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=12,
        pin_memory=True,
        persistent_workers=True
    )


def main():
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True

    data_path = Path("data/raw/binance_btcusdt_5m.parquet")
    model_dir = Path("models/patchtst")
    model_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df):,} real 5m candles")

    split_idx = int(len(df) * 0.8)
    train_loader = prepare_data_for_patchtst(df.iloc[:split_idx], sequence_length=512, horizon=96, batch_size=256)
    val_loader   = prepare_data_for_patchtst(df.iloc[split_idx:], sequence_length=512, horizon=96, batch_size=256)

    c_in = train_loader.dataset[0][0].shape[1]
    logger.info(f"Input feature count: {c_in}")

    model = PatchTST(c_in=c_in, c_out=1, seq_len=512, pred_len=96)

    lightning_model = PatchTSTLightning(model, learning_rate=1e-4, horizon=96)

    trainer = pl.Trainer(
        max_epochs=60,
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=15, mode="min", verbose=True),
            ModelCheckpoint(
                dirpath=str(model_dir),
                filename="patchtst_best",
                save_top_k=3,
                monitor="val_loss",
                save_last=True
            )
        ],
        logger=pl.loggers.TensorBoardLogger("lightning_logs", name="patchtst_math"),
        gradient_clip_val=1.0,
        accumulate_grad_batches=8,
        enable_progress_bar=True,
    )

    logger.info("🚀 Starting PatchTST training on GPU (high utilization mode)...")
    trainer.fit(lightning_model, train_loader, val_loader)

    logger.info("✅ Training session completed!")


if __name__ == "__main__":
    main()