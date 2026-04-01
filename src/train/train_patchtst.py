import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import joblib
import logging
from datetime import datetime

from src.features.ta_regime_features import build_mathematical_features
from src.utils.patchtst_dataset import PatchTSTDataset  # we'll create/improve this if needed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PatchTSTTrainer(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-4, horizon=96):
        super().__init__()
        self.model = model
        self.lr = learning_rate
        self.horizon = horizon  # prediction steps ahead (96 = ~8 hours)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

def prepare_data_for_patchtst(df: pd.DataFrame, sequence_length=512, horizon=96):
    """Prepare data with mathematical features injected"""
    feats = build_mathematical_features(df)
    # Align price data with features
    price_data = df[['open', 'high', 'low', 'close', 'volume']].loc[feats.index]
    
    # Combine price + rich mathematical features
    combined = pd.concat([price_data, feats.drop(columns=['regime'])], axis=1)
    combined = combined.fillna(0)
    
    # Normalize (important for PatchTST)
    scaler = joblib.load("data/processed/patchtst/scaler_params.pkl") if Path("data/processed/patchtst/scaler_params.pkl").exists() else None
    # ... add proper scaling logic here (you can expand later)
    
    dataset = PatchTSTDataset(combined.values, sequence_length=sequence_length, horizon=horizon)
    return DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

def main():
    # Load your existing partially trained model
    model_path = Path("models/patchtst/patchtst_best.pt")  # or your best checkpoint
    if model_path.exists():
        logger.info(f"Loading existing model from {model_path}")
        # Load your enhanced_patchtst model here (adjust based on your enhanced_patchtst.py)
        from src.models.enhanced_patchtst import EnhancedPatchTST
        model = EnhancedPatchTST.load_from_checkpoint(model_path)
    else:
        logger.warning("No existing checkpoint found. Initializing new model.")
        from src.models.enhanced_patchtst import EnhancedPatchTST
        model = EnhancedPatchTST()

    # Load full real data
    df = pd.read_parquet("data/raw/binance_btcusdt_5m.parquet")
    logger.info(f"Loaded {len(df):,} real 5m candles")

    train_loader = prepare_data_for_patchtst(df.iloc[:-int(len(df)*0.2)], sequence_length=512, horizon=96)
    val_loader = prepare_data_for_patchtst(df.iloc[-int(len(df)*0.2):], sequence_length=512, horizon=96)

    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed",           # Good for your RTX 5070
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=8, mode="min"),
            ModelCheckpoint(dirpath="models/patchtst/", filename="patchtst_best", save_top_k=1, monitor="val_loss")
        ],
        logger=pl.loggers.TensorBoardLogger("lightning_logs", name="patchtst_math"),
        gradient_clip_val=1.0,
        accumulate_grad_batches=4,      # Effective larger batch
    )

    logger.info("Starting rigorous PatchTST fine-tuning with mathematical features...")
    trainer.fit(model, train_loader, val_loader)

    logger.info("Training completed. Best model saved in models/patchtst/")

if __name__ == "__main__":
    main()