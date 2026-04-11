# src/train/train_enhanced_patchtst.py
import torch
import torch.nn as nn
import numpy as np
import logging
import pandas as pd
import os
from torch.utils.data import DataLoader, TensorDataset
from src.models.patchtst import PatchTST
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class EnhancedPatchTSTTrainer:
    """Memory-optimized PatchTST Trainer with rich features + log returns"""

    def __init__(self, model_config=None):
        self.model_config = model_config or {
            'd_model': 128,
            'n_heads': 8,
            'e_layers': 3,
            'd_ff': 256,
            'dropout': 0.1,
            'patch_len': 16,
            'stride': 8,
        }
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        logger.info(f"Using device: {self.device}")

    def load_and_prepare_data(
        self,
        data_path: str = "data/processed/BTC_USDT_5m_enhanced.parquet",
        seq_len: int = 256,      # Reduced for memory
        pred_len: int = 20,
        max_features: int = 48,  # Reduced for memory
    ):
        """Memory-optimized data loading"""
        logger.info(f"Loading enhanced dataset from {data_path} (memory optimized)")
        df = pd.read_parquet(data_path)

        logger.info(f"Loaded {len(df):,} bars with {df.shape[1]} features")

        # Select numeric features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Prioritize important features
        priority = ['open', 'high', 'low', 'close', 'volume', 'adx', 'atr', 'atr_ratio',
                    'regime', 'volume_ratio', 'volume_zscore']

        selected_cols = [col for col in priority if col in numeric_cols]
        remaining = [col for col in numeric_cols if col not in selected_cols]
        selected_cols += remaining[:max_features - len(selected_cols)]

        features = df[selected_cols].values
        features_scaled = self.feature_scaler.fit_transform(features)

        # Log returns target
        df['log_close'] = np.log(df['close'])
        df['target_log_return'] = df['log_close'].shift(-pred_len) - df['log_close']
        df = df.dropna(subset=['target_log_return'])

        targets = df['target_log_return'].values
        targets_scaled = self.target_scaler.fit_transform(targets.reshape(-1, 1)).flatten()

        # Create sequences (step=2 to further reduce memory)
        X, y = [], []
        step = 2
        for i in range(0, len(features_scaled) - seq_len - pred_len, step):
            X.append(features_scaled[i:i + seq_len])
            y.append(targets_scaled[i + seq_len])

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32).reshape(-1, 1)

        logger.info(f"Created sequences → X: {X.shape} ({len(selected_cols)} features), y: {y.shape}")

        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)

        # Split 80/10/10
        n_total = len(X_tensor)
        n_train = int(0.8 * n_total)
        n_val = int(0.1 * n_total)

        train_ds = TensorDataset(X_tensor[:n_train], y_tensor[:n_train])
        val_ds   = TensorDataset(X_tensor[n_train:n_train + n_val], y_tensor[n_train:n_train + n_val])
        test_ds  = TensorDataset(X_tensor[n_train + n_val:], y_tensor[n_train + n_val:])

        return train_ds, val_ds, test_ds, selected_cols

    def create_model(self, input_dim: int, seq_len: int = 256, pred_len: int = 20):
        logger.info(f"Creating PatchTST | input_dim={input_dim}, seq_len={seq_len}, pred_len={pred_len}")

        self.model = PatchTST(
            c_in=input_dim,
            c_out=1,
            seq_len=seq_len,
            pred_len=pred_len,
            d_model=self.model_config['d_model'],
            n_heads=self.model_config['n_heads'],
            e_layers=self.model_config['e_layers'],
            d_ff=self.model_config['d_ff'],
            dropout=self.model_config['dropout'],
            patch_len=self.model_config['patch_len'],
            stride=self.model_config['stride'],
        )
        self.model.to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"✅ PatchTST model created with {total_params:,} parameters")

    def train_model(
        self,
        train_dataset,
        val_dataset,
        epochs: int = 60,
        batch_size: int = 16,      # Smaller batch size for stability
        learning_rate: float = 1e-4,
    ):
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        if self.model is None:
            sample_x, _ = train_dataset[0]
            input_dim = sample_x.shape[1]
            self.create_model(input_dim=input_dim)

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=8
        )

        criterion = nn.MSELoss()

        best_val_loss = float('inf')
        train_losses, val_losses = [], []

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                self.optimizer.zero_grad()
                preds = self.model(batch_x)
                loss = criterion(preds.squeeze(), batch_y.squeeze())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                train_loss += loss.item()

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    preds = self.model(batch_x)
                    loss = criterion(preds.squeeze(), batch_y.squeeze())
                    val_loss += loss.item()

            avg_train = train_loss / len(train_loader)
            avg_val = val_loss / len(val_loader)
            train_losses.append(avg_train)
            val_losses.append(avg_val)

            self.scheduler.step(avg_val)

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                self.save_checkpoint('models/patchtst_enhanced_best.pt')

            if epoch % 10 == 0 or epoch == epochs - 1:
                logger.info(f"Epoch {epoch:3d} | Train: {avg_train:.6f} | Val: {avg_val:.6f} | Best: {best_val_loss:.6f}")

        self.plot_training_curves(train_losses, val_losses)
        return train_losses, val_losses

    def evaluate_model(self, test_dataset):
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        criterion = nn.MSELoss()

        self.model.eval()
        predictions, targets = [], []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                preds = self.model(batch_x)
                predictions.extend(preds.cpu().numpy())
                targets.extend(batch_y.cpu().numpy())

        predictions = np.array(predictions).flatten()
        targets = np.array(targets).flatten()

        mse = mean_squared_error(targets, predictions)
        mae = mean_absolute_error(targets, predictions)
        rmse = np.sqrt(mse)
        correlation = np.corrcoef(targets, predictions)[0, 1]

        pred_returns = self.target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        true_returns = self.target_scaler.inverse_transform(targets.reshape(-1, 1)).flatten()

        logger.info(f"Test Results → MSE: {mse:.6f} | MAE: {mae:.6f} | RMSE: {rmse:.6f} | Corr: {correlation:.4f}")
        logger.info(f"Mean pred return: {pred_returns.mean():.6f} | Mean actual: {true_returns.mean():.6f}")

        self.plot_predictions(targets, predictions)
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'correlation': correlation
        }

    def plot_training_curves(self, train_losses, val_losses):
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('PatchTST Training Progress (Optimized)')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_predictions(self, targets, predictions):
        plt.figure(figsize=(12, 6))
        plt.scatter(targets, predictions, alpha=0.5, s=8)
        plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', lw=2)
        plt.xlabel('Actual Scaled Log Return')
        plt.ylabel('Predicted Scaled Log Return')
        plt.title('Predictions vs Actual')
        plt.grid(True)
        plt.savefig('predictions_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()

    def save_checkpoint(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'state_dict': self.model.state_dict(),
            'model_config': self.model_config,
            'feature_scaler': self.feature_scaler,
            'target_scaler': self.target_scaler,
        }, path)
        logger.info(f"Model checkpoint saved → {path}")


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    logger.info("🚀 Starting Memory-Optimized PatchTST Training...")

    trainer = EnhancedPatchTSTTrainer()

    train_ds, val_ds, test_ds, feature_cols = trainer.load_and_prepare_data(
        data_path="data/processed/BTC_USDT_5m_enhanced.parquet",
        seq_len=256,
        pred_len=20,
        max_features=48,
    )

    logger.info(f"Using {len(feature_cols)} features")
    logger.info(f"Datasets ready → Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    trainer.train_model(train_ds, val_ds, epochs=60, batch_size=16)

    results = trainer.evaluate_model(test_ds)

    trainer.save_checkpoint('models/patchtst_enhanced_final.pt')
    logger.info(f"✅ Training completed! Final correlation: {results['correlation']:.4f}")


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    main()