# src/train/train_enhanced_patchtst.py
import torch
import torch.nn as nn
import numpy as np
import logging
from torch.utils.data import DataLoader, TensorDataset
from src.models.patchtst import PatchTST  # Your existing model
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class EnhancedPatchTSTTrainer:
    """
    Enhanced trainer for PatchTST with comprehensive evaluation
    """
    
    def __init__(self, model_config=None):
        self.model_config = model_config or {
            'n_layers': 4,
            'n_heads': 8,
            'd_model': 256,
            'd_ff': 512,
            'dropout': 0.2,
            'patch_len': 16,
            'stride': 8
        }
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
    
    def load_dataset(self, dataset_path: str):
        """Load prepared dataset"""
        logger.info(f"Loading dataset from {dataset_path}")
        dataset_dict = torch.load(dataset_path)
        
        features = torch.FloatTensor(dataset_dict['features'])
        targets = torch.FloatTensor(dataset_dict['targets'])
        
        # Split data
        n_total = len(features)
        n_train = int(0.8 * n_total)
        n_val = int(0.1 * n_total)
        
        train_features, train_targets = features[:n_train], targets[:n_train]
        val_features, val_targets = features[n_train:n_train+n_val], targets[n_train:n_train+n_val]
        test_features, test_targets = features[n_train+n_val:], targets[n_train+n_val:]
        
        # Create datasets
        train_dataset = TensorDataset(train_features, train_targets)
        val_dataset = TensorDataset(val_features, val_targets)
        test_dataset = TensorDataset(test_features, test_targets)
        
        return train_dataset, val_dataset, test_dataset
    
    def create_model(self, input_dim: int):
        """Create PatchTST model"""
        self.model = PatchTST(
            c_in=input_dim,
            c_out=1,
            seq_len=1024,
            pred_len=1,
            **self.model_config
        )
        self.model.to(self.device)
        logger.info(f"Model created with {sum(p.numel() for p in self.model.parameters()):,} parameters")
    
    def train_model(
        self, 
        train_dataset, 
        val_dataset, 
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-4
    ):
        """Train the model"""
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Create model if not exists
        if self.model is None:
            input_dim = train_dataset[0][0].shape[1]  # Features per timestep
            self.create_model(input_dim)
        
        # Setup optimizer and loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        criterion = nn.MSELoss()
        
        # Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            for batch_features, batch_targets in train_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                self.optimizer.zero_grad()
                predictions = self.model(batch_features)
                loss = criterion(predictions.squeeze(), batch_targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_features, batch_targets in val_loader:
                    batch_features = batch_features.to(self.device)
                    batch_targets = batch_targets.to(self.device)
                    
                    predictions = self.model(batch_features)
                    loss = criterion(predictions.squeeze(), batch_targets)
                    val_loss += loss.item()
            
            # Calculate average losses
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            # Update learning rate
            self.scheduler.step(avg_val_loss)
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.save_checkpoint('models/patchtst_enhanced_best.pt')
            
            # Log progress
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        # Plot training curves
        self.plot_training_curves(train_losses, val_losses)
        
        return train_losses, val_losses
    
    def evaluate_model(self, test_dataset):
        """Evaluate model on test set"""
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        criterion = nn.MSELoss()
        
        self.model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            total_loss = 0.0
            for batch_features, batch_targets in test_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                batch_predictions = self.model(batch_features)
                loss = criterion(batch_predictions.squeeze(), batch_targets)
                total_loss += loss.item()
                
                predictions.extend(batch_predictions.cpu().numpy())
                targets.extend(batch_targets.cpu().numpy())
        
        # Calculate metrics
        predictions = np.array(predictions).flatten()
        targets = np.array(targets).flatten()
        
        mse = mean_squared_error(targets, predictions)
        mae = mean_absolute_error(targets, predictions)
        rmse = np.sqrt(mse)
        
        # Correlation
        correlation = np.corrcoef(targets, predictions)[0, 1]
        
        logger.info(f"Test Results:")
        logger.info(f"  MSE: {mse:.6f}")
        logger.info(f"  MAE: {mae:.6f}")
        logger.info(f"  RMSE: {rmse:.6f}")
        logger.info(f"  Correlation: {correlation:.4f}")
        
        # Plot predictions vs actual
        self.plot_predictions(targets, predictions)
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'correlation': correlation,
            'predictions': predictions,
            'targets': targets
        }
    
    def plot_training_curves(self, train_losses, val_losses):
        """Plot training and validation loss curves"""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_predictions(self, targets, predictions):
        """Plot predictions vs actual values"""
        plt.figure(figsize=(12, 6))
        plt.scatter(targets, predictions, alpha=0.5)
        plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Predictions vs Actual Values')
        plt.grid(True)
        plt.savefig('predictions_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Time series plot of first 200 samples
        plt.figure(figsize=(15, 6))
        plt.plot(targets[:200], label='Actual', alpha=0.7)
        plt.plot(predictions[:200], label='Predicted', alpha=0.7)
        plt.xlabel('Sample')
        plt.ylabel('Log Return')
        plt.title('Time Series: Actual vs Predicted (First 200 Samples)')
        plt.legend()
        plt.grid(True)
        plt.savefig('predictions_timeseries.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        checkpoint = {
            'state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'model_config': self.model_config
        }
        torch.save(checkpoint, path)
        logger.info(f"Model checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        if self.model is None:
            self.model_config = checkpoint.get('model_config', self.model_config)
            # Recreate model with saved config
            # You'd need to know input dimensions here
        self.model.load_state_dict(checkpoint['state_dict'])
        if self.optimizer and checkpoint['optimizer_state_dict']:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Model checkpoint loaded from {path}")

def main():
    """Main training function"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logger.info("Starting Enhanced PatchTST Training...")
    
    # Initialize trainer
    trainer = EnhancedPatchTSTTrainer()
    
    # Load dataset
    train_dataset, val_dataset, test_dataset = trainer.load_dataset("data/processed/enhanced_dataset.pt")
    logger.info(f"Dataset loaded: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    # Train model
    logger.info("Starting training...")
    train_losses, val_losses = trainer.train_model(
        train_dataset, 
        val_dataset, 
        epochs=100,
        batch_size=32,
        learning_rate=1e-4
    )
    
    # Evaluate model
    logger.info("Evaluating model...")
    results = trainer.evaluate_model(test_dataset)
    
    # Save final model
    trainer.save_checkpoint('models/patchtst_enhanced_final.pt')
    
    logger.info("Training completed successfully!")
    logger.info(f"Final test correlation: {results['correlation']:.4f}")

if __name__ == "__main__":
    main()