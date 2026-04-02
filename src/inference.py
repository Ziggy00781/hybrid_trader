import torch
import pandas as pd
from pathlib import Path
import joblib
import logging

from src.features.ta_regime_features import build_mathematical_features
from src.models.patchtst import PatchTST
from src.utils.patchtst_dataset import PatchTSTDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_best_model(model_dir="models/patchtst"):
    """Load the best PatchTST model"""
    checkpoint_path = Path(model_dir) / "patchtst_best.ckpt"
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Best model not found at {checkpoint_path}")

    logger.info(f"Loading best model from {checkpoint_path}")

    # Load the Lightning checkpoint and extract the raw model
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get the number of input features from the checkpoint or assume 19
    c_in = 19  # from our feature set

    model = PatchTST(c_in=c_in, c_out=1, seq_len=512, pred_len=96)
    
    # Load state dict safely
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        # Remove 'model.' prefix if present (Lightning adds it)
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model.eval()
    logger.info("✅ Model loaded successfully")
    return model


def predict_signal(recent_df: pd.DataFrame, model, sequence_length=512):
    """Generate trading signal from recent data"""
    # Build features
    feats = build_mathematical_features(recent_df)
    price_cols = ['open', 'high', 'low', 'close', 'volume']
    price_data = recent_df[price_cols].loc[feats.index]

    combined = pd.concat([price_data, feats.drop(columns=['regime'], errors='ignore')], axis=1)
    combined = combined.ffill().bfill().fillna(0.0)

    # Scale (use the same scaler as training)
    scaler_path = Path("data/processed/patchtst/scaler.pkl")
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        scaled_values = scaler.transform(combined.values)
    else:
        logger.warning("Scaler not found, using raw values")
        scaled_values = combined.values

    # Prepare input tensor
    input_tensor = torch.tensor(scaled_values[-sequence_length:], dtype=torch.float32).unsqueeze(0)

    # Inference
    with torch.no_grad():
        prediction = model(input_tensor).item()

    # Simple signal logic (can be improved later)
    if prediction > 0.001:
        signal = "LONG"
        confidence = min(0.95, prediction * 50)  # rough scaling
    elif prediction < -0.001:
        signal = "SHORT"
        confidence = min(0.95, abs(prediction) * 50)
    else:
        signal = "FLAT"
        confidence = 0.8

    return {
        "signal": signal,
        "confidence": round(confidence * 100, 1),
        "raw_prediction": round(prediction, 6),
        "suggested_horizon": "6-8 hours"
    }


if __name__ == "__main__":
    # Example usage
    df = pd.read_parquet("data/raw/binance_btcusdt_5m.parquet")
    recent_df = df.tail(1000)   # last ~3.5 days

    model = load_best_model()
    result = predict_signal(recent_df, model)

    print("\n=== PatchTST Signal ===")
    print(f"Signal     : {result['signal']}")
    print(f"Confidence : {result['confidence']}%")
    print(f"Raw pred   : {result['raw_prediction']}")
    print(f"Horizon    : {result['suggested_horizon']}")