import torch
import pandas as pd
from pathlib import Path
import joblib
import logging
from datetime import datetime

from src.features.ta_regime_features import build_mathematical_features
from src.models.patchtst import PatchTST

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_best_model(model_dir="models/patchtst"):
    checkpoint_path = Path(model_dir) / "patchtst_best.ckpt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Best model not found at {checkpoint_path}")

    logger.info(f"Loading best model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    c_in = 19
    model = PatchTST(c_in=c_in, c_out=1, seq_len=512, pred_len=96)

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model.eval()
    logger.info("✅ Model loaded successfully")
    return model


def predict_signal(recent_df: pd.DataFrame, model, sequence_length=512):
    """Improved signal logic with better calibration"""
    feats = build_mathematical_features(recent_df)
    price_cols = ['open', 'high', 'low', 'close', 'volume']
    price_data = recent_df[price_cols].loc[feats.index]

    combined = pd.concat([price_data, feats.drop(columns=['regime'], errors='ignore')], axis=1)
    combined = combined.ffill().bfill().fillna(0.0)

    scaler_path = Path("data/processed/patchtst/scaler.pkl")
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None

    if scaler:
        scaled_values = scaler.transform(combined.values)
    else:
        scaled_values = combined.values

    input_tensor = torch.tensor(scaled_values[-sequence_length:], dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        raw_pred = model(input_tensor).item()

    current_price = recent_df['close'].iloc[-1]
    current_regime = feats['regime'].iloc[-1]

    # Calibrated estimated return
    estimated_return = raw_pred * 0.0065   # Adjusted scaling factor

    vol = recent_df['close'].pct_change().rolling(20).std().iloc[-1] * 100

    # More sensitive signal logic
    if estimated_return > 0.0025 and current_regime in ["trending_up", "ranging"]:
        signal = "LONG"
        confidence = min(82, 45 + estimated_return * 12000)
        tp_price = current_price * (1 + estimated_return * 1.8)
        sl_price = current_price * (1 - estimated_return * 0.75)
        risk_reward = 1.8 / 0.75
    elif estimated_return > 0.001 and current_regime == "trending_up":
        signal = "LONG"
        confidence = min(70, 40 + estimated_return * 10000)
        tp_price = current_price * (1 + estimated_return * 1.6)
        sl_price = current_price * (1 - estimated_return * 0.8)
        risk_reward = 1.6 / 0.8
    elif estimated_return < -0.0025 and current_regime in ["trending_down", "ranging"]:
        signal = "SHORT"
        confidence = min(80, 45 + abs(estimated_return) * 11000)
        tp_price = current_price * (1 + estimated_return * 1.7)
        sl_price = current_price * (1 - estimated_return * 0.8)
        risk_reward = 1.7 / 0.8
    elif estimated_return < -0.001 and current_regime == "trending_down":
        signal = "SHORT"
        confidence = min(68, 38 + abs(estimated_return) * 9500)
        tp_price = current_price * (1 + estimated_return * 1.5)
        sl_price = current_price * (1 - estimated_return * 0.85)
        risk_reward = 1.5 / 0.85
    else:
        signal = "FLAT"
        confidence = 85
        tp_price = None
        sl_price = None
        risk_reward = None

    return {
        "signal": signal,
        "confidence": round(confidence, 1),
        "estimated_return_pct": round(estimated_return * 100, 3),
        "current_price": round(current_price, 2),
        "tp_price": round(tp_price, 2) if tp_price else None,
        "sl_price": round(sl_price, 2) if sl_price else None,
        "risk_reward": round(risk_reward, 2) if risk_reward else None,
        "regime": current_regime,
        "volatility": round(vol, 3),
        "horizon": "6-8 hours",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
    }


if __name__ == "__main__":
    df = pd.read_parquet("data/raw/binance_btcusdt_5m.parquet")
    recent_df = df.tail(1500)   # ~5 days of context

    model = load_best_model()
    result = predict_signal(recent_df, model)

    print("\n" + "="*75)
    print("                  PATCHTST TRADING SIGNAL")
    print("="*75)
    print(f"Signal             : {result['signal']}")
    print(f"Confidence         : {result['confidence']}%")
    print(f"Estimated Move     : {result['estimated_return_pct']}%")
    print(f"Current Price      : ${result['current_price']}")
    print(f"Regime             : {result['regime']}")
    print(f"Volatility (20)    : {result['volatility']}%")

    if result['tp_price']:
        print(f"Take Profit        : ${result['tp_price']}")
    if result['sl_price']:
        print(f"Stop Loss          : ${result['sl_price']}")
    if result['risk_reward']:
        print(f"Risk/Reward        : {result['risk_reward']}:1")

    print(f"Horizon            : {result['horizon']}")
    print(f"Generated at       : {result['timestamp']}")
    print("="*75)