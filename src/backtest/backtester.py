import pandas as pd
from src.features.ta_regime_features import build_features
import joblib

def generate_signals(df: pd.DataFrame,
                     prob_threshold: float = 0.6,
                     model=None) -> pd.DataFrame:
    """Generate trading signals using LightGBM with feature alignment"""
    feats = build_features(df)
    
    if model is None:
        model = joblib.load("models/btcusdt_5m_lgbm.pkl")
    
    # Drop non-numeric / categorical columns
    X = feats.drop(columns=["regime"], errors='ignore')
    
    # CRITICAL: Align features to exactly what the model was trained on
    # We take only the first 9 numeric features (matching training)
    if X.shape[1] > 9:
        X = X.iloc[:, :9]          # Take first 9 columns
        print(f"⚠️  Feature count reduced from {feats.shape[1]} to 9 for model compatibility")
    elif X.shape[1] < 9:
        print(f"⚠️  Warning: Only {X.shape[1]} features available, model expects 9")
    
    # Predict
    probs = model.predict_proba(X, predict_disable_shape_check=True)[:, 1]
    
    # Build signals DataFrame
    signals = pd.DataFrame(index=feats.index)
    signals["prob_long"] = probs
    signals["regime"] = feats["regime"].values
    signals["signal"] = 0
    
    # Apply threshold + regime filter (this is your original logic)
    signals.loc[
        (signals["prob_long"] >= prob_threshold) &
        (signals["regime"] == "trending_up"),
        "signal"
    ] = 1
    
    return signals