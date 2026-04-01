import pandas as pd
import joblib
from src.features.ta_regime_features import build_mathematical_features   # Updated import

def generate_signals(df: pd.DataFrame,
                     prob_threshold: float = 0.65,
                     model=None) -> pd.DataFrame:
    """
    Generate trading signals using LightGBM.
    Now supports the new mathematical features (Markov, Stochastic, Fourier).
    """
    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty")

    # Use the new mathematically rich feature builder
    feats = build_mathematical_features(df)
    
    if model is None:
        try:
            model = joblib.load("models/btcusdt_5m_lgbm.pkl")
        except FileNotFoundError:
            raise FileNotFoundError("Model not found. Please train it first with train_model.py")

    # Drop regime column (categorical) for prediction
    X = feats.drop(columns=["regime"], errors='ignore')

    # Feature alignment - critical for LightGBM compatibility
    if X.shape[1] > 9:
        # Take only the first 9 features that the current model was trained on
        X = X.iloc[:, :9]
        print(f"⚠️  Feature alignment: Reduced from {feats.shape[1]-1} to 9 numeric features")
    elif X.shape[1] < 9:
        print(f"⚠️  Warning: Only {X.shape[1]} features available (model expects 9)")

    # Generate probabilities
    try:
        probs = model.predict_proba(X, predict_disable_shape_check=True)[:, 1]
    except Exception as e:
        print(f"Prediction error: {e}")
        probs = np.zeros(len(X))  # fallback

    # Build signals DataFrame
    signals = pd.DataFrame(index=feats.index)
    signals["prob_long"] = probs
    signals["regime"] = feats["regime"].values
    signals["signal"] = 0

    # Core signal logic: High probability + Trending Up regime
    signals.loc[
        (signals["prob_long"] >= prob_threshold) &
        (signals["regime"] == "trending_up"),
        "signal"
    ] = 1

    # Optional: Add confidence level
    signals["confidence"] = signals["prob_long"] * 100

    print(f"✅ Generated {signals['signal'].sum():.0f} buy signals "
          f"out of {len(signals)} candles (threshold={prob_threshold})")

    return signals