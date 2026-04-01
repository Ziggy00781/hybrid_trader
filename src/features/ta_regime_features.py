import pandas as pd
import numpy as np
import ta
from typing import Tuple

def dummy_regime_classifier(df: pd.DataFrame) -> pd.Series:
    """
    Simple but effective market regime classifier.
    Returns: 'trending_up', 'trending_down', or 'ranging'
    """
    if df.empty:
        return pd.Series("ranging", index=df.index, dtype="object")

    close = df["close"]
    returns = close.pct_change().fillna(0)
    
    # Rolling statistics
    short_return = returns.rolling(window=20, min_periods=10).mean()
    volatility = returns.rolling(window=50, min_periods=20).std()
    vol_baseline = volatility.rolling(window=100, min_periods=50).quantile(0.7)

    regime = pd.Series("ranging", index=df.index, dtype="object")

    # Trending Up: positive drift + relatively low volatility
    trending_up = (short_return > 0.0003) & (volatility < vol_baseline)
    regime[trending_up] = "trending_up"

    # Trending Down
    trending_down = (short_return < -0.0003) & (volatility < vol_baseline)
    regime[trending_down] = "trending_down"

    return regime


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build consistent features for LightGBM model.
    Returns exactly 9 numeric features + 'regime' column.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    # Core price & volume series
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df.get("volume", pd.Series(0.0, index=df.index))

    feat = pd.DataFrame(index=df.index)

    # ==================== 1. Trend Features ====================
    feat["ema_20"] = close.ewm(span=20, min_periods=1).mean()
    feat["ema_50"] = close.ewm(span=50, min_periods=1).mean()
    feat["ema_200"] = close.ewm(span=200, min_periods=1).mean()

    # ==================== 2. Momentum Features ====================
    feat["rsi_14"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    feat["roc_10"] = ta.momentum.ROCIndicator(close, window=10).roc()

    # ==================== 3. Volatility Features ====================
    atr = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14)
    feat["atr_14"] = atr.average_true_range()

    # ==================== 4. Volume Features ====================
    vol_mean = volume.rolling(window=50, min_periods=20).mean()
    vol_std = volume.rolling(window=50, min_periods=20).std()
    feat["vol_zscore_50"] = (volume - vol_mean) / (vol_std + 1e-8)

    # ==================== 5. Regime Features ====================
    feat["regime"] = dummy_regime_classifier(df)
    feat["regime_trending_up"] = (feat["regime"] == "trending_up").astype(int)

    # ==================== Final Cleanup ====================
    # Fill any remaining NaNs (important for early rows)
    feat = feat.fillna(method='ffill').fillna(method='bfill').fillna(0)

    # Ensure consistent column order (important for model compatibility)
    feature_columns = [
        "ema_20", "ema_50", "ema_200",
        "rsi_14", "roc_10",
        "atr_14",
        "vol_zscore_50",
        "regime_trending_up"
    ]

    # Return only the 8 numeric features + regime (total 9 features when regime is dropped)
    final_feat = feat[feature_columns + ["regime"]].copy()

    print(f"✅ build_features completed: {final_feat.shape[1]} columns "
          f"({len(feature_columns)} numeric + regime)")

    return final_feat