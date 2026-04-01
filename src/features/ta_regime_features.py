import pandas as pd
import numpy as np
import ta
from scipy.fft import fft, fftfreq
from typing import Tuple

def dummy_regime_classifier(df: pd.DataFrame) -> pd.Series:
    """Improved Markov-style regime classifier"""
    if df.empty or len(df) < 50:
        return pd.Series("ranging", index=df.index, dtype="object")

    close = df["close"]
    returns = close.pct_change().fillna(0)

    short_return = returns.rolling(window=20, min_periods=15).mean()
    volatility = returns.rolling(window=50, min_periods=30).std()
    vol_baseline = volatility.rolling(window=100, min_periods=60).quantile(0.7)

    regime = pd.Series("ranging", index=df.index, dtype="object")

    trending_up = (short_return > 0.0004) & (volatility <= vol_baseline)
    trending_down = (short_return < -0.0004) & (volatility <= vol_baseline)

    regime[trending_up] = "trending_up"
    regime[trending_down] = "trending_down"

    return regime


def build_mathematical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clean mathematical feature engineering for BTC 5m prediction"""
    if df is None or df.empty:
        return pd.DataFrame()

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df.get("volume", pd.Series(0.0, index=df.index))
    returns = close.pct_change().fillna(0)

    feat = pd.DataFrame(index=df.index)

    # === Traditional TA ===
    feat["ema_20"] = close.ewm(span=20, min_periods=1).mean()
    feat["ema_50"] = close.ewm(span=50, min_periods=1).mean()
    feat["ema_200"] = close.ewm(span=200, min_periods=1).mean()

    feat["rsi_14"] = ta.momentum.RSIIndicator(close, window=14).rsi().fillna(50)
    feat["roc_10"] = ta.momentum.ROCIndicator(close, window=10).roc().fillna(0)

    atr = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14)
    feat["atr_14"] = atr.average_true_range().fillna(0)

    # Volume z-score
    vol_mean = volume.rolling(50, min_periods=20).mean()
    vol_std = volume.rolling(50, min_periods=20).std()
    feat["vol_zscore_50"] = (volume - vol_mean) / (vol_std + 1e-8)

    # === Markov Regime ===
    feat["regime"] = dummy_regime_classifier(df)
    feat["regime_trending_up"] = (feat["regime"] == "trending_up").astype(int)

    state_map = {"trending_up": 0, "trending_down": 1, "ranging": 2}
    states = feat["regime"].map(state_map).fillna(2).astype(int)
    feat["markov_prob_up"] = states.rolling(60, min_periods=30).apply(
        lambda x: np.mean(x == 0) if len(x) > 0 else 0.0, raw=True
    ).fillna(0.333)

    # === Stochastic Features ===
    feat["realized_vol_20"] = returns.rolling(20).std() * np.sqrt(288)
    feat["stochastic_momentum"] = returns.rolling(20).mean() / (feat["realized_vol_20"] + 1e-8)
    feat["jump_indicator"] = (np.abs(returns) > 3 * returns.rolling(100).std()).astype(int)

    # === Fourier Transform (Cycle Detection) ===
    window = 256
    feat["fourier_trend"] = 0.0
    feat["fourier_cycle_strength"] = 0.0

    for i in range(window, len(df)):
        segment = close.iloc[i-window:i].values
        fft_vals = fft(segment)
        freqs = fftfreq(window, d=1/288)

        low_freq_mask = np.abs(freqs) < 5
        trend_power = np.abs(fft_vals[low_freq_mask]).mean()
        cycle_power = np.sum(np.abs(fft_vals[low_freq_mask])) / (np.sum(np.abs(fft_vals)) + 1e-8)

        feat.loc[feat.index[i-1], "fourier_trend"] = trend_power
        feat.loc[feat.index[i-1], "fourier_cycle_strength"] = cycle_power

    # === Final Cleanup ===
    feat = feat.ffill().bfill().fillna(0.0)

    # Select core features (we'll start with these)
    numeric_columns = [
        "ema_20", "ema_50", "ema_200", "rsi_14", "roc_10", "atr_14",
        "vol_zscore_50", "regime_trending_up", "markov_prob_up",
        "realized_vol_20", "stochastic_momentum", "jump_indicator",
        "fourier_trend", "fourier_cycle_strength"
    ]

    final_feat = feat[numeric_columns + ["regime"]].copy()

    print(f"✅ build_mathematical_features completed → {len(numeric_columns)} numeric + 'regime'")

    return final_feat