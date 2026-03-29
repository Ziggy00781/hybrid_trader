import pandas as pd
import numpy as np
import ta  # pip install ta

from src.models.timegpt_inference import timegpt_predict

def dummy_regime_classifier(df: pd.DataFrame) -> pd.Series:
    ema50 = df["close"].ewm(span=50).mean()
    regime = np.where(df["close"] > ema50, "trending_up", "trending_down")
    return pd.Series(regime, index=df.index)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # -------------------------
    # Trend indicators
    # -------------------------
    feat["ema_20"] = close.ewm(span=20).mean()
    feat["ema_50"] = close.ewm(span=50).mean()
    feat["ema_200"] = close.ewm(span=200).mean()

    # -------------------------
    # Momentum indicators
    # -------------------------
    feat["rsi_14"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    feat["roc_10"] = ta.momentum.ROCIndicator(close, window=10).roc()

    # -------------------------
    # Volatility indicators
    # -------------------------
    feat["atr_14"] = ta.volatility.AverageTrueRange(
        high=high, low=low, close=close, window=14
    ).average_true_range()

    # -------------------------
    # Volume indicators
    # -------------------------
    feat["vol_zscore_50"] = (
        (volume - volume.rolling(50).mean()) /
        (volume.rolling(50).std() + 1e-9)
    )

    # -------------------------
    # Regime classification
    # -------------------------
    feat["regime"] = dummy_regime_classifier(df)
    feat["regime_trending_up"] = (feat["regime"] == "trending_up").astype(int)
    feat["regime_trending_down"] = (feat["regime"] == "trending_down").astype(int)

    # -------------------------
    # TimesFM deep forecasting features
    # -------------------------
logret = np.log(close / close.shift(1))
window = 256
horizon = 12

tgpt_mean = []
tgpt_std = []
tgpt_up = []
tgpt_down = []

for i in range(len(df)):
    if i < window:
        tgpt_mean.append(np.nan)
        tgpt_std.append(np.nan)
        tgpt_up.append(np.nan)
        tgpt_down.append(np.nan)
        continue

    seq = logret.iloc[i-window:i].values
    preds = timegpt_predict(seq, horizon=horizon)

    tgpt_mean.append(preds["tgpt_mean"])
    tgpt_std.append(preds["tgpt_std"])
    tgpt_up.append(preds["tgpt_up_prob"])
    tgpt_down.append(preds["tgpt_down_prob"])

feat["tgpt_mean"] = tgpt_mean
feat["tgpt_std"] = tgpt_std
feat["tgpt_up_prob"] = tgpt_up
feat["tgpt_down_prob"] = tgpt_down

    # -------------------------
    # Final cleanup
    # -------------------------
    feat = feat.dropna()
    return feat