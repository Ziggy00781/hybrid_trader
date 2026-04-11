# src/features/feature_engineering.py
"""
Feature Engineering Pipeline for Hybrid Trader
Adds TA indicators, Fourier cycles, volume features, and regime detection.
"""

import pandas as pd
import numpy as np
from ta import add_all_ta_features
from ta.volatility import AverageTrueRange
import os
import logging
from scipy.fft import fft, fftfreq

logger = logging.getLogger(__name__)

def add_fourier_features(df: pd.DataFrame, column: str = 'close', n_harmonics: int = 5, window: int = 256) -> pd.DataFrame:
    """Add dominant Fourier cycles as features using rolling windows"""
    df = df.copy()
    for i in range(n_harmonics):
        df[f'fourier_amp_{i}'] = np.nan
        df[f'fourier_freq_{i}'] = np.nan
    
    for i in range(window, len(df)):
        window_data = df[column].iloc[i-window:i].values
        if len(window_data) < window:
            continue
        yf = fft(window_data)
        freq = fftfreq(window)
        amplitudes = 2.0 / window * np.abs(yf[0:window//2])
        # Sort by amplitude (strongest cycles)
        idx = np.argsort(amplitudes)[::-1]
        for j in range(min(n_harmonics, len(idx))):
            df.iloc[i, df.columns.get_loc(f'fourier_amp_{j}')] = amplitudes[idx[j]]
            df.iloc[i, df.columns.get_loc(f'fourier_freq_{j}')] = freq[idx[j]]
    
    return df


def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Simple regime detection: Trending vs Ranging using ADX and volatility"""
    df = df.copy()
    
    # ADX (trend strength)
    from ta.trend import ADXIndicator
    adx = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['adx'] = adx.adx()
    
    # Volatility (ATR normalized)
    atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['atr'] = atr.average_true_range()
    df['atr_ratio'] = df['atr'] / df['close']
    
    # Regime label (simple threshold)
    df['regime'] = 0  # 0 = ranging, 1 = trending
    df.loc[df['adx'] > 25, 'regime'] = 1
    
    return df


def engineer_features(input_path: str = "data/raw/BTC_USDT_5m.parquet",
                      output_path: str = "data/processed/BTC_USDT_5m_enhanced.parquet") -> pd.DataFrame:
    """Main feature engineering pipeline"""
    logger.info(f"Starting feature engineering on {input_path}")
    
    df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(df):,} raw bars")

    # 1. Basic TA features (momentum, volatility, volume, trend)
    df = add_all_ta_features(
        df, open="open", high="high", low="low", close="close", volume="volume",
        fillna=True
    )

    # 2. Extra volume features
    df['volume_sma_20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma_20']
    df['volume_zscore'] = (df['volume'] - df['volume'].rolling(50).mean()) / df['volume'].rolling(50).std()

    # 3. Fourier Cycle Detection
    df = add_fourier_features(df, column='close', n_harmonics=5, window=256)

    # 4. Regime Detection
    df = add_regime_features(df)

    # 5. Clean up (drop rows with NaN from rolling calculations)
    df = df.dropna()

    logger.info(f"Feature engineering complete. Final shape: {df.shape}")
    logger.info(f"Columns added: {len(df.columns) - 5} new features")

    # Save enhanced dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, compression='snappy')
    logger.info(f"Enhanced data saved to {output_path}")

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    engineer_features()