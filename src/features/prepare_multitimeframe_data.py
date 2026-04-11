# src/features/prepare_multitimeframe_data.py
"""
Clean Multi-Timeframe Data Preparation
Fetches deep 5m history, then resamples to create 1m, 15m, 30m, 1h features.
This guarantees good data volume and perfect alignment.
"""

import pandas as pd
import numpy as np
from src.data_fetch.unified_fetcher import UnifiedDataFetcher
import os
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def prepare_multitimeframe_data(
    symbol: str = "BTC/USDT",
    years_back: int = 2,
    output_path: str = "data/processed/BTC_USDT_multitimeframe.parquet"
):
    fetcher = UnifiedDataFetcher()
    logger.info(f"Fetching deep 5m history for {symbol}...")

    # Fetch deep 5m data with proper pagination (using your unified fetcher)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years_back)

    df_5m = fetcher.fetch_data(
        symbol=symbol,
        timeframe="5m",
        start_date=start_date,
        source="ccxt",          # Force CCXT for deeper history
        ccxt_exchange="binance"
    )

    logger.info(f"Fetched {len(df_5m):,} bars on 5m from {df_5m.index[0]} to {df_5m.index[-1]}")

    # Create multi-timeframe features by resampling the 5m data
    df = df_5m.copy()

    # 1m features (resample 5m down and ffill - approximate)
    df_1m = df.resample("1min").ffill()
    df = df.join(df_1m.add_suffix("_1m"), how="left")

    # 15m, 30m, 1h features
    for tf, minutes in [("15m", 15), ("30m", 30), ("1h", 60)]:
        df_tf = df.resample(f"{minutes}min").agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        df_tf = df_tf.resample("5min").ffill()
        df = df.join(df_tf.add_suffix(f"_{tf}"), how="left")

    df = df.ffill().dropna()

    # Add target: next 5m log return
    df['log_close'] = np.log(df['close'])
    df['target_log_return'] = df['log_close'].shift(-1) - df['log_close']
    df = df.dropna()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, compression="snappy")

    logger.info(f"✅ Multi-timeframe dataset created: {df.shape}")
    logger.info(f"Time range: {df.index[0]} → {df.index[-1]}")
    logger.info(f"Total features: {df.shape[1]}")
    logger.info(f"Saved to {output_path}")

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    prepare_multitimeframe_data()