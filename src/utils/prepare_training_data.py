# src/utils/prepare_training_data.py
"""
Prepare training data for Hybrid Trader models (PatchTST, LightGBM, etc.)
Supports BTC/USDT (Binance via CCXT) and other assets via Alpaca.
"""

import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

from src.data_fetch.unified_fetcher import UnifiedDataFetcher


def prepare_training_data(
    symbol: str = "BTC/USDT",
    timeframe: str = "5m",
    years_back: int = 2,
    source: str = "auto",
    output_dir: str = "data/raw",
    filename: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch OHLCV data and save as Parquet for model training.
    
    Args:
        symbol: e.g. "BTC/USDT", "BTC/USD", "AAPL"
        timeframe: "5m", "15m", "1h", "1d"
        years_back: How many years of history to fetch
        source: "auto", "ccxt", or "alpaca"
        output_dir: Directory to save the parquet file
        filename: Custom filename (auto-generated if None)
    """
    print(f"🚀 Preparing training data for {symbol} ({timeframe})...")

    fetcher = UnifiedDataFetcher()

    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years_back + 30)  # extra buffer

    df = fetcher.fetch_data(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        source=source,
        limit=10000,          # Large limit + loop in fetcher handles pagination
    )

    # Basic cleaning and validation
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]   # remove any duplicate timestamps
    df = df.dropna()                             # drop any bad rows

    print(f"✅ Fetched {len(df):,} bars")
    print(f"   From: {df.index[0]}")
    print(f"   To:   {df.index[-1]}")
    print(f"   Columns: {list(df.columns)}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate filename if not provided
    if filename is None:
        clean_symbol = symbol.replace("/", "_").replace("-", "_")
        filename = f"{clean_symbol}_{timeframe}.parquet"

    output_path = os.path.join(output_dir, filename)

    df.to_parquet(output_path, compression="snappy")
    print(f"💾 Saved to: {output_path}")
    print(f"   File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB\n")

    return df


if __name__ == "__main__":
    # Default usage - prepare BTC 5m data (your main pair)
    df = prepare_training_data(
        symbol="BTC/USDT",
        timeframe="5m",
        years_back=2,           # Change to 1 or 3 as needed
        source="auto"
    )

    # Uncomment for other examples:
    # prepare_training_data(symbol="BTC/USD", timeframe="5m", source="alpaca")
    # prepare_training_data(symbol="ETH/USDT", timeframe="15m", years_back=1)