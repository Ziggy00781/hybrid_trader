import ccxt
import pandas as pd
import time
from pathlib import Path
from datetime import datetime

def fetch_full_5m_history(symbol: str = "BTC/USDT",
                          timeframe: str = "5m",
                          max_days: int = 1095,   # ~3 years
                          output_path: str = "data/raw/binance_btcusdt_5m.parquet"):
    """
    Fetch the longest possible 5m history using proper pagination.
    Uses Binance (most reliable for long public history).
    """
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'adjustForTimeDifference': True},
        'timeout': 30000,
    })

    print(f"🔄 Fetching up to {max_days} days of {timeframe} BTC/USDT from Binance...")

    since = exchange.milliseconds() - (max_days * 24 * 60 * 60 * 1000)
    all_ohlcv = []

    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1  # next batch starts after last candle
            print(f"   → Fetched {len(ohlcv)} candles | Total: {len(all_ohlcv):,}")
            time.sleep(0.5)  # be gentle with rate limits
        except Exception as e:
            print(f"⚠️  Error: {e}. Waiting 5s...")
            time.sleep(5)

    if not all_ohlcv:
        raise ValueError("No data fetched!")

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df.sort_index()

    # Save exactly where the rest of the project expects it
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path)

    print(f"\n✅ COMPLETE DATASET READY!")
    print(f"   Period     : {df.index[0]} → {df.index[-1]}")
    print(f"   Total candles: {len(df):,}")
    print(f"   Saved to     : {output_path}")
    return df


if __name__ == "__main__":
    fetch_full_5m_history()