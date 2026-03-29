import ccxt
import pandas as pd
from pathlib import Path
import time

def fetch_binance_global_ohlcv(symbol="BTC/USDT", timeframe="5m", limit=1000, since=None, stop_date="2017-08-01"):
    exchange = ccxt.binance()
    all_data = []
    batch = 0

    # Convert stop_date to ms timestamp
    stop_ts = pd.to_datetime(stop_date, utc=True).value // 10**6

    while True:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not ohlcv:
            break

        all_data.extend(ohlcv)
        batch += 1

        first_ts = pd.to_datetime(ohlcv[0][0], unit="ms", utc=True)
        last_ts  = pd.to_datetime(ohlcv[-1][0], unit="ms", utc=True)
        print(f"[Batch {batch}] Rows: {len(all_data)} | Range: {first_ts} → {last_ts}")

        # Move 'since' backwards using the oldest candle in this batch
        since = ohlcv[0][0] - limit * 5 * 60 * 1000

        # Stop if we've reached or passed the cutoff
        if since <= stop_ts:
            break

        time.sleep(exchange.rateLimit / 1000)

    df = pd.DataFrame(all_data, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    return df

if __name__ == "__main__":
    df = fetch_binance_global_ohlcv()
    out_path = Path("data/raw/binance/binance_global_btcusdt_5m.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    print(f"✅ Saved {len(df)} rows to {out_path}")