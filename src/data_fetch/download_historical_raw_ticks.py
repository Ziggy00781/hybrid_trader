import requests
import zipfile
import io
import os
import pandas as pd
import shutil
from datetime import datetime, timedelta
import time
import logging
from tqdm import tqdm
import ccxt
import traceback

DATA_DIR = "data/ticks"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_top_usdt_symbols(n=10):
    try:
        exchange = ccxt.binance({'enableRateLimit': True})
        markets = exchange.load_markets()
        tickers = exchange.fetch_tickers()
        usdt_data = []
        for symbol, market in markets.items():
            if market.get('quote') == 'USDT' and market.get('spot') and market.get('active'):
                volume = float(tickers.get(symbol, {}).get('quoteVolume') or 0)
                usdt_data.append((symbol.replace('/', '').lower(), volume))
        usdt_data.sort(key=lambda x: x[1], reverse=True)
        top = [sym for sym, _ in usdt_data[:n]]
        print(f"✅ Top {n} symbols: {[s.upper() for s in top]}")
        return top
    except Exception:
        print("⚠️ Using fallback top 10")
        return ["btcusdt", "ethusdt", "solusdt", "bnbusdt", "xrpusdt",
                "dogeusdt", "tonusdt", "adausdt", "shibusdt", "trxusdt"]

def download_raw_historical_ticks(symbol: str, start_date: str, end_date: str = None, max_retries=3):
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    current = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    symbol_upper = symbol.upper()
    dir_path = os.path.join(DATA_DIR, symbol_upper)
    os.makedirs(dir_path, exist_ok=True)

    total_days = (end_dt - current).days + 1
    pbar = tqdm(total=total_days, desc=f"{symbol_upper}")

    while current <= end_dt:
        date_str = current.strftime("%Y-%m-%d")
        filename = f"{symbol_upper}-trades-{date_str}.zip"
        url = f"https://data.binance.vision/data/spot/daily/trades/{symbol_upper}/{filename}"
        parquet_path = os.path.join(dir_path, f"{symbol_upper}_{date_str}.parquet")

        if os.path.exists(parquet_path):
            pbar.update(1)
            current += timedelta(days=1)
            continue

        for attempt in range(max_retries):
            try:
                response = requests.get(url, stream=True, timeout=60)
                response.raise_for_status()

                with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                    csv_name = z.namelist()[0]
                    with z.open(csv_name) as f:
                        df = pd.read_csv(f, header=None, names=[
                            'trade_id', 'price', 'quantity', 'quote_quantity',
                            'timestamp_ms', 'is_buyer_maker', 'is_best_match'
                        ])
                        df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms')
                        df = df[['timestamp', 'price', 'quantity', 'is_buyer_maker']].copy()

                temp_path = parquet_path + ".tmp"
                df.to_parquet(temp_path, index=False)
                shutil.move(temp_path, parquet_path)

                pbar.update(1)
                pbar.set_postfix({"ticks": f"{len(df):,}", "last": f"${df['price'].iloc[-1]:.2f}"})
                break

            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"\n❌ Failed {symbol_upper} {date_str} after {max_retries} tries: {e}")
                time.sleep(2 ** attempt)  # exponential backoff

        time.sleep(1.2)   # Be nice to the server
        current += timedelta(days=1)

    pbar.close()

def main():
    print("🚀 Raw Tick Downloader (2024 → Today) - Highest Resolution")
    start_date = "2024-01-01"
    end_date = input(f"End date (default today {datetime.now().strftime('%Y-%m-%d')}): ").strip() or None

    symbols = get_top_usdt_symbols(10)

    for symbol in symbols:
        print(f"\n=== Starting {symbol.upper()} ===")
        download_raw_historical_ticks(symbol, start_date, end_date)

    print("\n🎉 Download session finished! Check data/ticks/ folder.")
    print("   You can run the script again later to catch any missing days.")

if __name__ == "__main__":
    main()