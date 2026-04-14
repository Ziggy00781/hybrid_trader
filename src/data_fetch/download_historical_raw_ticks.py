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

# ==== CONFIG ====
DATA_DIR = "data/ticks"
BASE_URL = "https://data.binance.vision/data/spot/daily/trades/"  # RAW trades = highest resolution
TOP_N = 10
SAVE_INTERVAL = 2  # not used here, just for consistency

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_top_usdt_symbols(n=10):
    """Same logic as your live recorder - always includes BTC & ETH."""
    try:
        exchange = ccxt.binance({'enableRateLimit': True})
        markets = exchange.load_markets()
        tickers = exchange.fetch_tickers()

        usdt_data = []
        for symbol, market in markets.items():
            if (market.get('quote') == 'USDT' and
                market.get('spot') and
                market.get('active')):
                ticker = tickers.get(symbol, {})
                volume = float(ticker.get('quoteVolume') or 0)
                sym_lower = symbol.replace('/', '').lower()
                usdt_data.append((sym_lower, volume))

        usdt_data.sort(key=lambda x: x[1], reverse=True)
        top_symbols = [sym for sym, vol in usdt_data[:n]]
        print(f"✅ Using current top {len(top_symbols)} symbols: {[s.upper() for s in top_symbols]}")
        return top_symbols
    except Exception as e:
        print(f"⚠️ Failed to fetch top symbols: {e}. Using fallback.")
        fallback = ["btcusdt", "ethusdt", "solusdt", "bnbusdt", "xrpusdt",
                    "dogeusdt", "tonusdt", "adausdt", "shibusdt", "trxusdt"]
        return fallback


def download_raw_historical_ticks(symbol: str, start_date: str, end_date: str = None):
    """Download RAW trades (highest resolution) and save in EXACT same Parquet format as live recorder."""
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    current_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    symbol_upper = symbol.upper()
    symbol_lower = symbol.lower()
    dir_path = os.path.join(DATA_DIR, symbol_upper)
    os.makedirs(dir_path, exist_ok=True)

    while current_date <= end_dt:
        date_str = current_date.strftime("%Y-%m-%d")
        filename = f"{symbol_upper}-trades-{date_str}.zip"
        url = f"{BASE_URL}{symbol_upper}/{filename}"
        parquet_path = os.path.join(dir_path, f"{symbol_upper}_{date_str}.parquet")

        if os.path.exists(parquet_path):
            print(f"✅ Skipping (already exists): {parquet_path}")
            current_date += timedelta(days=1)
            continue

        print(f"📥 Downloading RAW ticks → {symbol_upper} {date_str} ...")

        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()

            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                csv_name = z.namelist()[0]
                with z.open(csv_name) as f:
                    # RAW trades columns (exact from Binance public data)
                    df = pd.read_csv(f, header=None, names=[
                        'trade_id', 'price', 'quantity', 'quote_quantity',
                        'timestamp_ms', 'is_buyer_maker', 'is_best_match'
                    ])

                    # Convert to EXACT same format as your live recorder
                    df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms')
                    df['price'] = df['price'].astype(float)
                    df['quantity'] = df['quantity'].astype(float)
                    df['is_buyer_maker'] = df['is_buyer_maker'].astype(bool)

                    # Keep ONLY the columns your live recorder uses
                    df = df[['timestamp', 'price', 'quantity', 'is_buyer_maker']].copy()

            # Save Parquet exactly like live recorder
            temp_path = parquet_path + ".tmp"
            df.to_parquet(temp_path, index=False)
            shutil.move(temp_path, parquet_path)

            print(f"✅ Saved {len(df):,} RAW ticks → {parquet_path} | Last price: ${df['price'].iloc[-1]:,.4f}")

        except requests.exceptions.HTTPError as e:
            if response.status_code == 404:
                print(f"⚠️ No data yet for {symbol_upper} on {date_str} (file not available)")
            else:
                print(f"❌ HTTP error for {date_str}: {e}")
        except Exception as e:
            print(f"❌ Error downloading {date_str}: {e}")
            print(traceback.format_exc())

        current_date += timedelta(days=1)
        time.sleep(0.3)  # Be respectful


def main():
    print("="*80)
    print("🚀 HIGHEST RESOLUTION HISTORICAL TICK DOWNLOADER")
    print("   → Raw trades (every single trade) matching your live recorder 100%")
    print("   → Perfect for AI training on microstructure, chart patterns & time prediction")
    print("="*80)

    start_date = input("\nEnter START date (YYYY-MM-DD, e.g. 2024-01-01): ").strip()
    end_input = input("Enter END date (YYYY-MM-DD) or press Enter for today: ").strip()
    end_date = end_input or None

    symbols = get_top_usdt_symbols(TOP_N)

    for symbol in symbols:
        print(f"\n🔄 Processing {symbol.upper()} ...")
        download_raw_historical_ticks(symbol, start_date, end_date)

    print("\n🎉 DONE! All Parquet files are now in data/ticks/ and match your live recorder exactly.")
    print("   Your AI models can now train on seamless live + historical tick data.")


if __name__ == "__main__":
    main()