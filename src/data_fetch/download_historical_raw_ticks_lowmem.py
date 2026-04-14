import requests
import zipfile
import io
import os
import time
import logging
from datetime import datetime, timedelta
import polars as pl
import tempfile
import gc
import ccxt
from tqdm import tqdm

# ====================== CONFIG ======================
DATA_DIR = "data/ticks"
MIN_24H_VOLUME_USD = 100_000_000
SLEEP_BETWEEN_DAYS = 2.0
MAX_RETRIES = 3

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

os.makedirs(DATA_DIR, exist_ok=True)

def get_top_usdt_symbols(n=10):
    try:
        exchange = ccxt.binance({'enableRateLimit': True})
        markets = exchange.load_markets()
        tickers = exchange.fetch_tickers()

        usdt_data = []
        for symbol, market in markets.items():
            if (market.get('quote') == 'USDT' and market.get('spot') and market.get('active')):
                volume = float(tickers.get(symbol, {}).get('quoteVolume') or 0)
                if volume >= MIN_24H_VOLUME_USD:
                    sym_lower = symbol.replace('/', '').lower()
                    usdt_data.append((sym_lower, volume))

        usdt_data.sort(key=lambda x: x[1], reverse=True)
        top_symbols = [sym for sym, vol in usdt_data[:n]]

        logger.info(f"✅ Selected top {len(top_symbols)} symbols:")
        for i, s in enumerate(top_symbols, 1):
            logger.info(f"   {i:2d}. {s.upper()}")
        return top_symbols
    except Exception:
        logger.warning("Using safe fallback list")
        return ["btcusdt", "ethusdt", "solusdt", "bnbusdt", "xrpusdt",
                "dogeusdt", "tonusdt", "adausdt", "shibusdt", "trxusdt"]


def get_latest_existing_date(symbol_upper: str) -> datetime:
    """Find the latest date we already have for this symbol to resume cleanly"""
    dir_path = os.path.join(DATA_DIR, symbol_upper)
    if not os.path.exists(dir_path):
        return datetime.strptime("2024-01-01", "%Y-%m-%d")

    files = [f for f in os.listdir(dir_path) if f.endswith(".parquet")]
    if not files:
        return datetime.strptime("2024-01-01", "%Y-%m-%d")

    dates = []
    for f in files:
        try:
            date_part = f.split('_')[1].replace('.parquet', '')
            dates.append(datetime.strptime(date_part, "%Y-%m-%d"))
        except:
            pass
    return max(dates) if dates else datetime.strptime("2024-01-01", "%Y-%m-%d")


def download_one_day(symbol: str, date_str: str) -> bool:
    symbol_upper = symbol.upper()
    dir_path = os.path.join(DATA_DIR, symbol_upper)
    os.makedirs(dir_path, exist_ok=True)
    parquet_path = os.path.join(dir_path, f"{symbol_upper}_{date_str}.parquet")

    if os.path.exists(parquet_path):
        logger.debug(f"⏭️ Already exists: {symbol_upper} {date_str}")
        return True

    filename = f"{symbol_upper}-trades-{date_str}.zip"
    url = f"https://data.binance.vision/data/spot/daily/trades/{symbol_upper}/{filename}"

    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"📥 {symbol_upper} {date_str} (attempt {attempt+1})")
            response = requests.get(url, stream=True, timeout=90)
            response.raise_for_status()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
                for chunk in response.iter_content(chunk_size=8*1024*1024):
                    tmp.write(chunk)
                tmp_path = tmp.name

            with zipfile.ZipFile(tmp_path) as z:
                csv_name = z.namelist()[0]
                with z.open(csv_name) as f:
                    df = pl.scan_csv(
                        f,
                        has_header=False,
                        new_columns=['trade_id','price','quantity','quote_quantity','timestamp_ms','is_buyer_maker','is_best_match'],
                        dtypes={'price': pl.Float32, 'quantity': pl.Float32, 'timestamp_ms': pl.Int64, 'is_buyer_maker': pl.Boolean}
                    )
                    df = df.with_columns([
                        pl.col('timestamp_ms').cast(pl.Datetime('ms')).alias('timestamp')
                    ]).select(['timestamp', 'price', 'quantity', 'is_buyer_maker'])

                    df.sink_parquet(parquet_path)

            os.unlink(tmp_path)
            gc.collect()
            logger.info(f"✅ Saved {symbol_upper} {date_str}")
            return True

        except requests.exceptions.HTTPError as e:
            if getattr(response, 'status_code', 0) == 404:
                logger.info(f"⚠️ No file yet for {symbol_upper} {date_str}")
                return True
        except Exception as e:
            logger.warning(f"Attempt {attempt+1} failed: {e}")

        if attempt < MAX_RETRIES - 1:
            time.sleep(2 ** attempt)

    logger.error(f"❌ Failed {symbol_upper} {date_str}")
    return False


def main():
    logger.info("🚀 Low-Memory Resume Downloader Started (1GB RAM Optimized)")
    end_input = input(f"End date (default today {datetime.now().strftime('%Y-%m-%d')}): ").strip()
    end_date = end_input or datetime.now().strftime("%Y-%m-%d")

    symbols = get_top_usdt_symbols(10)

    for symbol in symbols:
        symbol_upper = symbol.upper()
        start_dt = get_latest_existing_date(symbol_upper) + timedelta(days=1)
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        if start_dt > end_dt:
            logger.info(f"✅ {symbol_upper} is already up to date")
            continue

        logger.info(f"\n🔄 Resuming {symbol_upper} from {start_dt.strftime('%Y-%m-%d')} to {end_date}")
        current = start_dt
        total = (end_dt - current).days + 1
        pbar = tqdm(total=total, desc=symbol_upper)

        while current <= end_dt:
            date_str = current.strftime("%Y-%m-%d")
            download_one_day(symbol, date_str)
            pbar.update(1)
            current += timedelta(days=1)
            time.sleep(SLEEP_BETWEEN_DAYS)

        pbar.close()
        gc.collect()

    logger.info("\n🎉 Resume session completed! Run again anytime for missing days.")

if __name__ == "__main__":
    main()