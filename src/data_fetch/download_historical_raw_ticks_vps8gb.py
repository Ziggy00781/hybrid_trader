import requests
import zipfile
import tempfile
import os
import time
import logging
import gc
from datetime import datetime, timedelta
import polars as pl
from tqdm import tqdm

# ====================== CONFIG FOR 8GB VPS ======================
DATA_DIR = "data/ticks"

# Top 10 cryptos (spot USDT pairs) - your original list
TARGET_SYMBOLS = ["btcusdt", "usdcusdt", "ethusdt", "solusdt", "bnbusdt",
                  "xrpusdt", "dogeusdt", "tonusdt", "adausdt", "shibusdt"]

GLOBAL_START_DATE = "2024-01-01"   # Change to "2017-08-01" if you want full history (more 404s but safe)
FORCE_FULL = {"btcusdt", "usdcusdt"}

SLEEP_BETWEEN_DAYS = 1.8
MAX_RETRIES = 5
ROW_GROUP_SIZE = 400_000
CHUNK_SIZE_MB = 8

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

os.makedirs(DATA_DIR, exist_ok=True)

def should_force_full(symbol: str) -> bool:
    return symbol.lower() in FORCE_FULL

def get_existing_dates(symbol_upper: str) -> set:
    """Return set of dates that already have a Parquet file"""
    dir_path = os.path.join(DATA_DIR, symbol_upper)
    if not os.path.exists(dir_path):
        return set()
    files = [f for f in os.listdir(dir_path) if f.endswith(".parquet")]
    dates = set()
    for f in files:
        try:
            date_part = f.split('_')[1].replace('.parquet', '')
            dates.add(date_part)
        except:
            pass
    return dates

def download_one_day(symbol: str, date_str: str) -> bool:
    symbol_upper = symbol.upper()
    dir_path = os.path.join(DATA_DIR, symbol_upper)
    os.makedirs(dir_path, exist_ok=True)
    parquet_path = os.path.join(dir_path, f"{symbol_upper}_{date_str}.parquet")

    if os.path.exists(parquet_path) and not should_force_full(symbol):
        return True  # already good

    url = f"https://data.binance.vision/data/spot/daily/trades/{symbol_upper}/{symbol_upper}-trades-{date_str}.zip"

    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"{symbol_upper} {date_str} (attempt {attempt+1}/{MAX_RETRIES})")
            response = requests.get(url, stream=True, timeout=180)
            response.raise_for_status()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE_MB * 1024 * 1024):
                    tmp.write(chunk)
                zip_path = tmp.name

            # Stream parse → Parquet (richer schema for AI)
            with zipfile.ZipFile(zip_path) as z:
                csv_name = z.namelist()[0]
                with z.open(csv_name) as f:
                    lf = pl.scan_csv(
                        f,
                        has_header=False,
                        new_columns=['trade_id', 'price', 'quantity', 'quote_quantity',
                                     'timestamp_ms', 'is_buyer_maker', 'is_best_match'],
                        schema_overrides={
                            "trade_id": pl.Int64,
                            "price": pl.Float32,
                            "quantity": pl.Float32,
                            "quote_quantity": pl.Float32,
                            "timestamp_ms": pl.Int64,
                            "is_buyer_maker": pl.Boolean,
                            "is_best_match": pl.Boolean
                        }
                    ).with_columns([
                        pl.col("timestamp_ms").cast(pl.Datetime("ms")).alias("timestamp")
                    ]).select([
                        "trade_id", "timestamp", "price", "quantity", "quote_quantity", "is_buyer_maker"
                    ])

                    lf.sink_parquet(
                        parquet_path,
                        compression="snappy",
                        row_group_size=ROW_GROUP_SIZE
                    )

            os.unlink(zip_path)
            gc.collect()
            logger.info(f"✅ Saved {symbol_upper} {date_str} ({os.path.getsize(parquet_path)/1e6:.1f} MB)")
            return True

        except requests.exceptions.HTTPError as e:
            if response.status_code == 404:
                logger.info(f"ℹ️  No data yet for {symbol_upper} {date_str}")
                return True
        except Exception as e:
            logger.warning(f"Attempt {attempt+1} failed: {e}")

        if attempt < MAX_RETRIES - 1:
            wait = 4 ** attempt
            time.sleep(wait)

    logger.error(f"❌ Failed {symbol_upper} {date_str} after {MAX_RETRIES} attempts")
    return False

def main():
    logger.info("=" * 80)
    logger.info("Hybrid Trader - Tick Data Downloader (8GB VPS optimized)")
    logger.info("True tick-level → daily Parquet | Continuous + AI-ready")
    logger.info("=" * 80)

    end_input = input(f"End date (default today {datetime.now().strftime('%Y-%m-%d')}): ").strip()
    end_date = end_input or datetime.now().strftime("%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    start_dt = datetime.strptime(GLOBAL_START_DATE, "%Y-%m-%d")

    for symbol in TARGET_SYMBOLS:
        symbol_upper = symbol.upper()
        existing_dates = get_existing_dates(symbol_upper)

        if should_force_full(symbol):
            logger.info(f"🔄 FORCE FULL REDOWNLOAD for {symbol_upper}")
            existing_dates.clear()  # will re-download everything

        current = start_dt
        total_days = (end_dt - current).days + 1
        pbar = tqdm(total=total_days, desc=f"{symbol_upper} ticks")

        downloaded_count = 0
        while current <= end_dt:
            date_str = current.strftime("%Y-%m-%d")
            if date_str not in existing_dates:
                success = download_one_day(symbol, date_str)
                if success and date_str in existing_dates:  # just in case
                    downloaded_count += 1
            else:
                pbar.set_postfix({"status": "already exists"})
            pbar.update(1)
            current += timedelta(days=1)
            time.sleep(SLEEP_BETWEEN_DAYS)

        pbar.close()
        logger.info(f"Finished {symbol_upper} — {downloaded_count} new day(s) added")
        gc.collect()

    logger.info("\n✅ Download session complete! Data is continuous and ready for AI training.")
    logger.info("Next step: run the consolidate script (below) if you want one big file per symbol.")

if __name__ == "__main__":
    main()