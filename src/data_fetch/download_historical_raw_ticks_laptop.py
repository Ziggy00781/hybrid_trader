import requests
import zipfile
import tempfile
import os
import time
import logging
import gc
from datetime import datetime, timedelta
import polars as pl
import ccxt
from tqdm import tqdm

# ====================== CONFIG ======================
DATA_DIR = "data/ticks"

# Symbols you want (BTC + USDC + others)
TARGET_SYMBOLS = ["btcusdt", "usdcusdt", "ethusdt", "solusdt", "bnbusdt", 
                  "xrpusdt", "dogeusdt", "tonusdt", "adausdt", "shibusdt"]

GLOBAL_START_DATE = "2024-01-01"
FORCE_FULL_DOWNLOAD = True          # ← Set to True for BTC redownload
FORCE_SYMBOLS = {"btcusdt": True, "usdcusdt": True}  # Force redownload for these

SLEEP_BETWEEN_DAYS = 1.5            # Faster on laptop
MAX_RETRIES = 3
ROW_GROUP_SIZE = 300_000            # Balanced for 13GB RAM

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

os.makedirs(DATA_DIR, exist_ok=True)

def should_force_full(symbol: str) -> bool:
    return FORCE_FULL_DOWNLOAD or FORCE_SYMBOLS.get(symbol.lower(), False)

def get_latest_existing_date(symbol_upper: str) -> datetime:
    if should_force_full(symbol_upper.lower()):
        return datetime.strptime(GLOBAL_START_DATE, "%Y-%m-%d")

    dir_path = os.path.join(DATA_DIR, symbol_upper)
    if not os.path.exists(dir_path):
        return datetime.strptime(GLOBAL_START_DATE, "%Y-%m-%d")

    files = [f for f in os.listdir(dir_path) if f.endswith(".parquet")]
    if not files:
        return datetime.strptime(GLOBAL_START_DATE, "%Y-%m-%d")

    dates = []
    for f in files:
        try:
            date_part = f.split('_')[1].replace('.parquet', '')
            dates.append(datetime.strptime(date_part, "%Y-%m-%d"))
        except:
            pass
    return max(dates) if dates else datetime.strptime(GLOBAL_START_DATE, "%Y-%m-%d")


def download_one_day(symbol: str, date_str: str) -> bool:
    symbol_upper = symbol.upper()
    dir_path = os.path.join(DATA_DIR, symbol_upper)
    os.makedirs(dir_path, exist_ok=True)
    parquet_path = os.path.join(dir_path, f"{symbol_upper}_{date_str}.parquet")

    if os.path.exists(parquet_path) and not should_force_full(symbol):
        logger.info(f"⏭️ Already exists: {symbol_upper} {date_str}")
        return True

    url = f"https://data.binance.vision/data/spot/daily/trades/{symbol_upper}/{symbol_upper}-trades-{date_str}.zip"

    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"📥 Downloading {symbol_upper} {date_str} (attempt {attempt+1})")

            response = requests.get(url, stream=True, timeout=120)
            response.raise_for_status()

            # Use temp file on disk
            with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
                for chunk in response.iter_content(chunk_size=8*1024*1024):
                    tmp.write(chunk)
                zip_path = tmp.name

            # Low-memory Polars streaming
            with zipfile.ZipFile(zip_path) as z:
                csv_name = z.namelist()[0]
                with z.open(csv_name) as f:
                    lf = pl.scan_csv(
                        f,
                        has_header=False,
                        new_columns=['trade_id', 'price', 'quantity', 'quote_quantity',
                                     'timestamp_ms', 'is_buyer_maker', 'is_best_match'],
                        schema_overrides={
                            "price": pl.Float32,
                            "quantity": pl.Float32,
                            "timestamp_ms": pl.Int64,
                            "is_buyer_maker": pl.Boolean
                        }
                    )

                    lf = lf.with_columns([
                        pl.col("timestamp_ms").cast(pl.Datetime("ms")).alias("timestamp")
                    ]).select(["timestamp", "price", "quantity", "is_buyer_maker"])

                    lf.sink_parquet(
                        parquet_path,
                        compression="snappy",
                        row_group_size=ROW_GROUP_SIZE
                    )

            os.unlink(zip_path)
            gc.collect()

            logger.info(f"✅ Saved {symbol_upper} {date_str}")
            return True

        except requests.exceptions.HTTPError as e:
            if getattr(response, "status_code", 0) == 404:
                logger.info(f"⚠️ No file yet for {symbol_upper} {date_str}")
                return True
        except Exception as e:
            logger.warning(f"Attempt {attempt+1} failed for {symbol_upper} {date_str}: {e}")

        if attempt < MAX_RETRIES - 1:
            time.sleep(2 ** attempt)

    logger.error(f"❌ Failed {symbol_upper} {date_str} after {MAX_RETRIES} attempts")
    return False


def main():
    logger.info("🚀 Laptop Downloader - Redownloading BTC + USDC from 2024-01-01")
    logger.info(f"Target symbols: {TARGET_SYMBOLS}")
    
    end_input = input(f"End date (default today {datetime.now().strftime('%Y-%m-%d')}): ").strip()
    end_date = end_input or datetime.now().strftime("%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    for symbol in TARGET_SYMBOLS:
        symbol_upper = symbol.upper()
        start_dt = get_latest_existing_date(symbol_upper) + timedelta(days=1)

        if start_dt > end_dt:
            logger.info(f"✅ {symbol_upper} is already up to date")
            continue

        logger.info(f"\n🔄 Processing {symbol_upper} from {start_dt.strftime('%Y-%m-%d')} to {end_date}")

        current = start_dt
        total_days = (end_dt - current).days + 1
        pbar = tqdm(total=total_days, desc=symbol_upper)

        while current <= end_dt:
            date_str = current.strftime("%Y-%m-%d")
            download_one_day(symbol, date_str)
            pbar.update(1)
            current += timedelta(days=1)
            time.sleep(SLEEP_BETWEEN_DAYS)

        pbar.close()
        gc.collect()

    logger.info("\n🎉 Download session completed!")
    logger.info("   BTC has been redownloaded. USDCUSDT has been added.")
    logger.info("   Next: You can now start building training datasets.")

if __name__ == "__main__":
    main()