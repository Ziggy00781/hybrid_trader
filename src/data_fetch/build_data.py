import pandas as pd
import numpy as np
import ccxt
import logging
from pathlib import Path
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def build_dashboard_data(days: int = 60, output_path: str = "data/raw/binance_btcusdt_5m.parquet"):
    """Build clean 5m BTC/USDT data for the dashboard"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Attempting to fetch {days} days of BTC/USDT 5m data...")

    # Try real data from Binance (most reliable for public data)
    try:
        exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'adjustForTimeDifference': True},
            'timeout': 30000,
        })

        since = exchange.milliseconds() - (days * 24 * 60 * 60 * 1000)
        ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe='5m', since=since, limit=1000)

        # Paginate if needed (ccxt doesn't always return full history in one call)
        all_ohlcv = ohlcv
        while len(ohlcv) == 1000:
            since = ohlcv[-1][0] + 1
            ohlcv = exchange.fetch_ohlcv('BTC/USDT', '5m', since=since, limit=1000)
            if ohlcv:
                all_ohlcv.extend(ohlcv)
            else:
                break

        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df.sort_index()

        logger.info(f"✅ Successfully fetched {len(df):,} real candles from Binance")

    except Exception as e:
        logger.warning(f"Real fetch failed: {e}. Generating high-quality demo data instead.")
        
        # High-quality synthetic data (much better than the minimal 100-sample demo)
        n = days * 288  # 288 = 5m candles per day
        start_date = datetime.now() - timedelta(days=days)
        dates = pd.date_range(start=start_date, periods=n, freq='5min')

        base_price = 92000.0
        np.random.seed(42)
        returns = np.random.normal(0, 0.0008, n)          # realistic 5m volatility
        prices = np.cumprod(1 + returns) * base_price

        df = pd.DataFrame({
            'open': prices,
            'high': prices * (1 + np.abs(np.random.normal(0, 0.0006, n))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.0006, n))),
            'close': prices,
            'volume': np.random.lognormal(9, 1.1, n)      # realistic volume distribution
        }, index=dates)

        logger.info(f"✅ Generated realistic demo data with {len(df):,} candles")

    # Save exactly what the dashboard expects
    df = df[['open', 'high', 'low', 'close', 'volume']]
    df.to_parquet(output_path)
    logger.info(f"✅ Dashboard data saved to: {output_path} ({len(df):,} rows)")

    # Also save a copy in enhanced folder for consistency
    enhanced_path = Path("data/raw/enhanced/BTC_USDT_enhanced.parquet")
    enhanced_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(enhanced_path)

    return df


if __name__ == "__main__":
    build_dashboard_data(days=90)   # Change to 30/60/180 as needed