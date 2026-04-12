# src/recorder/tick_recorder.py
"""
24/7 Binance BTC/USDT Tick Recorder with Gap Detection
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
import pandas as pd
import websockets
import time
from src.data_fetch.unified_fetcher import UnifiedDataFetcher

logger = logging.getLogger(__name__)

SYMBOL = "btcusdt"
SAVE_DIR = "data/ticks/BTC_USDT"
os.makedirs(SAVE_DIR, exist_ok=True)

# Global state
current_date = None
buffer = []
tick_count_today = 0
last_log_time = time.time()
last_trade_time = None   # To detect gaps


async def save_buffer():
    global buffer, current_date, tick_count_today
    if not buffer:
        return

    df = pd.DataFrame(buffer, columns=["timestamp", "price", "quantity", "is_buyer_maker"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)

    filename = f"{SAVE_DIR}/BTC_USDT_{current_date}.parquet"
    df.to_parquet(filename, compression="snappy")
    
    logger.info(f"💾 Saved {len(df):,} ticks for {current_date} | Total today: {tick_count_today:,}")
    buffer = []


async def backfill_gap(start_time, end_time):
    """Try to backfill small gaps using historical data"""
    logger.info(f"Attempting to backfill gap from {start_time} to {end_time}")
    try:
        fetcher = UnifiedDataFetcher()
        # Use 1m candles to backfill (good compromise)
        df = fetcher.fetch_data(
            symbol="BTC/USDT",
            timeframe="1m",
            start_date=start_time,
            end_date=end_time,
            source="ccxt"
        )
        if not df.empty:
            logger.info(f"Backfilled {len(df)} 1m candles for the gap")
            # Note: Converting 1m candles to ticks is approximate - we can improve later
    except Exception as e:
        logger.warning(f"Backfill failed: {e}")


async def tick_recorder():
    global current_date, buffer, tick_count_today, last_log_time, last_trade_time

    url = f"wss://stream.binance.com:9443/ws/{SYMBOL}@aggTrade"

    logger.info(f"🚀 Starting 24/7 tick recorder for {SYMBOL.upper()}...")

    while True:
        try:
            async with websockets.connect(url) as ws:
                logger.info("✅ Connected to Binance WebSocket")

                while True:
                    message = await ws.recv()
                    data = json.loads(message)

                    trade_time = data["T"]
                    trade = {
                        "timestamp": trade_time,
                        "price": float(data["p"]),
                        "quantity": float(data["q"]),
                        "is_buyer_maker": data["m"],
                    }

                    # Detect gap
                    if last_trade_time is not None:
                        gap_seconds = (trade_time - last_trade_time) / 1000
                        if gap_seconds > 10:   # more than 10 seconds gap
                            logger.warning(f"Gap detected: {gap_seconds:.1f} seconds")
                            # Optional: try to backfill small gaps
                            if gap_seconds < 300:  # less than 5 minutes
                                await backfill_gap(
                                    datetime.utcfromtimestamp(last_trade_time/1000),
                                    datetime.utcfromtimestamp(trade_time/1000)
                                )

                    last_trade_time = trade_time
                    buffer.append(trade)
                    tick_count_today += 1

                    # Progress log every 10 seconds
                    if time.time() - last_log_time > 10:
                        logger.info(f"📊 Today: {tick_count_today:,} ticks | "
                                  f"Buffer: {len(buffer):,} | Last price: ${trade['price']:,.2f}")
                        last_log_time = time.time()

                    # Daily rotation
                    trade_date = datetime.utcfromtimestamp(trade_time / 1000).date()
                    if current_date is None:
                        current_date = trade_date
                    elif trade_date != current_date:
                        await save_buffer()
                        current_date = trade_date
                        tick_count_today = 0
                        last_trade_time = None

        except Exception as e:
            logger.error(f"Connection lost: {e}. Reconnecting in 5s...")
            await asyncio.sleep(5)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    try:
        asyncio.run(tick_recorder())
    except KeyboardInterrupt:
        logger.info("⛔ Recorder stopped by user. Saving final buffer...")
        asyncio.run(save_buffer())
        logger.info("✅ Recorder shut down cleanly.")


if __name__ == "__main__":
    main()