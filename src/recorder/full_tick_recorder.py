import asyncio
import json
import random
import traceback
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
import websockets
import pandas as pd


BINANCE_WS = "wss://stream.binance.com:9443/stream"
ROOT_DIR = Path("data/ticks")
ROOT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 40
FLUSH_INTERVAL = 2          # seconds
BUFFER_LIMIT = 1000         # flush if buffer reaches this many ticks


# ---------------------------------------------------------
# SYMBOL DISCOVERY
# ---------------------------------------------------------
async def get_usdt_spot_symbols():
    url = "https://api.binance.com/api/v3/exchangeInfo"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as r:
            data = await r.json()

    symbols = []
    for s in data["symbols"]:
        if (
            s["status"] == "TRADING"
            and s["isSpotTradingAllowed"]
            and s["quoteAsset"] == "USDT"
        ):
            symbols.append(s["symbol"].lower())

    return symbols


# ---------------------------------------------------------
# DAILY PARQUET WRITER
# ---------------------------------------------------------
class DailyParquetWriter:
    def __init__(self, symbol):
        self.symbol = symbol
        self.buffer = []
        self.current_date = self._today()

        # ensure directory exists
        self.dir = ROOT_DIR / symbol
        self.dir.mkdir(parents=True, exist_ok=True)

    def _today(self):
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _path(self):
        return self.dir / f"{self.current_date}.parquet"

    def add(self, tick):
        self.buffer.append(tick)

    def flush(self):
        if not self.buffer:
            return

        # rollover at midnight UTC
        today = self._today()
        if today != self.current_date:
            self.current_date = today

        df = pd.DataFrame(self.buffer)
        df.to_parquet(self._path(), index=False, append=True)
        self.buffer.clear()


# ---------------------------------------------------------
# WEBSOCKET CONSUMER
# ---------------------------------------------------------
async def consume_batch(batch_id, symbols):
    streams = "/".join([f"{s}@trade" for s in symbols])
    url = f"{BINANCE_WS}?streams={streams}"

    writers = {s: DailyParquetWriter(s) for s in symbols}
    last_flush = datetime.now(timezone.utc)

    while True:
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                print(f"[Batch {batch_id}] Connected ({len(symbols)} symbols)")

                async for msg in ws:
                    try:
                        raw = json.loads(msg)
                        data = raw.get("data", raw)

                        if data.get("e") != "trade":
                            continue

                        symbol = data["s"].lower()
                        tick = {
                            "symbol": symbol,
                            "price": float(data["p"]),
                            "qty": float(data["q"]),
                            "side": "buy" if data["m"] is False else "sell",
                            "ts_event": data["E"],
                            "ts_trade": data["T"],
                        }

                        w = writers[symbol]
                        w.add(tick)

                        # periodic flush
                        now = datetime.now(timezone.utc)
                        if (
                            len(w.buffer) >= BUFFER_LIMIT
                            or (now - last_flush).total_seconds() >= FLUSH_INTERVAL
                        ):
                            for writer in writers.values():
                                writer.flush()
                            last_flush = now

                    except Exception:
                        print(f"[Batch {batch_id}] Message error:")
                        traceback.print_exc()

        except Exception:
            print(f"[Batch {batch_id}] Connection lost, retrying...")
            traceback.print_exc()
            await asyncio.sleep(2 + random.random() * 3)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
async def main():
    symbols = await get_usdt_spot_symbols()
    print(f"Discovered {len(symbols)} USDT spot symbols")

    batches = [
        symbols[i:i + BATCH_SIZE]
        for i in range(0, len(symbols), BATCH_SIZE)
    ]

    tasks = []
    for i, batch in enumerate(batches):
        tasks.append(asyncio.create_task(consume_batch(i, batch)))

    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())