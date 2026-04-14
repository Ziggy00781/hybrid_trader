import asyncio
import json
import random
import traceback
from datetime import datetime
from pathlib import Path

import aiohttp
import websockets
import pandas as pd


BINANCE_WS = "wss://stream.binance.com:9443/stream"
OUTPUT_DIR = Path("data/ticks")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 40
CHUNK_SIZE = 500          # flush every 500 ticks
SAVE_INTERVAL = 1         # or every 1 second


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


class TickWriter:
    def __init__(self, symbol):
        self.symbol = symbol
        self.buffer = []

    def add(self, tick):
        self.buffer.append(tick)

    def flush(self):
        if not self.buffer:
            return

        df = pd.DataFrame(self.buffer)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        out = OUTPUT_DIR / f"{self.symbol}_{ts}.parquet"

        df.to_parquet(out, index=False)
        self.buffer.clear()


async def consume_batch(batch_id, symbols):
    streams = "/".join([f"{s}@trade" for s in symbols])
    url = f"{BINANCE_WS}?streams={streams}"

    writers = {s: TickWriter(s) for s in symbols}
    last_save = datetime.utcnow()

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

                        now = datetime.utcnow()
                        if (
                            len(w.buffer) >= CHUNK_SIZE
                            or (now - last_save).total_seconds() >= SAVE_INTERVAL
                        ):
                            w.flush()
                            last_save = now

                    except Exception:
                        print(f"[Batch {batch_id}] Message error:")
                        traceback.print_exc()

        except Exception:
            print(f"[Batch {batch_id}] Connection lost, retrying...")
            traceback.print_exc()
            await asyncio.sleep(2 + random.random() * 3)


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