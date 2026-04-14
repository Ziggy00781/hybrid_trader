import websocket
import json
import pandas as pd
import os
import time
import signal
import sys
import threading
from datetime import datetime
import shutil
import ccxt

# ========================= CONFIG =========================
DATA_DIR = "data/ticks"
SAVE_INTERVAL = 5
BATCH_SIZE = 180

os.makedirs(DATA_DIR, exist_ok=True)

buffers = {}
last_save = {}
file_locks = {}
stop_event = threading.Event()

last_price = {}        # Keep track of latest price per symbol for logging
last_log_time = time.time()

# =========================================================

def get_all_spot_symbols():
    try:
        exchange = ccxt.binance()
        markets = exchange.load_markets()
        symbols = [
            market['symbol'].lower().replace("/", "") 
            for market in markets.values() 
            if market.get('spot') and market.get('active')
        ]
        print(f"✅ Loaded {len(symbols)} active spot trading pairs.")
        return sorted(symbols)
    except Exception as e:
        print(f"⚠️ Failed to fetch symbols: {e}. Using fallback.")
        return ["btcusdt", "ethusdt", "solusdt"]

def get_today_file(symbol):
    date_str = datetime.now().strftime("%Y-%m-%d")
    symbol_upper = symbol.upper()
    dir_path = os.path.join(DATA_DIR, symbol_upper)
    os.makedirs(dir_path, exist_ok=True)
    return os.path.join(dir_path, f"{symbol_upper}_{date_str}.parquet")

def save_buffer(symbol, force=False):
    if symbol not in buffers or not buffers[symbol]:
        return

    with file_locks[symbol]:
        try:
            new_df = pd.DataFrame(buffers[symbol])
            file_path = get_today_file(symbol)
            temp_path = file_path + ".tmp"

            if os.path.exists(file_path):
                existing = pd.read_parquet(file_path)
                combined = pd.concat([existing, new_df], ignore_index=True)
            else:
                combined = new_df

            combined.to_parquet(temp_path, index=False)
            shutil.move(temp_path, file_path)

            last_price[symbol] = combined['price'].iloc[-1]

            print(f"💾 [{symbol.upper()}] Saved {len(buffers[symbol]):,} ticks | "
                  f"Total: {len(combined):,} | Last: ${last_price[symbol]:,.2f}")

            buffers[symbol].clear()
            last_save[symbol] = time.time()

        except Exception as e:
            print(f"Save error [{symbol}]: {e}")

def on_message(ws, message, batch_symbols):
    global last_log_time
    try:
        data = json.loads(message)
        if data.get('e') == 'trade':
            symbol = data['s'].lower()
            if symbol in buffers:
                trade = {
                    'timestamp': pd.to_datetime(data['T'], unit='ms'),
                    'price': float(data['p']),
                    'quantity': float(data['q']),
                    'is_buyer_maker': bool(data['m'])
                }
                buffers[symbol].append(trade)
                last_price[symbol] = trade['price']

                # Auto save
                if time.time() - last_save.get(symbol, 0) >= SAVE_INTERVAL:
                    save_buffer(symbol)

                # Print BTC price every 10 seconds for monitoring
                if symbol == "btcusdt" and time.time() - last_log_time >= 10:
                    print(f"📊 BTC/USDT Live → ${last_price.get('btcusdt', 0):,.2f} | "
                          f"Time: {datetime.now().strftime('%H:%M:%S')}")
                    last_log_time = time.time()

    except Exception as e:
        pass  # Silent fail on individual messages

def create_websocket_for_batch(batch_id, batch_symbols):
    streams = [f"{s}@trade" for s in batch_symbols]
    url = f"wss://stream.binance.com:9443/stream?streams={'/'.join(streams)}"

    def on_open(ws):
        print(f"✅ Batch {batch_id} connected ({len(batch_symbols)} symbols)")

    ws = websocket.WebSocketApp(
        url,
        on_open=on_open,
        on_message=lambda ws, msg: on_message(ws, msg, batch_symbols),
        on_error=lambda ws, err: print(f"Batch {batch_id} error: {err}"),
        on_close=lambda ws, code, msg: print(f"Batch {batch_id} closed")
    )
    ws.run_forever(ping_interval=30, ping_timeout=10)

def periodic_saver():
    while not stop_event.is_set():
        time.sleep(SAVE_INTERVAL)
        for symbol in list(buffers.keys()):
            save_buffer(symbol)

def start_recorder():
    all_symbols = get_all_spot_symbols()
    
    for symbol in all_symbols:
        buffers[symbol] = []
        last_save[symbol] = time.time()
        file_locks[symbol] = threading.Lock()
        last_price[symbol] = 0.0

    batches = [all_symbols[i:i + BATCH_SIZE] for i in range(0, len(all_symbols), BATCH_SIZE)]
    
    print(f"🚀 Starting Full Tick Recorder for {len(all_symbols)} symbols in {len(batches)} batches...")

    # Start periodic saver
    threading.Thread(target=periodic_saver, daemon=True).start()

    # Start WebSocket batches
    for i, batch in enumerate(batches):
        t = threading.Thread(target=create_websocket_for_batch, args=(i+1, batch), daemon=True)
        t.start()
        time.sleep(0.4)

    print("✅ Recorder is now running. You will see BTC price updates every 10 seconds.")

    # Keep alive
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down recorder...")
        stop_event.set()
        for symbol in buffers:
            save_buffer(symbol, force=True)
        print("Final save completed.")

if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
    start_recorder()