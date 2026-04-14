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
import queue

# ========================= CONFIG =========================
DATA_DIR = "data/ticks"
SAVE_INTERVAL = 5          # seconds
BATCH_SIZE = 180           # Safe number of symbols per WebSocket (Binance limit ~200-250)

os.makedirs(DATA_DIR, exist_ok=True)

# Global buffers and locks
buffers = {}
last_save = {}
file_locks = {}
symbol_to_batch = {}

stop_event = threading.Event()

# =========================================================

def get_all_spot_symbols():
    """Get all active spot trading pairs from Binance"""
    exchange = ccxt.binance()
    markets = exchange.load_markets()
    symbols = [
        market['symbol'].lower().replace("/", "") 
        for market in markets.values() 
        if market['spot'] and market['active']
    ]
    print(f"Found {len(symbols)} active spot trading pairs.")
    return sorted(symbols)

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

            print(f"💾 [{symbol.upper()}] Saved {len(buffers[symbol]):,} ticks | "
                  f"Total: {len(combined):,} | Last: ${combined['price'].iloc[-1]:,.2f}")

            buffers[symbol].clear()
            last_save[symbol] = time.time()

        except Exception as e:
            print(f"Save error [{symbol}]: {e}")

def on_message(ws, message, batch_symbols):
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

                # Save periodically
                if time.time() - last_save[symbol] >= SAVE_INTERVAL:
                    save_buffer(symbol)
    except Exception as e:
        print(f"Message error: {e}")

def create_websocket_for_batch(batch_id, batch_symbols):
    streams = [f"{s}@trade" for s in batch_symbols]
    url = f"wss://stream.binance.com:9443/stream?streams={'/'.join(streams)}"

    def on_open(ws):
        print(f"✅ Batch {batch_id} connected | {len(batch_symbols)} symbols")

    def on_message_wrapper(ws, message):
        on_message(ws, message, batch_symbols)

    ws = websocket.WebSocketApp(
        url,
        on_open=on_open,
        on_message=on_message_wrapper,
        on_error=lambda ws, err: print(f"Batch {batch_id} error: {err}"),
        on_close=lambda ws, code, msg: print(f"Batch {batch_id} closed. Reconnecting...")
    )
    ws.run_forever(ping_interval=30, ping_timeout=10)

def start_recorder():
    all_symbols = get_all_spot_symbols()
    
    # Initialize buffers and locks
    for symbol in all_symbols:
        buffers[symbol] = []
        last_save[symbol] = time.time()
        file_locks[symbol] = threading.Lock()

    # Split into batches
    batches = [all_symbols[i:i + BATCH_SIZE] for i in range(0, len(all_symbols), BATCH_SIZE)]
    
    print(f"Starting {len(batches)} WebSocket connections for {len(all_symbols)} symbols...")

    threads = []
    for i, batch in enumerate(batches):
        t = threading.Thread(
            target=create_websocket_for_batch,
            args=(i+1, batch),
            daemon=True
        )
        t.start()
        threads.append(t)
        time.sleep(0.5)  # Avoid overwhelming Binance

    print("All batches started. Recording all spot ticks...")

    # Periodic save thread
    def periodic_save():
        while not stop_event.is_set():
            time.sleep(SAVE_INTERVAL)
            for symbol in list(buffers.keys()):
                save_buffer(symbol)
    threading.Thread(target=periodic_save, daemon=True).start()

    # Keep main thread alive
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("\nShutting down recorder...")
        stop_event.set()
        for symbol in buffers:
            save_buffer(symbol, force=True)
        print("Final save completed.")

if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
    start_recorder()