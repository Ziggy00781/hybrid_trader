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
SAVE_INTERVAL = 5          # seconds
BATCH_SIZE = 120           # Reduced for stability (was 180)

os.makedirs(DATA_DIR, exist_ok=True)

buffers = {}
last_save = {}
file_locks = {}
last_price = {}
stop_event = threading.Event()

def get_all_spot_symbols():
    try:
        exchange = ccxt.binance()
        markets = exchange.load_markets()
        symbols = [m['symbol'].lower().replace("/", "") 
                   for m in markets.values() 
                   if m.get('type') == 'spot' and m.get('active')]
        print(f"✅ Loaded {len(symbols)} active spot trading pairs.")
        return sorted(symbols)
    except Exception as e:
        print(f"⚠️ Failed to fetch symbols: {e}")
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
            print(f"❌ Save error [{symbol}]: {e}")

def on_message(ws, message):
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

                # Auto-save
                if time.time() - last_save.get(symbol, 0) >= SAVE_INTERVAL:
                    save_buffer(symbol)

                # BTC live price every 8 seconds
                if symbol == "btcusdt" and time.time() - last_btc_log >= 8:
                    print(f"📊 BTC/USDT → ${last_price.get('btcusdt', 0):,.2f} | "
                          f"{datetime.now().strftime('%H:%M:%S')}")
                    last_btc_log = time.time()

    except Exception as e:
        print(f"❌ Message error: {e}")

def create_websocket_for_batch(batch_id, batch_symbols):
    streams = [f"{s}@trade" for s in batch_symbols]
    url = f"wss://stream.binance.com:9443/stream?streams={'/'.join(streams)}"

    def on_open(ws):
        print(f"✅ Batch {batch_id} connected ({len(batch_symbols)} symbols)")

    def on_error(ws, error):
        print(f"❌ Batch {batch_id} error: {error}")
        if not stop_event.is_set():
            ws.close()

    def on_close(ws, close_status_code, close_msg):
        print(f"❌ Batch {batch_id} closed")
        if not stop_event.is_set():
            print(f"🔄 Reconnecting batch {batch_id}...")
            create_websocket_for_batch(batch_id, batch_symbols)

    ws = websocket.WebSocketApp(
        url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
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

    batches = [all_symbols[i:i+BATCH_SIZE] for i in range(0, len(all_symbols), BATCH_SIZE)]

    print(f"🚀 Starting recorder for {len(all_symbols)} symbols in {len(batches)} batches...")

    threading.Thread(target=periodic_saver, daemon=True).start()

    for i, batch in enumerate(batches):
        t = threading.Thread(target=create_websocket_for_batch, args=(i+1, batch), daemon=True)
        t.start()
        time.sleep(0.4)

    print("✅ All batches started. Waiting for trades... (BTC price should appear soon)")

    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        stop_event.set()
        for symbol in buffers:
            save_buffer(symbol, force=True)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
    start_recorder()