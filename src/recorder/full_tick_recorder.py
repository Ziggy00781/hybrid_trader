import websocket
import json
import pandas as pd
import os
import time
import threading
from datetime import datetime
import ccxt
import logging

# ==== CONFIGURATION ====
DATA_DIR = "data/ticks"
SAVE_INTERVAL = 5  # seconds
MAX_CONNECTIONS = 100  # Binance rate limit: ~100 streams per IP
LOG_FILE = "tick_recorder.log"

# Setup logging
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize directories
os.makedirs(DATA_DIR, exist_ok=True)

# Global state
buffers = {}        # {symbol: [trade_data]}
last_save = {}      # {symbol: timestamp}
file_locks = {}     # {symbol: threading.Lock()}
last_prices = {}    # {symbol: last_price}
stop_event = threading.Event()
connection_count = 0
active_symbols = []

def get_all_spot_symbols():
    try:
        exchange = ccxt.binance()
        markets = exchange.load_markets()
        symbols = [m['symbol'].lower().replace("/", "") 
                   for m in markets.values() 
                   if m.get('type') == 'spot' and m.get('active')]
        logging.info(f"✅ Loaded {len(symbols)} active spot trading pairs.")
        return sorted(symbols)
    except Exception as e:
        logging.error(f"⚠️ Failed to fetch symbols: {e}")
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

    with file_locks.get(symbol, threading.Lock()):
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

            last_prices[symbol] = combined['price'].iloc[-1]

            logging.info(f"💾 [{symbol.upper()}] Saved {len(buffers[symbol]):,} ticks | "
                         f"Total: {len(combined):,} | Last: ${last_prices[symbol]:,.2f}")

            buffers[symbol].clear()
            last_save[symbol] = time.time()

        except Exception as e:
            logging.error(f"❌ Save error [{symbol}]: {e}")

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
                last_prices[symbol] = trade['price']

                if time.time() - last_save.get(symbol, 0) >= SAVE_INTERVAL:
                    save_buffer(symbol)

                if symbol == "btcusdt" and time.time() - last_btc_log >= 8:
                    logging.info(f"📊 BTC/USDT → ${last_prices.get('btcusdt', 0):,.2f} | "
                                 f"{datetime.now().strftime('%H:%M:%S')}")
                    last_btc_log = time.time()

    except Exception as e:
        logging.error(f"❌ Message error: {e}")

def create_websocket_for_symbol(symbol):
    global connection_count
    if connection_count >= MAX_CONNECTIONS:
        logging.warning(f"⚠️ Max connections ({MAX_CONNECTIONS}) reached. Skipping {symbol}")
        return

    streams = [f"{symbol}@trade"]
    url = f"wss://stream.binance.com:9443/stream?streams={'/'.join(streams)}"

    def on_open(ws):
        logging.info(f"✅ Connected to {symbol}")

    def on_error(ws, error):
        logging.error(f"❌ Error in {symbol}: {error}")
        if not stop_event.is_set():
            ws.close()

    def on_close(ws, close_status_code, close_msg):
        logging.info(f"❌ {symbol} disconnected. Reconnecting...")
        if not stop_event.is_set():
            create_websocket_for_symbol(symbol)

    ws = websocket.WebSocketApp(
        url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    ws.run_forever(ping_interval=30, ping_timeout=10)
    connection_count += 1

def periodic_saver():
    while not stop_event.is_set():
        time.sleep(SAVE_INTERVAL)
        for symbol in list(buffers.keys()):
            save_buffer(symbol)

def start_recorder():
    global active_symbols, connection_count
    active_symbols = get_all_spot_symbols()
    logging.info(f"🔄 Starting recorder for {len(active_symbols)} symbols")

    for symbol in active_symbols:
        buffers[symbol] = []
        last_save[symbol] = time.time()
        file_locks[symbol] = threading.Lock()
        last_prices[symbol] = 0.0

    # Start background saver
    threading.Thread(target=periodic_saver, daemon=True).start()

    # Connect to symbols in batches
    for symbol in active_symbols:
        t = threading.Thread(target=create_websocket_for_symbol, args=(symbol,), daemon=True)
        t.start()
        time.sleep(0.1)  # Throttle connection creation

    logging.info("✅ All connections started. Waiting for trades...")

    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        logging.info("\n🛑 Shutting down...")
        stop_event.set()
        for symbol in buffers:
            save_buffer(symbol, force=True)

if __name__ == "__main__":
    last_btc_log = time.time()
    start_recorder()