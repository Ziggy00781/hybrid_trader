import websocket
import json
import pandas as pd
import os
import time
import threading
from datetime import datetime
import ccxt
import logging
import shutil

# ==== CONFIGURATION ====
DATA_DIR = "data/ticks"
SAVE_INTERVAL = 2          # seconds → Parquet files updated frequently (as requested)
TOP_N = 10
LOG_FILE = "tick_recorder.log"

# Setup directories and logging
os.makedirs(DATA_DIR, exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'
)

# Global state
buffers = {}          # {symbol: list of trades}
last_save = {}
file_locks = {}
last_prices = {}
stop_event = threading.Event()
active_symbols = []


def get_top_usdt_symbols(n=10):
    """Dynamically get top N USDT spot pairs by 24h quote volume (includes BTC & ETH)."""
    try:
        exchange = ccxt.binance()
        markets = exchange.load_markets()
        tickers = exchange.fetch_tickers()

        usdt_data = []
        for symbol, market in markets.items():
            if (market.get('quote') == 'USDT' and
                market.get('spot') and
                market.get('active')):
                ticker = tickers.get(symbol, {})
                volume = ticker.get('quoteVolume', 0) or 0
                sym_lower = symbol.replace('/', '').lower()
                usdt_data.append((sym_lower, volume))

        usdt_data.sort(key=lambda x: x[1], reverse=True)
        top_symbols = [sym for sym, vol in usdt_data[:n]]
        logging.info(f"Loaded top {len(top_symbols)} cryptocurrencies by 24h volume: {[s.upper() for s in top_symbols]}")
        return top_symbols
    except Exception as e:
        logging.error(f"Failed to fetch top symbols: {e}")
        fallback = ["btcusdt", "ethusdt", "solusdt", "bnbusdt", "xrpusdt",
                    "dogeusdt", "tonusdt", "adausdt", "shibusdt", "trxusdt"]
        logging.info(f"Using fallback top 10 (includes BTC & ETH): {[s.upper() for s in fallback]}")
        return fallback


def get_today_file(symbol):
    date_str = datetime.now().strftime("%Y-%m-%d")
    symbol_upper = symbol.upper()
    dir_path = os.path.join(DATA_DIR, symbol_upper)
    os.makedirs(dir_path, exist_ok=True)
    return os.path.join(dir_path, f"{symbol_upper}_{date_str}.parquet")


def save_buffer(symbol):
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

            # Ensure clean sorted data
            combined = combined.sort_values('timestamp').reset_index(drop=True)
            combined.to_parquet(temp_path, index=False)
            shutil.move(temp_path, file_path)

            if len(combined) > 0:
                last_prices[symbol] = combined['price'].iloc[-1]

            logging.info(f"[{symbol.upper()}] Saved {len(new_df):,} ticks | Total: {len(combined):,} | Last: ${last_prices.get(symbol, 0):,.2f}")

            buffers[symbol].clear()
            last_save[symbol] = time.time()
        except Exception as e:
            logging.error(f"Save error [{symbol}]: {e}")


def on_message(ws, message):
    try:
        msg = json.loads(message)

        # Handle Binance combined-stream wrapper
        if isinstance(msg, dict) and 'stream' in msg:
            stream_name = msg['stream']
            data = msg.get('data', msg)
            symbol = stream_name.split('@')[0]          # e.g. "btcusdt"
        else:
            data = msg
            symbol = data.get('s', '').lower()

        if data.get('e') == 'trade' and symbol in buffers:
            trade = {
                'timestamp': pd.to_datetime(data['T'], unit='ms'),
                'price': float(data['p']),
                'quantity': float(data['q']),
                'is_buyer_maker': bool(data['m'])
            }
            buffers[symbol].append(trade)
            last_prices[symbol] = trade['price']

            # Frequent Parquet update
            if time.time() - last_save.get(symbol, 0) >= SAVE_INTERVAL:
                save_buffer(symbol)

            # Nice BTC price log every ~8 seconds
            if symbol == "btcusdt":
                if time.time() - getattr(on_message, 'last_btc_log', 0) >= 8:
                    logging.info(f"BTC/USDT → ${last_prices.get('btcusdt', 0):,.2f} | {datetime.now().strftime('%H:%M:%S')}")
                    on_message.last_btc_log = time.time()

    except Exception as e:
        logging.error(f"On message error: {e}")


def periodic_saver():
    """Background thread that forces a save every SAVE_INTERVAL seconds."""
    while not stop_event.is_set():
        time.sleep(SAVE_INTERVAL)
        for symbol in list(buffers.keys()):
            save_buffer(symbol)


def on_open(ws):
    logging.info("✅ Connected to Binance multi-stream WebSocket (top 10 cryptos)")


def on_error(ws, error):
    logging.error(f"WebSocket error: {error}")


def on_close(ws, *args):
    logging.warning("WebSocket closed. Reconnecting in 3 seconds...")
    if not stop_event.is_set():
        time.sleep(3)
        start_websocket()


def start_websocket():
    streams = [f"{sym}@trade" for sym in active_symbols]
    url = f"wss://stream.binance.com:9443/stream?streams={'/'.join(streams)}"
    logging.info(f"Starting multi-stream WebSocket for {len(streams)} symbols")

    ws = websocket.WebSocketApp(url,
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    ws.run_forever(ping_interval=25, ping_timeout=10)


def start_recorder():
    global active_symbols
    active_symbols = get_top_usdt_symbols(TOP_N)

    for symbol in active_symbols:
        buffers[symbol] = []
        last_save[symbol] = time.time()
        file_locks[symbol] = threading.Lock()
        last_prices[symbol] = 0.0

    logging.info(f"🚀 Recorder started for top {TOP_N} cryptocurrencies (BTC & ETH included)")

    # Background saver ensures Parquet files stay up-to-date even during quiet periods
    threading.Thread(target=periodic_saver, daemon=True).start()

    # Start the single multi-stream WebSocket (blocks until shutdown)
    start_websocket()


if __name__ == "__main__":
    on_message.last_btc_log = time.time()   # initialize BTC logger
    try:
        start_recorder()
    except KeyboardInterrupt:
        logging.info("🛑 Shutdown signal received")
        stop_event.set()
        for symbol in buffers:
            save_buffer(symbol)
        logging.info("All data saved. Recorder stopped.")
    except Exception as e:
        logging.critical(f"Critical error: {e}")