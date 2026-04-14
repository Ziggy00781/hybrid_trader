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
import traceback

# ==== CONFIGURATION ====
DATA_DIR = "data/ticks"
SAVE_INTERVAL = 2          # seconds - frequent Parquet updates
TOP_N = 10
LOG_FILE = "tick_recorder.log"

# Setup directories and logging (console + file)
os.makedirs(DATA_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for maximum verbosity
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a'),
        logging.StreamHandler()  # This prints everything to console too!
    ]
)

logger = logging.getLogger(__name__)

# Global state
buffers = {}
last_save = {}
file_locks = {}
last_prices = {}
stop_event = threading.Event()
active_symbols = []
trade_counters = {}   # For limiting verbose output per symbol

def get_top_usdt_symbols(n=10):
    logger.info("🔍 Fetching top cryptocurrencies by 24h USDT volume...")
    try:
        exchange = ccxt.binance({'enableRateLimit': True})
        markets = exchange.load_markets()
        tickers = exchange.fetch_tickers()

        usdt_data = []
        for symbol, market in markets.items():
            if (market.get('quote') == 'USDT' and
                market.get('spot') and
                market.get('active')):
                ticker = tickers.get(symbol, {})
                volume = float(ticker.get('quoteVolume') or 0)
                sym_lower = symbol.replace('/', '').lower()
                usdt_data.append((sym_lower, volume))

        usdt_data.sort(key=lambda x: x[1], reverse=True)
        top_symbols = [sym for sym, vol in usdt_data[:n]]

        logger.info(f"✅ Loaded top {len(top_symbols)} symbols:")
        for i, sym in enumerate(top_symbols, 1):
            logger.info(f"   {i:2d}. {sym.upper()}")

        return top_symbols
    except Exception as e:
        logger.error(f"❌ Failed to fetch top symbols: {e}")
        logger.error(traceback.format_exc())
        fallback = ["btcusdt", "ethusdt", "solusdt", "bnbusdt", "xrpusdt",
                    "dogeusdt", "tonusdt", "adausdt", "shibusdt", "trxusdt"]
        logger.info(f"Using fallback list (includes BTC & ETH): {[s.upper() for s in fallback]}")
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

            combined = combined.sort_values('timestamp').reset_index(drop=True)
            combined.to_parquet(temp_path, index=False)
            shutil.move(temp_path, file_path)

            if len(combined) > 0:
                last_prices[symbol] = combined['price'].iloc[-1]

            logger.info(f"💾 [{symbol.upper()}] SAVED {len(new_df):,} new ticks → Total: {len(combined):,} | Last price: ${last_prices.get(symbol, 0):,.4f}")
            buffers[symbol].clear()
            last_save[symbol] = time.time()
        except Exception as e:
            logger.error(f"❌ Save error for {symbol}: {e}")
            logger.error(traceback.format_exc())


def on_message(ws, message):
    try:
        msg = json.loads(message)

        # Handle combined stream format
        if isinstance(msg, dict) and 'stream' in msg and 'data' in msg:
            stream_name = msg['stream']
            data = msg['data']
            symbol = stream_name.split('@')[0].lower()
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
            trade_counters[symbol] = trade_counters.get(symbol, 0) + 1

            # Massive debug output (but not too spammy)
            count = trade_counters[symbol]
            if count <= 5 or count % 50 == 0:   # First 5 + every 50th trade
                logger.debug(f"📥 [{symbol.upper()}] Trade #{count} | Price: ${trade['price']:.4f} | Qty: {trade['quantity']:.6f} | BuyerMaker: {trade['is_buyer_maker']}")

            # Frequent save
            if time.time() - last_save.get(symbol, 0) >= SAVE_INTERVAL:
                save_buffer(symbol)

            # BTC price heartbeat
            if symbol == "btcusdt" and time.time() - getattr(on_message, 'last_btc_log', 0) >= 5:
                logger.info(f"🚀 BTC/USDT LIVE → ${last_prices.get('btcusdt', 0):,.2f}  |  Total trades received: {trade_counters.get('btcusdt', 0):,}")
                on_message.last_btc_log = time.time()

    except Exception as e:
        logger.error(f"❌ Error processing message: {e}")
        logger.error(traceback.format_exc())


def periodic_saver():
    logger.info("⏰ Periodic saver thread started")
    while not stop_event.is_set():
        time.sleep(SAVE_INTERVAL)
        for symbol in list(buffers.keys()):
            if buffers.get(symbol):
                logger.debug(f"⏰ Periodic save triggered for {symbol.upper()} (buffer size: {len(buffers[symbol])})")
                save_buffer(symbol)


def on_open(ws):
    logger.info("✅ WebSocket CONNECTED successfully!")
    logger.info(f"   Subscribed to {len(active_symbols)} streams: {[s.upper() for s in active_symbols]}")


def on_error(ws, error):
    logger.error(f"❌ WebSocket ERROR: {error}")
    logger.error(traceback.format_exc())


def on_close(ws, close_status_code, close_msg):
    logger.warning(f"⚠️ WebSocket CLOSED (code: {close_status_code}, msg: {close_msg})")
    if not stop_event.is_set():
        logger.info("🔄 Attempting to reconnect in 5 seconds...")
        time.sleep(5)
        start_websocket()


def start_websocket():
    streams = [f"{sym}@trade" for sym in active_symbols]
    url = f"wss://stream.binance.com:9443/stream?streams={'/'.join(streams)}"

    logger.info("🌐 Starting WebSocket connection...")
    logger.info(f"   URL: {url}")

    ws = websocket.WebSocketApp(url,
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)

    ws.run_forever(ping_interval=20, ping_timeout=10)


def start_recorder():
    global active_symbols
    logger.info("🚀 Starting Hybrid Trader Tick Recorder (Debug Mode)")

    active_symbols = get_top_usdt_symbols(TOP_N)

    for symbol in active_symbols:
        buffers[symbol] = []
        last_save[symbol] = time.time()
        file_locks[symbol] = threading.Lock()
        last_prices[symbol] = 0.0
        trade_counters[symbol] = 0

    logger.info(f"🎯 Recorder initialized for top {TOP_N} cryptos (BTC & ETH included)")

    # Start background saver
    threading.Thread(target=periodic_saver, daemon=True).start()

    # Start WebSocket (this will block)
    start_websocket()


if __name__ == "__main__":
    on_message.last_btc_log = time.time()

    try:
        start_recorder()
    except KeyboardInterrupt:
        logger.info("🛑 Keyboard interrupt received - shutting down gracefully")
        stop_event.set()
        for symbol in buffers:
            save_buffer(symbol)
        logger.info("✅ All buffers saved. Recorder stopped.")
    except Exception as e:
        logger.critical(f"💥 Critical error: {e}")
        logger.critical(traceback.format_exc())