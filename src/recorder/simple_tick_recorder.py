import websocket
import json
import pandas as pd
import os
import time
import signal
import sys
from datetime import datetime

# Config
SYMBOL = "btcusdt"
DATA_DIR = "data/ticks/BTC_USDT"
os.makedirs(DATA_DIR, exist_ok=True)

buffer = []
last_save = time.time()
SAVE_INTERVAL = 5

def get_today_file():
    date_str = datetime.now().strftime("%Y-%m-%d")
    return os.path.join(DATA_DIR, f"BTC_USDT_{date_str}.parquet")

def save_buffer(force=False):
    global last_save
    if not buffer:
        return
    if force or (time.time() - last_save >= SAVE_INTERVAL):
        try:
            new_df = pd.DataFrame(buffer)
            file_path = get_today_file()
            if os.path.exists(file_path):
                existing = pd.read_parquet(file_path)
                combined = pd.concat([existing, new_df], ignore_index=True)
            else:
                combined = new_df
            combined.to_parquet(file_path, index=False)
            print(f"💾 Saved {len(buffer):,} ticks | Total: {len(combined):,} | Last price: ${combined['price'].iloc[-1]:,.2f}")
            buffer.clear()
            last_save = time.time()
        except Exception as e:
            print(f"Save error: {e}")

def on_message(ws, message):
    try:
        data = json.loads(message)
        if data.get('e') == 'trade':
            trade = {
                'timestamp': pd.to_datetime(data['T'], unit='ms'),
                'price': float(data['p']),
                'quantity': float(data['q']),
                'is_buyer_maker': bool(data['m'])
            }
            buffer.append(trade)
            save_buffer()
    except Exception as e:
        print(f"Msg error: {e}")

def on_error(ws, error):
    print(f"WebSocket error: {error}")

def on_close(ws, close_status_code, close_msg):
    print(f"Connection closed (code: {close_status_code}). Reconnecting in 10s...")

def on_open(ws):
    print("✅ Connected to Binance trade stream!")

def signal_handler(sig, frame):
    print("\n🛑 Stopping recorder...")
    save_buffer(force=True)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def start_websocket():
    ws = websocket.WebSocketApp(
        f"wss://stream.binance.com:9443/ws/{SYMBOL}@trade",
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    ws.run_forever(ping_interval=30, ping_timeout=10)

if __name__ == "__main__":
    print("🚀 Starting simple Binance tick recorder...")
    print("Press Ctrl+C to stop cleanly")
    while True:
        try:
            start_websocket()
        except Exception as e:
            print(f"Restarting due to error: {e}")
            time.sleep(10)