import websocket
import json
import pandas as pd
import os
import time
import signal
import sys
from datetime import datetime
import tempfile
import shutil

# ================== CONFIG ==================
SYMBOL = "btcusdt"
DATA_DIR = "data/ticks/BTC_USDT"
SAVE_INTERVAL = 5

os.makedirs(DATA_DIR, exist_ok=True)

buffer = []
last_save = time.time()

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
            temp_path = file_path + ".tmp"

            if os.path.exists(file_path):
                existing = pd.read_parquet(file_path)
                combined = pd.concat([existing, new_df], ignore_index=True)
            else:
                combined = new_df

            # Atomic write: write to temp file first, then rename (much safer)
            combined.to_parquet(temp_path, index=False)
            shutil.move(temp_path, file_path)   # atomic operation

            print(f"💾 Saved {len(buffer):,} ticks | Total: {len(combined):,} | "
                  f"Last price: ${combined['price'].iloc[-1]:,.2f} | {datetime.now().strftime('%H:%M:%S')}")
            
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
        print(f"Message error: {e}")

def on_error(ws, error):
    print(f"WebSocket Error: {error}")

def on_close(ws, close_status_code, close_msg):
    print(f"Connection closed. Reconnecting in 10s...")

def on_open(ws):
    print("✅ Successfully connected to Binance BTC/USDT trade stream!")

def signal_handler(sig, frame):
    print("\n🛑 Stopping recorder... Final save")
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
    print("🚀 Starting Binance Tick Recorder on Tokyo VPS...")
    print("Press Ctrl+C to stop")
    
    while True:
        try:
            start_websocket()
        except Exception as e:
            print(f"Unexpected error: {e}. Restarting in 10s...")
            time.sleep(10)