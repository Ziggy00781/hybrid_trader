import time
import pandas as pd
import os
import nest_asyncio   # ← Added to fix event loop issues
from datetime import datetime
from binance import ThreadedWebsocketManager
import logging

nest_asyncio.apply()   # This patches asyncio for Python 3.10+ compatibility

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

class TickRecorder:
    def __init__(self, symbol: str = "BTCUSDT"):
        self.symbol = symbol
        self.buffer = []
        self.last_save_time = time.time()
        self.save_interval = 5
        self.max_buffer_size = 10000

        self.today = datetime.now().date()
        self.data_dir = "data/ticks/BTC_USDT"
        os.makedirs(self.data_dir, exist_ok=True)
        self.current_file = self._get_today_file()

    def _get_today_file(self):
        date_str = self.today.strftime("%Y-%m-%d")
        return os.path.join(self.data_dir, f"BTC_USDT_{date_str}.parquet")

    def _save_buffer(self, force: bool = False):
        if not self.buffer:
            return
        if force or (time.time() - self.last_save_time >= self.save_interval) or len(self.buffer) >= self.max_buffer_size:
            try:
                new_df = pd.DataFrame(self.buffer)
                if os.path.exists(self.current_file):
                    existing = pd.read_parquet(self.current_file)
                    combined = pd.concat([existing, new_df], ignore_index=True)
                else:
                    combined = new_df

                combined.to_parquet(self.current_file, index=False)
                logger.info(f"💾 Saved {len(self.buffer):,} ticks | Total: {len(combined):,} | Last price: ${combined['price'].iloc[-1]:,.2f}")
                self.buffer.clear()
                self.last_save_time = time.time()
            except Exception as e:
                logger.error(f"Save failed: {e}")

    def handle_message(self, msg):
        try:
            if msg.get('e') == 'trade':
                trade = {
                    'timestamp': pd.to_datetime(msg['T'], unit='ms'),
                    'price': float(msg['p']),
                    'quantity': float(msg['q']),
                    'is_buyer_maker': bool(msg['m'])
                }
                self.buffer.append(trade)
                self._save_buffer()

                if datetime.now().date() != self.today:
                    logger.info("🌅 New day - rolling file")
                    self._save_buffer(force=True)
                    self.today = datetime.now().date()
                    self.current_file = self._get_today_file()
                    self.buffer.clear()
        except Exception as e:
            logger.error(f"Msg error: {e}")

    def start(self):
        logger.info(f"🚀 Starting recorder for {self.symbol}...")

        for attempt in range(8):  # More attempts
            try:
                twm = ThreadedWebsocketManager()
                twm.start()
                twm.start_trade_socket(callback=self.handle_message, symbol=self.symbol.lower())
                logger.info("✅ Websocket connected!")
                break
            except Exception as e:
                logger.warning(f"Attempt {attempt+1}/8 failed: {e}")
                time.sleep(8 if attempt > 2 else 4)
        else:
            logger.error("❌ Could not connect after retries. Check internet/DNS/VPN.")
            return

        try:
            while True:
                time.sleep(1)
                self._save_buffer()
        except KeyboardInterrupt:
            logger.info("🛑 Stopping - final save")
            self._save_buffer(force=True)
            twm.stop()


if __name__ == "__main__":
    recorder = TickRecorder()
    recorder.start()