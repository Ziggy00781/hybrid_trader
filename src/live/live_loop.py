# src/live/live_loop.py (clean version)
import time
import logging
import numpy as np
from collections import deque
from src.live.runtime import initialize_runtime, live_predict, fetch_live_ohlcv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class LiveDataManager:
    """Manages real-time data collection and buffering"""
    
    def __init__(self, buffer_size=1100):
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        logger.info(f"LiveDataManager initialized with buffer size {buffer_size}")
    
    def fetch_latest_candle(self) -> bool:
        """Fetch and store the latest candle"""
        try:
            ohlcv = fetch_live_ohlcv(limit=1)
            if ohlcv is not None and len(ohlcv) > 0:
                latest_candle = ohlcv[0]
                self.buffer.append(latest_candle)
                logger.debug(f"Added latest candle: {latest_candle}")
                return True
            else:
                logger.warning("No candle data received")
                return False
        except Exception as e:
            logger.error(f"Error fetching latest candle: {e}")
            return False
    
    def get_buffered_data(self, limit=1024) -> np.ndarray:
        """Get the most recent N candles as numpy array"""
        if len(self.buffer) < limit:
            logger.warning(f"Buffer has only {len(self.buffer)} candles, requested {limit}")
            return None
        
        recent_data = list(self.buffer)[-limit:]
        return np.array(recent_data, dtype=np.float32)
    
    def is_buffer_ready(self, required_length=1024) -> bool:
        """Check if buffer has enough data"""
        return len(self.buffer) >= required_length
    
    def initialize_buffer(self) -> bool:
        """Initialize buffer with historical data"""
        logger.info("Initializing data buffer with historical data...")
        initial_data = fetch_live_ohlcv(limit=self.buffer_size - 100)  # Leave some room
        if initial_data is not None:
            for candle in initial_data:
                self.buffer.append(candle)
            logger.info(f"Initialized buffer with {len(initial_data)} candles")
            return True
        else:
            logger.error("Failed to initialize buffer with historical data")
            return False

def create_buffered_fetch_function(data_manager):
    """Create a fetch function that uses buffered data"""
    def buffered_fetch(limit=1024):
        return data_manager.get_buffered_data(limit)
    return buffered_fetch

def main():
    """Main live prediction loop with 1-minute data updates and 5-minute predictions"""
    logger.info("[live_loop] Starting live prediction loop...")
    logger.info("[live_loop] Configuration: 1m data updates, 5m predictions")
    
    # Initialize runtime components
    if not initialize_runtime():
        logger.error("[live_loop] Failed to initialize runtime components")
        return
    
    # Initialize data manager
    data_manager = LiveDataManager(buffer_size=1100)
    if not data_manager.initialize_buffer():
        logger.error("[live_loop] Failed to initialize data buffer")
        return
    
    # Timing control
    last_prediction_time = time.time()
    prediction_interval = 300  # 5 minutes in seconds
    data_update_interval = 60   # 1 minute in seconds
    
    iteration_count = 0
    last_data_update = time.time()
    
    while True:
        try:
            iteration_count += 1
            current_time = time.time()
            
            # Update data buffer every minute
            if current_time - last_data_update >= data_update_interval:
                logger.info(f"[live_loop] Iteration {iteration_count}: Updating data buffer...")
                success = data_manager.fetch_latest_candle()
                if success:
                    logger.info(f"[live_loop] Buffer status: {len(data_manager.buffer)}/{data_manager.buffer_size} candles")
                last_data_update = current_time
            
            # Make prediction every 5 minutes
            if current_time - last_prediction_time >= prediction_interval:
                logger.info("[live_loop] Time for prediction cycle...")
                
                if data_manager.is_buffer_ready(1024):
                    # Create buffered fetch function
                    buffered_fetch = create_buffered_fetch_function(data_manager)
                    
                    # Execute prediction using buffered data
                    prediction, signal = live_predict(fetch_function=buffered_fetch)
                    
                    # Handle results
                    if prediction is not None:
                        logger.info(f"[live_loop] PREDICTION: {prediction:.6f}, SIGNAL: {signal}")
                        print("\n" + "="*50)
                        print(f"=== PREDICTION RESULT ===")
                        print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                        print(f"Predicted return: {prediction:.6f}")
                        print(f"Trading signal: {signal}")
                        print(f"Buffer size: {len(data_manager.buffer)} candles")
                        print("="*50 + "\n")
                    else:
                        logger.error(f"[live_loop] Prediction failed: {signal}")
                        print(f"\nPrediction failed with signal: {signal}\n")
                        
                    last_prediction_time = current_time
                    
                else:
                    logger.warning("[live_loop] Not enough data in buffer for prediction")
                    print("Waiting for more data to accumulate...")
            
        except KeyboardInterrupt:
            logger.info("[live_loop] Received interrupt signal, shutting down...")
            break
        except Exception as e:
            logger.error(f"[live_loop] Unexpected error in main loop: {e}", exc_info=True)
            print(f"Loop error: {e}")
        
        # Wait before next iteration
        time.sleep(30)  # Check every 30 seconds

if __name__ == "__main__":
    main()