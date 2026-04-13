import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import glob
import time
from datetime import datetime
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="pandas")

st.set_page_config(page_title="Live BTC Chart", layout="wide")
st.title("🔴 Live BTC/USDT — 1 Minute Candles")
st.caption("Built from saved tick data (recorder must be running)")

class LiveChart:
    def __init__(self):
        self.TICK_DIR = "data/ticks/BTC_USDT"
        self.latest_file = self._get_latest_tick_file()
        self.candles = None
        self.display_df = None
        
    def _get_latest_tick_file(self):
        """Get the latest tick file with improved file sorting"""
        try:
            if not os.path.exists(self.TICK_DIR):
                st.error(f"Tick directory not found: {self.TICK_DIR}")
                st.warning("Please ensure:")
                st.warning("1. The tick recorder is running")
                st.warning("2. You have write permissions for this directory")
                st.stop()
                
            # Find all matching files
            file_paths = glob.glob(f"{self.TICK_DIR}/BTC_USDT_*.parquet")
            if not file_paths:
                st.warning("No tick data found. Start the tick recorder first.")
                st.stop()
                
            # Sort by timestamp and return latest
            sorted_files = sorted(file_paths, key=lambda x: os.path.getctime(x))
            return sorted_files[-1]
        except Exception as e:
            st.error(f"Error getting tick file: {str(e)}")
            st.stop()
    
    def _load_latest_ticks(self):
        """Load latest tick data with caching"""
        try:
            if not hasattr(self, 'latest_file') or self.latest_file is None:
                self.latest_file = self._get_latest_tick_file()
                
            if self.latest_file is None:
                st.warning("No tick data found. Start the tick recorder first.")
                st.stop()
                
            st.info(f"Reading from: {os.path.basename(self.latest_file)}")
            return pd.read_parquet(self.latest_file)
        except Exception as e:
            st.error(f"Error loading tick data: {str(e)}")
            st.stop()
    
    def _process_to_candles(self, ticks_df):
        """Convert ticks to 1-minute candles with validation"""
        try:
            if ticks_df.empty:
                return pd.DataFrame()
                
            # Dynamically detect time column (case-insensitive)
            time_col = next((col for col in ticks_df.columns if col.lower() in ['time', 'datetime', 'timestamp']), None)
            
            if time_col is None:
                st.error("No valid time column found in data. Expected: 'time', 'datetime', or 'timestamp'")
                st.stop()
                
            # Ensure timestamps are properly formatted
            ticks_df[time_col] = pd.to_datetime(ticks_df[time_col])
            ticks_df = ticks_df.set_index(time_col)
            
            # Resample to 1-minute candles
            candles = ticks_df.resample('1T').agg({
                'price': ['first', 'max', 'min', 'last'],
                'quantity': 'sum'
            })
            
            candles.columns = ['open', 'high', 'low', 'close', 'volume']
            return candles
        except Exception as e:
            st.error(f"Error processing data to candles: {str(e)}")
            st.stop()
    
    def _render_chart(self):
        """Render the enhanced candlestick chart"""
        try:
            if self.display_df is None or self.candles.empty:
                return
                
            fig = go.Figure(data=[
                go.Candlestick(
                    x=self.display_df.index,
                    open=self.display_df['open'],
                    high=self.display_df['high'],
                    low=self.display_df['low'],
                    close=self.display_df['close'],
                    increasing_line_color='rgba(0, 255, 136, 0.8)',
                    decreasing_line_color='rgba(255, 68, 68, 0.8)',
                    name='Price'
                ),
                go.Scatter(
                    x=self.display_df.index,
                    y=self.display_df['volume'],
                    fill='tozeroy',
                    opacity=0.3,
                    name='Volume'
                )
            ])
            
            fig.update_layout(
                title='BTC/USDT - Live 1 Minute Candles',
                height=720,
                template='plotly_dark',
                xaxis_rangeslider_visible=True,
                xaxis=dict(rangeslider=dict(visible=True)),
                yaxis=dict(title='Price (USD)'),
                margin=dict(l=20, r=20, t=50, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Current price and metrics
            current_price = self.display_df['close'].iloc[-1]
            volume = self.display_df['volume'].iloc[-1]
            
            st.metric("Current Price", f"${current_price:,.2f}")
            st.metric("Last 1 Minute Volume", f"{volume:,.2f} USDT")
            
        except Exception as e:
            st.error(f"Error rendering chart: {str(e)}")
            st.stop()

def main():
    chart = LiveChart()
    
    # Load and process data
    with st.spinner('Loading latest data...'):
        ticks_df = chart._load_latest_ticks()
        chart.candles = chart._process_to_candles(ticks_df)
    
    # Prepare display data
    if not chart.candles.empty:
        chart.display_df = chart.candles.tail(240).copy()  # last ~4 hours
        
        # Render the chart
        chart._render_chart()
    else:
        st.info("Not enough ticks yet to build candles...")
    
    # Update timestamp and status
    st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')} | Recorder must be running in another terminal")
    
    # Auto refresh
    time.sleep(2)
    st.rerun()

if __name__ == "__main__":
    main()
