# src/dashboard/live_chart.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import glob
import time
from datetime import datetime

st.set_page_config(page_title="Live BTC Chart", layout="wide")
st.title("🔴 Live BTC/USDT — 1 Minute Candles")
st.caption("Built from saved tick data (recorder must be running)")

# Path to tick files
TICK_DIR = "data/ticks/BTC_USDT"

if not os.path.exists(TICK_DIR):
    st.error(f"Tick directory not found: {TICK_DIR}\nMake sure the tick recorder is running!")
    st.stop()

# Find the latest tick file
def get_latest_tick_file():
    files = glob.glob(f"{TICK_DIR}/BTC_USDT_*.parquet")
    if not files:
        return None
    return max(files, key=os.path.getctime)

# Load latest data
latest_file = get_latest_tick_file()

if latest_file is None:
    st.warning("No tick data found yet. Start the tick recorder first.")
    st.stop()

st.info(f"Reading from: {os.path.basename(latest_file)}")

@st.cache_data(ttl=2)  # Refresh every 2 seconds
def load_latest_ticks(file_path):
    df = pd.read_parquet(file_path)
    return df

ticks = load_latest_ticks(latest_file)

# Build 1-minute candles from raw ticks
def ticks_to_candles(ticks_df):
    if ticks_df.empty:
        return pd.DataFrame()
    # Resample to 1 minute
    candles = ticks_df.resample("1min").agg({
        "price": ["first", "max", "min", "last"],
        "quantity": "sum"
    })
    candles.columns = ["open", "high", "low", "close", "volume"]
    return candles

candles = ticks_to_candles(ticks)

# Display the chart
if not candles.empty:
    display_df = candles.tail(240).copy()  # last ~4 hours
    
    fig = go.Figure(data=[go.Candlestick(
        x=display_df.index,
        open=display_df['open'],
        high=display_df['high'],
        low=display_df['low'],
        close=display_df['close'],
        increasing_line_color='#00ff88',
        decreasing_line_color='#ff4444'
    )])
    
    fig.update_layout(
        title=f"BTC/USDT — Live 1 Minute Candles (from ticks)",
        height=720,
        template="plotly_dark",
        xaxis_rangeslider_visible=False
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Current price
    current_price = display_df['close'].iloc[-1]
    st.metric("Current Price", f"${current_price:,.2f}")
else:
    st.info("Not enough ticks yet to build candles...")

st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')} | Recorder must be running in another terminal")

# Auto refresh
time.sleep(2)
st.rerun()