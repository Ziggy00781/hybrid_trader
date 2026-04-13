import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import glob
import time
from datetime import datetime

st.set_page_config(page_title="Live BTC Chart", layout="wide")
st.title("🟢 Live BTC/USDT — 1 Minute Candles")
st.caption("Debug Version - Price should update now")

TICK_DIR = "data/ticks/BTC_USDT"

if not os.path.exists(TICK_DIR):
    st.error(f"❌ Tick directory not found: {TICK_DIR}")
    st.stop()

def get_latest_tick_file():
    files = glob.glob(f"{TICK_DIR}/BTC_USDT_*.parquet")
    if not files:
        st.error("No parquet files found in the folder")
        st.stop()
    latest = max(files, key=os.path.getmtime)
    st.info(f"📄 **Latest file detected:** {os.path.basename(latest)}")
    st.info(f"   Modified: {datetime.fromtimestamp(os.path.getmtime(latest)).strftime('%H:%M:%S')}")
    return latest

@st.cache_data(ttl=1, show_spinner=False)
def load_latest_ticks(file_path):
    df = pd.read_parquet(file_path)
    st.info(f"   Loaded {len(df):,} ticks | Columns: {list(df.columns)}")
    return df

def ticks_to_candles(ticks_df):
    if ticks_df.empty:
        return pd.DataFrame()

    # Force timestamp handling
    if not isinstance(ticks_df.index, pd.DatetimeIndex):
        if 'timestamp' in ticks_df.columns:
            ticks_df = ticks_df.copy()
            ticks_df['timestamp'] = pd.to_datetime(ticks_df['timestamp'])
            ticks_df = ticks_df.set_index('timestamp')
        else:
            st.warning("No 'timestamp' column and index is not datetime. Using fallback.")

    candles = ticks_df.resample("1min").agg({
        "price": ["first", "max", "min", "last"],
        "quantity": "sum"
    })
    candles.columns = ["open", "high", "low", "close", "volume"]
    return candles

# ====================== MAIN ======================
latest_file = get_latest_tick_file()

ticks = load_latest_ticks(latest_file)
candles = ticks_to_candles(ticks)

if not candles.empty and len(candles) >= 2:
    display_df = candles.tail(240).copy()

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=display_df.index,
        open=display_df['open'], high=display_df['high'],
        low=display_df['low'], close=display_df['close'],
        increasing_line_color='#00ff88', decreasing_line_color='#ff4444'
    ))
    fig.add_trace(go.Bar(
        x=display_df.index, y=display_df['volume'],
        name='Volume', marker_color='rgba(100,149,237,0.6)', yaxis='y2'
    ))

    fig.update_layout(
        title="BTC/USDT — Live 1 Minute Candles",
        height=720,
        template="plotly_dark",
        xaxis_rangeslider_visible=True,
        yaxis=dict(title="Price (USDT)"),
        yaxis2=dict(title="Volume", overlaying="y", side="right"),
    )

    st.plotly_chart(fig, width="stretch", key="live_chart_key")

    current_price = display_df['close'].iloc[-1]
    last_volume = display_df['volume'].iloc[-1]

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Current Price", f"${current_price:,.2f}", delta=None)
    with col2:
        st.metric("Last 1-min Volume", f"{last_volume:,.0f} USDT")

else:
    st.error("Not enough data to build candles yet.")

st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")

# Refresh
time.sleep(3)
st.rerun()