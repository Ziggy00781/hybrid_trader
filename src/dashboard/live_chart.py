import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import glob
import time
from datetime import datetime

st.set_page_config(page_title="Live BTC Chart", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <h1 style='text-align: center; color: #00ff88; margin-bottom: 8px;'>BTC/USDT Live</h1>
    <p style='text-align: center; color: #aaaaaa; font-size: 18px; margin-top: 0;'>1 Minute Candles • Real-time Tick Data</p>
""", unsafe_allow_html=True)

TICK_DIR = "data/ticks/BTCUSDT"

def get_latest_tick_file():
    files = glob.glob(f"{TICK_DIR}/BTCUSDT_*.parquet")
    if not files:
        st.error("No tick files found. Start the recorder first.")
        st.stop()
    return max(files, key=os.path.getmtime)

@st.cache_data(ttl=4, show_spinner=False)
def load_latest_ticks(file_path):
    """Safe loading with retry for concurrent write issues"""
    for attempt in range(5):
        try:
            return pd.read_parquet(file_path)
        except Exception as e:
            if attempt == 4:
                st.warning(f"Could not read parquet file after retries: {e}")
                time.sleep(1)
                return pd.DataFrame()
            time.sleep(0.5)  # Wait a bit and retry
    return pd.DataFrame()

def ticks_to_candles(ticks_df):
    if ticks_df.empty or len(ticks_df) < 10:
        return pd.DataFrame()

    if not isinstance(ticks_df.index, pd.DatetimeIndex):
        if 'timestamp' in ticks_df.columns:
            ticks_df = ticks_df.copy()
            ticks_df['timestamp'] = pd.to_datetime(ticks_df['timestamp'])
            ticks_df = ticks_df.set_index('timestamp')

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

if not candles.empty and len(candles) >= 5:
    display_df = candles.tail(300).copy()

    current_price = display_df['close'].iloc[-1]
    prev_price = display_df['close'].iloc[-2]
    price_change = current_price - prev_price

    last_vol_btc = display_df['volume'].iloc[-1]
    last_vol_usdt = last_vol_btc * current_price

    col_price, col_vol = st.columns([3, 1])
    
    with col_price:
        st.metric(
            label="Current BTC Price",
            value=f"${current_price:,.2f}",
            delta=f"{price_change:+.2f}"
        )
    
    with col_vol:
        st.metric(
            label="Last Minute Volume",
            value=f"{last_vol_btc:.6f} BTC",
            delta=f"${last_vol_usdt:,.0f} USDT"
        )

    # Chart
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=display_df.index,
        open=display_df['open'],
        high=display_df['high'],
        low=display_df['low'],
        close=display_df['close'],
        increasing_line_color='#00ff88',
        decreasing_line_color='#ff5252',
        name='Price'
    ))
    fig.add_trace(go.Bar(
        x=display_df.index,
        y=display_df['volume'],
        name='Volume',
        marker_color='rgba(100, 149, 237, 0.65)',
        yaxis='y2'
    ))

    fig.update_layout(
        height=720,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=40, b=30),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        xaxis_rangeslider_visible=False,
        xaxis=dict(showgrid=True, gridcolor="#1e1e1e"),
        yaxis=dict(title="Price (USDT)", showgrid=True, gridcolor="#1e1e1e"),
        yaxis2=dict(title="Volume (BTC)", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    st.plotly_chart(fig, width="stretch", key="live_chart_key")

else:
    st.info("Waiting for more tick data... The recorder is running but not enough candles yet.")

st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')} | Recorder must be running")

time.sleep(3)
st.rerun()