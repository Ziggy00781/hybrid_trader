import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import glob
from datetime import datetime

st.set_page_config(page_title="Live BTC Chart", layout="wide", initial_sidebar_state="collapsed")
st.title("🟢 Live BTC/USDT — 1 Minute Candles")
st.caption("Data from local tick recorder")

class LiveChart:
    def __init__(self):
        self.TICK_DIR = "data/ticks/BTC_USDT"

    def get_latest_file(self):
        if not os.path.exists(self.TICK_DIR):
            st.error(f"Directory not found: {self.TICK_DIR}")
            st.stop()
        
        files = glob.glob(f"{self.TICK_DIR}/BTC_USDT_*.parquet")
        if not files:
            st.warning("No tick files found. Start the recorder first.")
            st.stop()
        
        return max(files, key=os.path.getmtime)

    def process_to_candles(self, ticks_df):
        if ticks_df.empty:
            return pd.DataFrame()

        st.info(f"Columns: {list(ticks_df.columns)}")

        # Detect time column
        time_col = next((col for col in ticks_df.columns if col.lower() in ['timestamp', 'time', 'ts', 'datetime']), None)
        if time_col is None:
            st.error("No timestamp column found. Fix your recorder to save 'timestamp'.")
            st.stop()

        ticks_df = ticks_df.copy()
        ticks_df[time_col] = pd.to_datetime(ticks_df[time_col])
        ticks_df = ticks_df.set_index(time_col).sort_index()

        price_col = 'price'
        qty_col = 'quantity'

        candles = ticks_df.resample('1T').agg({
            price_col: ['first', 'max', 'min', 'last'],
            qty_col: 'sum'
        })
        candles.columns = ['open', 'high', 'low', 'close', 'volume']
        return candles

    def render(self):
        try:
            latest_file = self.get_latest_file()
            ticks_df = pd.read_parquet(latest_file)

            candles = self.process_to_candles(ticks_df)
            if candles.empty or len(candles) < 2:
                st.info("Waiting for more data...")
                return

            df = candles.tail(240).copy()  # last 4 hours

            fig = go.Figure()

            # Candlestick
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['open'], high=df['high'], low=df['low'], close=df['close'],
                increasing_line_color='#00ff88', decreasing_line_color='#ff4444',
                name='BTC/USDT'
            ))

            # Volume on secondary axis
            fig.add_trace(go.Bar(
                x=df.index, y=df['volume'],
                name='Volume',
                marker_color='rgba(100, 149, 237, 0.5)',
                yaxis='y2'
            ))

            fig.update_layout(
                title="BTC/USDT Live 1-Minute Chart",
                height=700,
                template="plotly_dark",
                xaxis_rangeslider_visible=True,
                yaxis=dict(title="Price (USDT)"),
                yaxis2=dict(title="Volume", overlaying="y", side="right"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            st.plotly_chart(fig, use_container_width=True, key="live_btc_chart")

            # Metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Current Price", f"${df['close'].iloc[-1]:,.2f}")
            with col2:
                st.metric("Last Minute Volume", f"{df['volume'].iloc[-1]:,.0f}")

        except Exception as e:
            st.error(f"Error: {str(e)}")

# ====================== MAIN ======================
chart = LiveChart()

if st.button("🔄 Refresh Chart Now", type="primary"):
    with st.spinner("Loading latest ticks..."):
        chart.render()

chart.render()   # Initial load

st.caption(f"Last checked: {datetime.now().strftime('%H:%M:%S')}")