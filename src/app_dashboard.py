import sys
from pathlib import Path

# Fix import path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib

from src.features.ta_regime_features import build_features
from src.backtest.backtester import generate_signals

# ====================== SIMPLE BACKTEST FUNCTION ======================
def simple_backtest(df: pd.DataFrame, signals: pd.DataFrame, 
                   tp_pct: float = 0.005, sl_pct: float = 0.005):
    """Simple backtester that works with your current generate_signals"""
    if df.empty or signals.empty:
        equity = pd.DataFrame(index=df.index)
        equity["equity"] = 1.0
        return equity

    equity = pd.DataFrame(index=df.index)
    equity["equity"] = 1.0
    position = 0
    entry_price = 0.0

    for i in range(1, len(df)):
        idx = df.index[i]
        prev_idx = df.index[i-1]
        
        signal = signals.loc[idx, "signal"] if idx in signals.index else 0
        current_price = df.loc[idx, "close"]
        prev_price = df.loc[prev_idx, "close"]

        # Exit position
        if position != 0:
            ret = (current_price / entry_price) - 1
            if (position == 1 and (ret >= tp_pct or ret <= -sl_pct)) or \
               (position == -1 and (ret <= -tp_pct or ret >= sl_pct)):
                position = 0

        # Enter new position (currently only LONG)
        if position == 0 and signal == 1:
            position = 1
            entry_price = current_price

        # Update equity
        if position != 0:
            equity.loc[idx, "equity"] = equity.loc[prev_idx, "equity"] * (current_price / prev_price)
        else:
            equity.loc[idx, "equity"] = equity.loc[prev_idx, "equity"]

    equity = equity.ffill()
    return equity


# ====================== MAIN DASHBOARD ======================
def main():
    st.set_page_config(page_title="Hybrid Trader", layout="wide")
    st.title("🚀 BTC/USDT 5m – Hybrid AI Backtest Dashboard")
    st.markdown("**Using LightGBM + Regime Features**")

    # ------------------- File Checks -------------------
    RAW_PATH = Path("data/raw/binance_btcusdt_5m.parquet")
    MODEL_PATH = Path("models/btcusdt_5m_lgbm.pkl")

    if not RAW_PATH.exists():
        st.error(f"❌ Raw data not found: `{RAW_PATH}`")
        st.info("Please run `prepare_data.sh` or place the parquet file in the correct location.")
        st.stop()

    if not MODEL_PATH.exists():
        st.error(f"❌ Model not found: `{MODEL_PATH}`")
        st.info("Please train the LightGBM model first (`python -m src.train.train_model`)")
        st.stop()

    # ------------------- Load Data & Model -------------------
    @st.cache_data
    def load_data():
        return pd.read_parquet(RAW_PATH)

    @st.cache_resource
    def load_model():
        return joblib.load(MODEL_PATH)

    df = load_data()
    model = load_model()

    st.success(f"✅ Loaded {len(df):,} candles | Model loaded successfully")

    # ------------------- Sidebar -------------------
    st.sidebar.header("Backtest Controls")
    
    prob_threshold = st.sidebar.slider("Probability Threshold", 0.50, 0.95, 0.65, 0.01)
    tp_pct = st.sidebar.slider("Take Profit (%)", 0.1, 5.0, 0.8, 0.1) 
    sl_pct = st.sidebar.slider("Stop Loss (%)", 0.1, 5.0, 0.8, 0.1)

    # ------------------- Run Analysis -------------------
    with st.spinner("Generating signals and running backtest..."):
        signals = generate_signals(df, prob_threshold=prob_threshold, model=model)
        
        equity = simple_backtest(
            df, 
            signals, 
            tp_pct=tp_pct/100.0, 
            sl_pct=sl_pct/100.0
        )

    # ------------------- Display Results -------------------
    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("Equity Curve")
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(x=equity.index, y=equity["equity"], 
                                  mode="lines", name="Equity", line=dict(color="#00ff88")))
        fig_eq.update_layout(height=400, template="plotly_dark")
        st.plotly_chart(fig_eq, use_container_width=True)

    with col2:
        st.subheader("Summary")
        final_equity = equity["equity"].iloc[-1]
        num_trades = (signals["signal"] == 1).sum()
        
        st.metric("Final Equity", f"{final_equity:.2f}x", 
                  delta=f"{(final_equity - 1)*100:+.1f}%")
        st.metric("Number of Trades", int(num_trades))

    # Price chart with signals
    st.subheader("Price Chart with Buy Signals")
    fig_price = go.Figure()

    fig_price.add_trace(go.Candlestick(
        x=df.index[-500:],  # Show last 500 candles for clarity
        open=df["open"].iloc[-500:],
        high=df["high"].iloc[-500:],
        low=df["low"].iloc[-500:],
        close=df["close"].iloc[-500:],
        name="BTC/USDT"
    ))

    buys = signals[signals["signal"] == 1].index
    if len(buys) > 0:
        buy_prices = df.loc[buys.intersection(df.index), "close"]
        fig_price.add_trace(go.Scatter(
            x=buy_prices.index,
            y=buy_prices,
            mode="markers",
            marker=dict(color="lime", size=9, symbol="triangle-up"),
            name="AI Buy"
        ))

    fig_price.update_layout(height=500, xaxis_rangeslider_visible=False, template="plotly_dark")
    st.plotly_chart(fig_price, use_container_width=True)


if __name__ == "__main__":
    main()