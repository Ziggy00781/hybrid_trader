import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import joblib

from src.features.ta_regime_features import build_features
from src.backtest.backtester import generate_signals, backtest

RAW_PATH = Path("data/raw/binance_btcusdt_5m.parquet")
MODEL_PATH = Path("models/btcusdt_5m_lgbm.pkl")

@st.cache_data
def load_data():
    df = pd.read_parquet(RAW_PATH)
    return df

@st.cache_data
def load_model():
    return joblib.load(MODEL_PATH)

def main():
    st.title("BTCUSDT 5m – Hybrid AI Backtest")

    df = load_data()
    model = load_model()

    st.sidebar.header("Parameters")
    prob_threshold = st.sidebar.slider("Prob. threshold", 0.5, 0.9, 0.6, 0.01)
    tp_pct = st.sidebar.slider("Take profit %", 0.1, 2.0, 0.4, 0.1) / 100.0
    sl_pct = st.sidebar.slider("Stop loss %", 0.1, 2.0, 0.4, 0.1) / 100.0

    # Features + signals
    feats = build_features(df)
    df_aligned = df.loc[feats.index]
    signals = generate_signals(df_aligned, prob_threshold=prob_threshold, model=model)

    # Backtest
    equity = backtest(df_aligned, signals, tp_pct=tp_pct, sl_pct=sl_pct)

    st.subheader("Equity curve")
    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(
        x=equity.index,
        y=equity["equity"],
        mode="lines",
        name="Equity"
    ))
    st.plotly_chart(fig_eq, use_container_width=True)

    st.subheader("Price with entry signals")
    fig_price = go.Figure()
    fig_price.add_trace(go.Candlestick(
        x=df_aligned.index,
        open=df_aligned["open"],
        high=df_aligned["high"],
        low=df_aligned["low"],
        close=df_aligned["close"],
        name="Price"
    ))

    buys = signals[signals["signal"] == 1]
    fig_price.add_trace(go.Scatter(
        x=buys.index,
        y=df_aligned.loc[buys.index, "close"],
        mode="markers",
        marker=dict(color="green", size=6),
        name="AI BUY"
    ))

    fig_price.update_layout(xaxis_rangeslider_visible=False)
    st.plotly_chart(fig_price, use_container_width=True)

    st.subheader("Summary")
    final_equity = equity["equity"].iloc[-1]
    st.write(f"**Final equity:** {final_equity:.3f}x initial capital")
    st.write(f"Number of entries: {int((signals['signal'] == 1).sum())}")

if __name__ == "__main__":
    main()