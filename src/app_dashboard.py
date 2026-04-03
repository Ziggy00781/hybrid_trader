import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import joblib
import torch
import sys

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.features.ta_regime_features import build_mathematical_features
from src.backtest.backtester import generate_signals
from src.inference import load_best_model, predict_signal   # Your inference module
from src.backtest.patchtst_backtester import run_patchtst_backtest   # Your backtester

st.set_page_config(page_title="Hybrid Trader Dashboard", layout="wide")

# ====================== SIDEBAR ======================
st.sidebar.title("Hybrid Trader Control")
model_choice = st.sidebar.radio("Select Model", ["LightGBM", "PatchTST"])

# ====================== MAIN APP ======================
st.title("🚀 Hybrid Trader Dashboard")
st.markdown("**Real 5m BTC/USDT Analysis | Educational Project**")

tab1, tab2, tab3 = st.tabs(["LightGBM Dashboard", "PatchTST Backtesting", "Live Analysis"])

# ====================== TAB 1: LightGBM (Existing) ======================
with tab1:
    st.subheader("LightGBM Dashboard & Backtest")
    # Your existing LightGBM code goes here (or keep your current logic)
    RAW_PATH = Path("data/raw/binance_btcusdt_5m.parquet")
    MODEL_PATH = Path("models/btcusdt_5m_lgbm.pkl")
    
    if RAW_PATH.exists() and MODEL_PATH.exists():
        df = pd.read_parquet(RAW_PATH)
        model = joblib.load(MODEL_PATH)
        st.success(f"✅ Loaded {len(df):,} candles + LightGBM model")
        # ... you can keep your original LightGBM backtest code here if you want ...
    else:
        st.warning("LightGBM data or model not found.")

# ====================== TAB 2: PatchTST Backtesting ======================
with tab2:
    st.subheader("PatchTST Backtesting")
    if st.button("Run PatchTST Backtest (Last 3000 candles)"):
        with st.spinner("Running walk-forward backtest..."):
            df = pd.read_parquet("data/raw/binance_btcusdt_5m.parquet")
            equity_curve, trades = run_patchtst_backtest(df, min_confidence=60.0)
            st.success("Backtest completed!")

# ====================== TAB 3: Live Analysis ======================
with tab3:
    st.subheader("🔴 Live Analysis - PatchTST Real-Time Signal")
    st.info("This uses the latest trained PatchTST model + mathematical features")

    if st.button("Generate Current Live Signal"):
        with st.spinner("Fetching latest data and running inference..."):
            df = pd.read_parquet("data/raw/binance_btcusdt_5m.parquet")
            recent_df = df.tail(1500)   # last ~5 days

            model = load_best_model()
            result = predict_signal(recent_df, model)

            # Display nice card
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Signal", result['signal'], delta=None)
            with col2:
                st.metric("Confidence", f"{result['confidence']}%")
            with col3:
                st.metric("Estimated Move", f"{result['estimated_return_pct']}%")

            st.write(f"**Current Price**: ${result['current_price']}")
            if result.get('tp_price'):
                st.write(f"**Suggested Take Profit**: ${result['tp_price']}")
            if result.get('sl_price'):
                st.write(f"**Suggested Stop Loss**: ${result['sl_price']}")
            st.write(f"**Regime**: {result.get('regime', 'N/A')}")
            st.caption(f"Generated at {result['timestamp']}")

# Footer
st.caption("Educational Project • PatchTST + Mathematical Features • Learning Focus")