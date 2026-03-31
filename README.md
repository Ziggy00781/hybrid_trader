# Hybrid Trader

**AI-Powered Hybrid Cryptocurrency Trading System for BTC/USDT (5-minute timeframe)**

A modular, production-ready framework that combines **classical machine learning**, **transformer-based deep learning**, and **foundation model forecasts** to generate trading signals, backtest strategies, and run live automated trading on Bybit.

![Price vs Prediction](price_vs_prediction2.png)
![Predictions](predictions.png)

## ✨ Features

- **Multi-exchange data pipeline** — Binance + Bybit OHLCV fetching, merging, and resampling
- **Rich feature engineering** — Technical indicators, volume features, market regime detection (trending / ranging), and foundation model predictions as features
- **Hybrid modeling**:
  - LightGBM (fast, interpretable, used in dashboard & backtesting)
  - PatchTST (Transformer with patching — state-of-the-art for time series)
  - Support for foundation models (TimeFM, Chronos, TimeGPT) as features or zero-shot forecasters
- **Backtesting engine** with realistic TP/SL and equity curve visualization
- **Live trading loop** with Bybit API integration, normalization, and signal generation
- **Interactive Streamlit dashboard** for signal visualization and strategy testing
- GPU-ready training scripts

## 🏗️ Architecture

```mermaid
flowchart TD
    A[Raw Data\nBinance + Bybit] --> B[Data Merge & Resample]
    B --> C[Feature Engineering\nTA + Regime + FM Forecasts]
    C --> D1[LightGBM Model\nDashboard & Backtest]
    C --> D2[PatchTST Transformer\nLive Trading]
    C --> D3[Foundation Models\nTimeFM / Chronos / TimeGPT]
    D1 & D2 --> E[Signal Generation\nLong / Short / Flat]
    E --> F[Backtester]
    E --> G[Live Trading Loop\nBybit]
    F & G --> H[Performance Visualization\nEquity Curve + Trades]