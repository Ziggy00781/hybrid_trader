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
    %% Data Layer
    A["Multi-Asset Raw Data\n(Crypto: Binance/Bybit/CCXT • Stocks: yfinance/Polygon)\nOHLCV + Fundamentals + News Feeds"] 
        --> B["Data Ingestion & Preparation\n(Merge • Resample to User Timeframe\nGap Handling • Caching)"]

    %% Feature & Analysis Engine
    B --> C["Advanced Feature Engineering\nTA Indicators + Volume + Market Context"]
    C --> D["Mathematical Analysis Engine"]
    D --> D1["Fourier Transform\n(Cycle Detection & Periodic Structures)"]
    D --> D2["Markov Chain / HMM\n(Regime Detection: Trending / Ranging / Volatile)"]
    D --> D3["Stochastic Analysis\n(Probabilistic Patterns & Randomness)"]
    
    %% Time-Series Prediction
    C --> E["Time-Series Prediction Models\n(PatchTST + LightGBM Ensemble + Foundation Models)\nProbabilistic Forecasts (up to 20 bars)"]

    %% LLM Research Layer
    A --> F["Real-Time & Historical Research\n(News API + Market Narratives)"]
    F --> G["LLM Integration Layer\n(Sentiment • Narrative Extraction • Macro/Geopolitical Context)\n(Ollama / Groq / OpenAI)"]

    %% Fusion & Decision Support
    D1 & D2 & D3 & E & G --> H["Unified Insight & Decision Support Engine\n(Fusion of Quant + LLM Outputs)"]
    H --> I["Actionable Outputs\n• Probabilistic Direction & Magnitude\n• Scenario Outlooks (Bullish/Bearish/Neutral)\n• Explainable 'Why' + Strategy Suggestions"]

    %% Execution & Visualization
    I --> J["Backtesting Engine\n(Multi-Asset • Realistic Slippage/Fees • Walk-Forward)"]
    I --> K["Live/Paper Trading Loop\n(Broker-Agnostic via CCXT/Alpaca/IBKR)"]
    
    J & K --> L["Interactive Streamlit Dashboard\n(Asset Selector • Dynamic Charts • Real-Time Insights\nRegime/Cycle Overlays • Sentiment Gauges)"]

    %% Infrastructure
    subgraph "Infrastructure & Config"
        M["Config Management\n(config.yaml: Assets, Models, Risk Params, LLM Provider)"]
        N["Caching & Storage\n(Parquet/ArcticDB)"]
        O["GPU/Compute Support\n(Training & Inference)"]
        P["Scheduling & Alerts\n(Celery/Airflow • Telegram/Email)"]
    end

    B & E & G & H -.-> M
    J & K -.-> O
    L -.-> P

    classDef data fill:#E3F2FD,stroke:#1976D2,stroke-width:2px
    classDef analysis fill:#E8F5E9,stroke:#388E3C,stroke-width:2px
    classDef llm fill:#F3E5F5,stroke:#7B1FA2,stroke-width:2px
    classDef decision fill:#FFF3E0,stroke:#F57C00,stroke-width:3px
    classDef execution fill:#FCE4EC,stroke:#C2185B,stroke-width:2px
    classDef ui fill:#E0F2F1,stroke:#00796B,stroke-width:2px

    class A,B data
    class C,D,D1,D2,D3,E analysis
    class F,G llm
    class H,I decision
    class J,K execution
    class L ui
```
    
Core Idea: Use PatchTST for the live forecasting engine and LightGBM for quick iteration and explainability. Foundation models enrich features or serve as experimental baselines.
📁 Project Structure

- 📁 hybrid_trader
  - 📄 .gitignore
  - 📄 bootstrap_gpu.sh
  - 📄 filetree.txt
  - 📄 launch_patchtst.sh
  - 📄 predictions.png
  - 📄 prepare_data.sh
  - 📄 price_vs_prediction2.png
  - 📄 qwen.html
  - 📄 README.md
  - 📄 requirements.txt
  - 📁 src
  - 📄 app_dashboard.py
  - 📄 test.py
  - 📁 analysis
    - 📄 visualize_predictions.py
    - 📄 visualize_price_vs_prediction.py
  - 📁 backtest
    - 📄 backtester.py
  - 📁 data_fetch
    - 📄 binance_ohlcv.py
    - 📄 build_data.py
    - 📄 bybit_ohlcv.py
    - 📄 enhanced_data_collector.py
  - 📁 features
    - 📄 build_features.py
    - 📄 ta_regime_features.py
  - 📁 live
    - 📄 api.py
    - 📄 live_loop.py
    - 📄 OLDruntime.py
    - 📄 runtime.py
  - 📁 train
    - 📄 train_enhanced_patchtst.py
    - 📄 train_model.py
    - 📄 train_patchtst.py
  - 📁 utils
- 📄 binance_archive_fetch.py
- 📄 device.py
- 📄 parquet_merge.py
- 📄 patchtst_dataset.py
- 📄 prepare_patchtst_dataset.py
- 📄 resample_5m.py

🚀 Quick Start
1. Clone the repository
Bashgit clone https://github.com/Ziggy00781/hybrid_trader.git
cd hybrid_trader
2. Install dependencies
```bash pip install -r requirements.txt ```
3. (Optional) GPU Environment Setup
```bash bootstrap_gpu.sh ```
4. Prepare Data
```bash prepare_data.sh ```
This script handles fetching, merging, resampling, and feature engineering.
5. Train Models (if needed)
```bash
# Train PatchTST
bash launch_patchtst.sh
```

# Or run specific training scripts
```bash python -m src.train.train_patchtst
python -m src.train.train_model      # for LightGBM
```
6. Run the Dashboard
```bash 
python streamlit run src/app_dashboard.py
```
7. Run Live Trading (Paper or Real)
```bash
# Review and set your Bybit API keys in src/live/api.py or config
python -m src.live.runtime
# or use live_loop.py for continuous operation
```
⚠️ Warning: Live trading involves real financial risk. Start with paper trading and small position sizes.

### 📊 How It Works

Data → Multi-exchange 5m BTC/USDT candles
Features → Classical TA + regime detection + foundation model forecasts as extra signals
Models:
LightGBM: Predicts probability of upward move → filtered by regime
PatchTST: Predicts next 5m return directly → converted to trading signal

Signals → Threshold-based (e.g., strong long if predicted return > 0.05%)
Execution → Backtester or live loop with position management

### 🛠️ Configuration & Customization

Model paths and thresholds are currently in the respective scripts (future improvement: centralized config/ with YAML).
Feature sets can be extended in src/features/ta_regime_features.py.
Backtest parameters (TP/SL, thresholds) are adjustable in the Streamlit sidebar.

### 📈 Results & Visualization
The repository includes example charts:

price_vs_prediction.png — Actual price vs model predictions
predictions.png — Raw forecast visualization

### 📋 Roadmap

 Centralized configuration (config.yaml)
 Ensemble model (PatchTST + LightGBM)
 Advanced risk management & position sizing
 Telegram / Discord alerts
 Walk-forward optimization
 Docker + GitHub Actions CI/CD
 Multi-asset & multi-timeframe support
 Proper unit tests

### ⚠️ Disclaimer
* This project is for educational and research purposes only.
No guarantee of profitability. Past performance does not indicate future results.
Trading cryptocurrencies involves substantial risk of loss. Use at your own risk.
Always backtest thoroughly and start with simulated trading. *
### 🤝 Contributing
Contributions are welcome! Feel free to open issues or pull requests for:

Bug fixes
New features
Improved documentation
Better risk management

### 📄 License
This project is licensed under the MIT License — see the LICENSE file for details (add one if you want).

### Made with ❤️ in Hawaii by Ziad (Ziggy00781)
Questions or ideas? Open an issue or reach out!

### Donations accepted!
Bitcoin Address: bc1qgxhaqx96t2esdj73qdkx6un5wun48jxs62ndfu