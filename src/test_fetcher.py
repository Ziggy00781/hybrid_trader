from src.data_fetch.unified_fetcher import UnifiedDataFetcher
import pandas as pd

fetcher = UnifiedDataFetcher()  # No keys → works for crypto right away

# Test 1: Your original crypto style (CCXT)
print("Fetching BTC/USDT via CCXT...")
df_ccxt = fetcher.fetch_data("BTC/USDT", timeframe="5m", limit=2000)
print(df_ccxt.tail(5))
print(f"Shape: {df_ccxt.shape}\n")

# Test 2: Same crypto via Alpaca (BTC/USD)
print("Fetching BTC/USD via Alpaca...")
df_alpaca = fetcher.fetch_data("BTC/USD", timeframe="5m", source="alpaca")
print(df_alpaca.tail(5))
print(f"Shape: {df_alpaca.shape}\n")

# Save one for training
df_ccxt.to_parquet("data/raw/BTC_5m_ccxt.parquet")
print("Saved to data/raw/BTC_5m_ccxt.parquet — ready for PatchTST training!")