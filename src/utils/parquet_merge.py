import pandas as pd
from pathlib import Path

RAW = Path("data/raw")
OUT = RAW / "merged"
OUT.mkdir(parents=True, exist_ok=True)

def load_parquet(path, prefix):
    df = pd.read_parquet(path)

    # If timestamp is a column, convert and set as index
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.set_index("timestamp")
    else:
        # Convert index to datetime
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")

    # Drop rows where index is NaT
    df = df[~df.index.isna()]

    # Prefix columns
    df = df.add_prefix(f"{prefix}_")
    df.index.name = "timestamp"
    return df

def main():
    bybit_path   = RAW / "bybit"   / "bybit_btcusdt_5m.parquet"
    binance_path = RAW / "binance" / "binance_global_btcusdt_5m.parquet"

    dfs = []
    if bybit_path.exists():
        dfs.append(load_parquet(bybit_path, "bybit"))
    if binance_path.exists():
        dfs.append(load_parquet(binance_path, "binance"))

    # Outer join on timestamp
    df = pd.concat(dfs, axis=1).sort_index()

    out_path = OUT / "btc_multi_exchange_5m.parquet"
    df.to_parquet(out_path)
    print(f"✅ Saved merged dataset to {out_path} with shape {df.shape}")

if __name__ == "__main__":
    main()