import pandas as pd
from pathlib import Path

RAW = Path("data/raw/merged")
OUT = Path("data/raw/merged_resampled")
OUT.mkdir(parents=True, exist_ok=True)

def main():
    src = RAW / "btc_multi_exchange_5m.parquet"
    if not src.exists():
        raise FileNotFoundError(f"Missing merged file: {src}")

    print(f"Loading {src} ...")
    df = pd.read_parquet(src)

    # Ensure index is datetime
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.set_index("timestamp")
    else:
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")

    # Drop invalid timestamps
    df = df[~df.index.isna()]

    # Build continuous 5-minute UTC index
    full_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq="5min",
        tz="UTC"
    )

    print(f"Resampling to continuous 5-minute grid ({len(full_index)} rows)...")

    # Reindex without aggregating — preserves raw values
    df_resampled = df.reindex(full_index)

    out_path = OUT / "btc_multi_exchange_5m_resampled.parquet"
    df_resampled.to_parquet(out_path)

    print(f"✅ Saved resampled dataset to {out_path} with shape {df_resampled.shape}")

if __name__ == "__main__":
    main()