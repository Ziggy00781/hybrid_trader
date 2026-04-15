#!/usr/bin/env python3
"""
Clean and nice reader for Binance raw trades Parquet files.
Shows human-readable timestamps.
"""

import argparse
import pyarrow.parquet as pq
import pandas as pd
from datetime import datetime, timezone


def main():
    parser = argparse.ArgumentParser(description="View Binance raw trades Parquet file")
    parser.add_argument("file", type=str, help="Path to the .parquet file")
    parser.add_argument("--max-rows", type=int, default=20, help="Number of rows to display (default: 20)")
    parser.add_argument("--show-raw", action="store_true", help="Also show raw table head")
    args = parser.parse_args()

    path = args.file
    print("=== FILE INFO ===")
    print(f"Path: {path}")

    pf = pq.ParquetFile(path)
    print(f"Row groups: {pf.num_row_groups}")
    print("Schema:")
    print(pf.schema)

    # Read data
    table = pf.read()
    df = table.to_pandas()

    if args.show_raw:
        print("\n=== RAW DATA (first 5 rows) ===")
        print(df.head())

    # === CORRECT CONVERSION (this was the missing piece) ===
    df["timestamp_ms"] = df["timestamp"].astype("int64")
    df["timestamp_us"] = df["timestamp_ms"] * 1000                     # for internal use if needed
    df["timestamp_iso"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)

    # Nice column order
    cols = ["timestamp_iso", "price", "quantity", "is_buyer_maker"]
    df = df[cols]

    print(f"\n=== FIRST {args.max_rows} TRADES ({datetime.now(timezone.utc).isoformat()}) ===")

    for _, row in df.head(args.max_rows).iterrows():
        print({
            "timestamp_iso": row["timestamp_iso"].strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + " UTC",
            "price": f"{row['price']:.2f}",
            "quantity": f"{row['quantity']:.8f}",
            "is_buyer_maker": row["is_buyer_maker"]
        })

    print(f"\nTotal trades in file: {len(df):,}")


if __name__ == "__main__":
    main()