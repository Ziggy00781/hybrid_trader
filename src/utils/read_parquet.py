#!/usr/bin/env python3
"""
Clean, readable viewer for Binance raw trades Parquet files.
"""

import argparse
import pyarrow.parquet as pq
import pandas as pd
from datetime import datetime, timezone


def main():
    parser = argparse.ArgumentParser(description="View Binance raw trades Parquet file")
    parser.add_argument("file", type=str, help="Path to the .parquet file")
    parser.add_argument("--max-rows", type=int, default=20, help="Number of rows to show")
    parser.add_argument("--show-raw", action="store_true", help="Show raw pandas head")
    args = parser.parse_args()

    print("=== FILE INFO ===")
    print(f"Path: {args.file}")

    pf = pq.ParquetFile(args.file)
    print(f"Row groups: {pf.num_row_groups}")
    print("Schema:")
    print(pf.schema)

    table = pf.read()
    df = table.to_pandas()

    if args.show_raw:
        print("\n=== RAW DATA (first 5 rows) ===")
        print(df.head())

    # === CORRECT CONVERSION ===
    df["timestamp_ms"] = df["timestamp"].astype("int64")
    df["timestamp_iso"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)

    # Format for nice display
    df_display = pd.DataFrame({
        "timestamp": df["timestamp_iso"].dt.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-3] + " UTC",
        "price": df["price"].round(2),
        "quantity": df["quantity"].round(8),
        "is_buyer_maker": df["is_buyer_maker"]
    })

    print(f"\n=== FIRST {args.max_rows} TRADES ({datetime.now(timezone.utc).isoformat()}) ===")

    for _, row in df_display.head(args.max_rows).iterrows():
        print(row.to_dict())

    print(f"\nTotal trades: {len(df):,}")


if __name__ == "__main__":
    main()