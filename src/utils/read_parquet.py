#!/usr/bin/env python3
"""
Utility to inspect Binance-style raw trades Parquet file.
"""

import argparse
import pyarrow.parquet as pq
import pandas as pd
from datetime import datetime, timezone


def main():
    parser = argparse.ArgumentParser(description="Inspect a raw trades Parquet file")
    parser.add_argument("file", type=str, help="Path to the .parquet file")
    parser.add_argument("--max-rows", type=int, default=20, help="Maximum number of rows to show (default: 20)")
    parser.add_argument("--show-raw", action="store_true", help="Show raw pyarrow table")
    args = parser.parse_args()

    path = args.file
    print("=== FILE INFO ===")
    print(f"Path: {path}")

    parquet_file = pq.ParquetFile(path)
    print(f"Row groups: {parquet_file.num_row_groups}")
    print("Schema:")
    print(parquet_file.schema)

    # Read the table
    table = parquet_file.read()
    df = table.to_pandas()

    if args.show_raw:
        print("\n=== RAW TABLE (first 5 rows) ===")
        print(df.head())

    # === FIXED CONVERSION LOGIC ===
    if "timestamp" in df.columns:
        # Keep original as int64 milliseconds (critical fix)
        df["timestamp_ms"] = df["timestamp"].astype("int64")

        # Create microsecond version
        df["timestamp_us"] = df["timestamp_ms"] * 1000

        # Create nice ISO string
        df["timestamp_iso"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    else:
        print("ERROR: 'timestamp' column not found!")
        return

    # Reorder columns for clean output
    cols = ["timestamp_us", "timestamp_ms", "timestamp_iso", "price", "quantity", "is_buyer_maker"]
    df = df[[c for c in cols if c in df.columns]]

    print(f"\n=== STREAMING ROWS ({datetime.now(timezone.utc).isoformat()}) ===")

    for i, row in enumerate(df.itertuples(index=False, name=None)):
        if i >= args.max_rows:
            print(f"[info] Reached max_rows={args.max_rows}")
            break
        print(dict(zip(df.columns, row)))

    print(f"\nTotal rows in file: {len(df):,}")


if __name__ == "__main__":
    main()