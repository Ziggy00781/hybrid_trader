#!/usr/bin/env python3
"""
Simple utility to inspect a trades Parquet file (Binance-style raw trades).
"""

import argparse
import pyarrow.parquet as pq
import pyarrow as pa
from datetime import datetime, timezone


def main():
    parser = argparse.ArgumentParser(description="Inspect a raw trades Parquet file")
    parser.add_argument("file", type=str, help="Path to the .parquet file")
    parser.add_argument("--max-rows", type=int, default=20, help="Maximum number of rows to show (default: 20)")
    parser.add_argument("--show-raw", action="store_true", help="Also show the raw pyarrow table before conversion")
    args = parser.parse_args()

    path = args.file
    print("=== FILE INFO ===")
    print(f"Path: {path}")

    # Read metadata
    parquet_file = pq.ParquetFile(path)
    print(f"Row groups: {parquet_file.num_row_groups}")
    print("Schema:")
    print(parquet_file.schema)

    if args.show_raw:
        table = parquet_file.read()
        print("\n=== RAW TABLE (first 5 rows) ===")
        print(table.to_pandas().head())

    # Read the data
    table = parquet_file.read()

    # Convert to pandas for easy manipulation
    df = table.to_pandas()

    # === KEY FIXES START HERE ===
    # The original column is 'timestamp' stored as milliseconds (int64)
    if "timestamp" in df.columns:
        # Convert ms -> us (multiply by 1000)
        df["timestamp_us"] = df["timestamp"] * 1000

        # Also keep the original ms version for clarity
        df["timestamp_ms"] = df["timestamp"]

        # Create proper ISO datetime (timezone-aware)
        df["timestamp_iso"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    else:
        print("Warning: No 'timestamp' column found in the schema!")
        # Fallback: try to find any timestamp-like column
        for col in df.columns:
            if "time" in col.lower() or "ts" in col.lower():
                print(f"Found possible timestamp column: {col}")

    # Ensure other columns exist
    for col in ["price", "quantity", "is_buyer_maker"]:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found!")

    # Reorder columns nicely
    cols = ["timestamp_us", "timestamp_ms", "timestamp_iso", "price", "quantity", "is_buyer_maker"]
    existing_cols = [c for c in cols if c in df.columns]
    df = df[existing_cols]

    print(f"\n=== STREAMING ROWS ({datetime.now(timezone.utc).isoformat()}) ===")

    for i, row in enumerate(df.itertuples(index=False)):
        if i >= args.max_rows:
            print(f"[info] Reached max_rows={args.max_rows}")
            break

        row_dict = row._asdict()
        print(row_dict)

    print(f"\nTotal rows in file: {len(df):,}")


if __name__ == "__main__":
    # pandas is used only for to_datetime and itertuples – it's already in your env
    import pandas as pd
    main()