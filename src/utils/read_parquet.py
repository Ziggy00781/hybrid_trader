#!/usr/bin/env python3
"""
Diagnostic reader for corrupted timestamp Parquet files.
"""

import argparse
import pyarrow.parquet as pq
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Parquet file to inspect")
    parser.add_argument("--max-rows", type=int, default=30)
    parser.add_argument("--show-raw", action="store_true")
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
        print("\n=== RAW FIRST 10 ROWS ===")
        print(df.head(10))

    if "timestamp" not in df.columns:
        print("ERROR: No timestamp column!")
        return

    ts = df["timestamp"].astype("int64")

    print("\n=== TIMESTAMP STATISTICS (raw values) ===")
    print(f"Min timestamp : {ts.min():,}")
    print(f"Max timestamp : {ts.max():,}")
    print(f"Mean          : {ts.mean():,.0f}")
    print(f"Unique values : {ts.nunique():,}")

    # Try to guess the correct unit
    print("\n=== POSSIBLE INTERPRETATIONS ===")
    for unit, factor in [("milliseconds", 1),
                         ("microseconds", 1000),
                         ("nanoseconds", 1_000_000)]:
        ts_corrected = ts // factor
        min_year = pd.to_datetime(ts_corrected.min(), unit='ms', errors='coerce').year
        max_year = pd.to_datetime(ts_corrected.max(), unit='ms', errors='coerce').year
        print(f"{unit:12} → min year ~{min_year} | max year ~{max_year}")

    # Best guess reader (tries microseconds first, common for trade data)
    print("\n=== BEST GUESS OUTPUT (assuming microseconds) ===")
    df["timestamp_us"] = ts
    df["timestamp_ms"] = ts // 1000
    df["timestamp_iso"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True, errors="coerce")

    cols = ["timestamp_us", "timestamp_ms", "timestamp_iso", "price", "quantity", "is_buyer_maker"]
    df_out = df[cols].head(args.max_rows)

    for _, row in df_out.iterrows():
        print(row.to_dict())

    print(f"\nTotal rows: {len(df):,}")


if __name__ == "__main__":
    main()