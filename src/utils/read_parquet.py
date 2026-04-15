#!/usr/bin/env python3
"""
Utility to inspect Binance-style raw trades Parquet file.
Handles corrupted/out-of-range timestamps gracefully.
"""

import argparse
import pyarrow.parquet as pq
import pandas as pd
from datetime import datetime, timezone


def ms_to_iso(ts_ms: int) -> str:
    """Safe conversion from milliseconds to ISO string."""
    try:
        if not (1_600_000_000_000 <= ts_ms <= 4_000_000_000_000):  # roughly 2020 to ~2096
            return f"INVALID_TS_{ts_ms}"
        dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
        return dt.isoformat()
    except Exception:
        return f"INVALID_TS_{ts_ms}"


def main():
    parser = argparse.ArgumentParser(description="Inspect a raw trades Parquet file")
    parser.add_argument("file", type=str, help="Path to the .parquet file")
    parser.add_argument("--max-rows", type=int, default=20, help="Maximum number of rows to show (default: 20)")
    parser.add_argument("--show-raw", action="store_true", help="Show raw pyarrow table")
    parser.add_argument("--skip-invalid", action="store_true", default=True,
                        help="Skip rows with obviously invalid timestamps")
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
        print("\n=== RAW TABLE (first 10 rows) ===")
        print(df.head(10))

    if "timestamp" not in df.columns:
        print("ERROR: 'timestamp' column not found!")
        return

    # Keep original as int64
    df["timestamp_ms"] = df["timestamp"].astype("int64")

    # Create microsecond version safely
    df["timestamp_us"] = df["timestamp_ms"] * 1000

    # Safe ISO conversion
    df["timestamp_iso"] = df["timestamp_ms"].apply(ms_to_iso)

    # Optional: filter obviously bad timestamps
    if args.skip_invalid:
        valid_mask = df["timestamp_ms"].between(1_600_000_000_000, 4_000_000_000_000)
        bad_count = len(df) - valid_mask.sum()
        if bad_count > 0:
            print(f"[warning] Found {bad_count:,} rows with invalid timestamps → skipped for display")
            df = df[valid_mask].reset_index(drop=True)

    # Reorder columns
    cols = ["timestamp_us", "timestamp_ms", "timestamp_iso", "price", "quantity", "is_buyer_maker"]
    df = df[[c for c in cols if c in df.columns]]

    print(f"\n=== STREAMING ROWS ({datetime.now(timezone.utc).isoformat()}) ===")

    shown = 0
    for i, row in enumerate(df.itertuples(index=False, name=None)):
        if shown >= args.max_rows:
            break
        row_dict = dict(zip(df.columns, row))
        print(row_dict)
        shown += 1

    if shown < len(df):
        print(f"[info] Reached max_rows={args.max_rows} (shown {shown:,} rows)")

    print(f"\nTotal rows in file: {len(df):,}")


if __name__ == "__main__":
    main()