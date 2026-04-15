#!/usr/bin/env python3
"""
Fix bad timestamp Parquet files caused by the old downloader.
"""

import argparse
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
import os
from pathlib import Path


def fix_parquet_file(input_path: str, output_path: str = None, dry_run: bool = False):
    if output_path is None:
        output_path = input_path

    print(f"Processing: {input_path}")

    table = pq.read_table(input_path)
    df = table.to_pandas()

    if "timestamp" not in df.columns:
        print("  No timestamp column - skipping")
        return False

    ts_raw = df["timestamp"].astype("int64")

    # Detect and correct overscaled timestamps (the main bug)
    if ts_raw.max() > 4_000_000_000_000:   # clearly too big (year >> 2096)
        print(f"  Detected overscaled timestamps (max={ts_raw.max():,}) → dividing by 1_000_000")
        ts_correct_ms = (ts_raw // 1_000_000).astype("int64")
    else:
        print("  Timestamps look reasonable - no scaling applied")
        ts_correct_ms = ts_raw

    # Rebuild clean table with only the columns we need
    clean_df = df[["price", "quantity", "is_buyer_maker"]].copy()
    clean_df["timestamp"] = ts_correct_ms

    # Convert to Arrow Table with explicit millisecond timestamp type
    clean_table = pa.Table.from_pandas(clean_df)

    # Cast the timestamp column to proper logical type (this is the reliable way)
    timestamp_field = pa.field("timestamp", pa.timestamp("ms"))
    schema = pa.schema([
        timestamp_field,
        pa.field("price", pa.float32()),
        pa.field("quantity", pa.float32()),
        pa.field("is_buyer_maker", pa.bool_())
    ])
    clean_table = clean_table.cast(schema)

    if dry_run:
        print("  [DRY RUN] Would write corrected file")
        return True

    # Backup original (only once)
    backup_path = f"{input_path}.bak"
    if not os.path.exists(backup_path):
        os.rename(input_path, backup_path)
        print(f"  Backed up original → {backup_path}")

    # Write without passing 'schema=' to avoid the conflict
    pq.write_table(
        clean_table,
        output_path,
        compression="snappy",
        version="2.6"
    )

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Fixed successfully ({size_mb:.1f} MB)")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Parquet file or directory to fix")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without writing")
    parser.add_argument("--max-files", type=int, default=None, help="Process only first N files")
    args = parser.parse_args()

    path = Path(args.path)

    if path.is_file() and path.suffix == ".parquet":
        files = [path]
    else:
        files = sorted(path.rglob("*.parquet"))

    if args.max_files:
        files = files[:args.max_files]

    print(f"Found {len(files)} .parquet files to process...\n")

    fixed = 0
    for i, f in enumerate(files, 1):
        print(f"[{i}/{len(files)}]")
        try:
            if fix_parquet_file(str(f), dry_run=args.dry_run):
                fixed += 1
        except Exception as e:
            print(f"  Error: {e}")

    print(f"\nFinished! Successfully fixed {fixed} files.")


if __name__ == "__main__":
    main()