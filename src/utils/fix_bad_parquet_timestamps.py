#!/usr/bin/env python3
"""
Fix bad timestamp Parquet files caused by the old downloader.
Run this on your existing files.
"""

import argparse
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
import os
from pathlib import Path
from datetime import datetime


def fix_parquet_file(input_path: str, output_path: str = None, dry_run: bool = False):
    if output_path is None:
        output_path = input_path  # overwrite in place (after backup)

    print(f"Processing: {input_path}")

    table = pq.read_table(input_path)
    df = table.to_pandas()

    if "timestamp" not in df.columns:
        print("  No timestamp column - skipping")
        return False

    ts_raw = df["timestamp"].astype("int64")

    # Heuristic: if values are ridiculously large (year > 2100), they were probably stored as ns instead of ms
    # So divide by 1_000_000 to get back to ms
    if ts_raw.max() > 4_000_000_000_000:   # way beyond year ~2096
        print(f"  Detected overscaled timestamps (max={ts_raw.max():,}) → dividing by 1_000_000")
        ts_correct_ms = (ts_raw // 1_000_000).astype("int64")
    else:
        ts_correct_ms = ts_raw

    # Build clean table
    clean_table = pa.Table.from_pandas(df.assign(
        timestamp=ts_correct_ms
    )[["timestamp", "price", "quantity", "is_buyer_maker"]])

    # Write with explicit millisecond timestamp type
    schema = pa.schema([
        ("timestamp", pa.timestamp("ms")),
        ("price", pa.float32()),
        ("quantity", pa.float32()),
        ("is_buyer_maker", pa.bool_())
    ])

    if dry_run:
        print("  Dry-run: would write corrected file")
        return True

    # Backup original
    backup_path = f"{input_path}.bak"
    if not os.path.exists(backup_path):
        os.rename(input_path, backup_path)
        print(f"  Backed up original to {backup_path}")

    pq.write_table(
        clean_table,
        output_path,
        compression="snappy",
        schema=schema,
        version="2.6"   # modern Parquet
    )

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Fixed and saved ({size_mb:.1f} MB)")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="File or directory containing bad .parquet files")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually write files")
    parser.add_argument("--max-files", type=int, default=None, help="Limit number of files to process")
    args = parser.parse_args()

    path = Path(args.path)

    if path.is_file():
        files = [path]
    else:
        files = list(path.rglob("*.parquet"))

    files = sorted(files)
    if args.max_files:
        files = files[:args.max_files]

    print(f"Found {len(files)} parquet files to process...")

    fixed = 0
    for i, f in enumerate(files, 1):
        print(f"[{i}/{len(files)}]")
        try:
            if fix_parquet_file(str(f), dry_run=args.dry_run):
                fixed += 1
        except Exception as e:
            print(f"  Error processing {f}: {e}")

    print(f"\nDone! Fixed {fixed} files.")


if __name__ == "__main__":
    main()