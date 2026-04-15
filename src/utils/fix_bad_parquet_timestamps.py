#!/usr/bin/env python3
"""
Fix bad timestamp Parquet files (final robust version).
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

    # Read original
    table = pq.read_table(input_path)
    df = table.to_pandas()

    if "timestamp" not in df.columns:
        print("  No timestamp column - skipping")
        return False

    ts_raw = df["timestamp"].astype("int64")

    # Fix the scaling (your case: divided by 1_000_000)
    if ts_raw.max() > 4_000_000_000_000:
        print(f"  Detected overscaled timestamps (max={ts_raw.max():,}) → dividing by 1_000_000")
        ts_correct_ms = (ts_raw // 1_000_000).astype("int64")
    else:
        print("  Timestamps look reasonable")
        ts_correct_ms = ts_raw

    # Rebuild DataFrame with correct column order
    clean_df = pd.DataFrame({
        "timestamp": ts_correct_ms,
        "price": df["price"].astype("float32"),
        "quantity": df["quantity"].astype("float32"),
        "is_buyer_maker": df["is_buyer_maker"].astype("bool")
    })

    # Create Arrow Table + explicit schema
    clean_table = pa.Table.from_pandas(clean_df)

    schema = pa.schema([
        ("timestamp", pa.timestamp("ms")),
        ("price", pa.float32()),
        ("quantity", pa.float32()),
        ("is_buyer_maker", pa.bool_())
    ])

    clean_table = clean_table.cast(schema)

    if dry_run:
        print("  [DRY RUN] Would write corrected file")
        # Show a sample of corrected timestamps
        print(f"  Sample corrected timestamp_ms: {ts_correct_ms.iloc[0]:,}")
        return True

    # Backup original
    backup_path = f"{input_path}.bak"
    if not os.path.exists(backup_path):
        os.rename(input_path, backup_path)
        print(f"  Backed up original → {backup_path}")

    # Write clean file
    pq.write_table(clean_table, output_path, compression="snappy")

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Fixed successfully ({size_mb:.1f} MB)")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Parquet file or directory")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-files", type=int, default=None)
    args = parser.parse_args()

    path = Path(args.path)
    if path.is_file() and path.suffix == ".parquet":
        files = [path]
    else:
        files = sorted(path.rglob("*.parquet"))

    if args.max_files:
        files = files[:args.max_files]

    print(f"Found {len(files)} parquet files...\n")

    fixed = 0
    for i, f in enumerate(files, 1):
        print(f"[{i}/{len(files)}]")
        try:
            if fix_parquet_file(str(f), dry_run=args.dry_run):
                fixed += 1
        except Exception as e:
            print(f"  Error: {e}")

    print(f"\nDone! Fixed {fixed} files.")


if __name__ == "__main__":
    main()