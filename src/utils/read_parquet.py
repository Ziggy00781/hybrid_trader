#!/usr/bin/env python

"""
read_parquet.py

Correct Parquet inspector for hybrid_trader:
- Your timestamp column *claims* timestamp(ms)
- But actual stored values are int64 microseconds
- This script reads raw int64, converts safely, and avoids OverflowError

Usage:
    python -m src.utils.read_parquet file.parquet --max-rows 20 --show-raw-us
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.parquet as pq


# Python datetime max: year 9999 → 253402300799000 ms
MAX_TS_MS = 253402300799000
MIN_TS_MS = 0


def parse_args():
    parser = argparse.ArgumentParser(description="Safe Parquet reader for hybrid_trader.")
    parser.add_argument("path", type=str, help="Path to Parquet file")
    parser.add_argument("--max-rows", type=int, default=None, help="Limit printed rows")
    parser.add_argument("--batch-size", type=int, default=5000, help="Batch size")
    parser.add_argument("--show-raw-us", action="store_true",
                        help="Show raw timestamp_us values")
    return parser.parse_args()


def print_file_info(path, pf):
    print("\n=== FILE INFO ===")
    print(f"Path: {path}")
    print(f"Row groups: {pf.num_row_groups}")
    print("Schema:")
    print(pf.schema)


def valid_ms(ts_ms):
    return ts_ms is not None and MIN_TS_MS < ts_ms < MAX_TS_MS


def ms_to_iso(ts_ms):
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat()


def stream_rows(pf, max_rows, batch_size, show_raw_us):
    print("\n=== STREAMING ROWS ===")

    printed = 0
    schema = pf.schema
    col_names = schema.names
    has_ts = "timestamp" in col_names

    for batch in pf.iter_batches(batch_size=batch_size):
        cols = {name: batch.column(name) for name in col_names}

        # Extract raw microseconds (actual physical data)
        if has_ts:
            ts_us_list = cols["timestamp"].cast(pa.int64()).to_pylist()
        else:
            ts_us_list = None

        for i in range(batch.num_rows):

            # Handle timestamp
            if has_ts:
                ts_us = ts_us_list[i]
                if ts_us is None:
                    continue

                ts_ms = ts_us // 1000  # convert microseconds → milliseconds

                if not valid_ms(ts_ms):
                    continue
            else:
                ts_us = None
                ts_ms = None

            # Build row dict
            row = {}

            if has_ts:
                if show_raw_us:
                    row["timestamp_us"] = ts_us
                row["timestamp_ms"] = ts_ms
                row["timestamp_iso"] = ms_to_iso(ts_ms)

            # Other columns
            for name, col in cols.items():
                if name == "timestamp":
                    continue
                row[name] = col[i].as_py()

            print(row)
            printed += 1

            if max_rows is not None and printed >= max_rows:
                print(f"\n[info] Reached max_rows={max_rows}")
                return

    if printed == 0:
        print("[info] No valid rows found.")


def main():
    args = parse_args()
    path = Path(args.path)

    if not path.exists():
        print(f"[error] File not found: {path}", file=sys.stderr)
        sys.exit(1)

    pf = pq.ParquetFile(path)

    print_file_info(path, pf)
    stream_rows(pf, args.max_rows, args.batch_size, args.show_raw_us)


if __name__ == "__main__":
    main()