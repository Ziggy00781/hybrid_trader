#!/usr/bin/env python
"""
read_parquet.py

Operator-grade Parquet inspector/streamer with:
- File info (path, row groups, schema)
- Streaming iteration over batches
- Robust handling of corrupted timestamps (overflow-safe)
- Optional row limit for quick peeks

Usage:
    python -m src.utils.read_parquet <file.parquet> [--max-rows 100]
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


# Timestamp is stored as ms since epoch; Python datetime supports up to year 9999.
# 9999-12-31T23:59:59.999 UTC in ms since epoch:
MAX_TS_MS = 253402300799000  # conservative upper bound
MIN_TS_MS = 0  # reject negative / zero timestamps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect and stream a Parquet file safely.")
    parser.add_argument(
        "path",
        type=str,
        help="Path to the Parquet file",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Maximum number of rows to print (across all batches).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10_000,
        help="Batch size for streaming iteration.",
    )
    return parser.parse_args()


def print_file_info(path: Path, pf: pq.ParquetFile) -> None:
    print("\n=== FILE INFO ===")
    print(f"Path: {path}")
    print(f"Row groups: {pf.num_row_groups}")
    print("Schema:")
    print(pf.schema)


def ts_array_to_int_ms(col: pa.ChunkedArray) -> np.ndarray:
    """
    Convert a timestamp(ms) column to int64 milliseconds without going through Python datetime.

    Returns:
        np.ndarray[int64] of ms since epoch; NaT becomes np.iinfo(np.int64).min
    """
    # to_numpy() gives datetime64[ms]; view as int64 to get raw ms
    dt64 = col.to_numpy(zero_copy_only=False)
    int_ms = dt64.view("int64")
    return int_ms


def is_valid_timestamp_ms(ts_ms: int) -> bool:
    return (ts_ms is not None) and (MIN_TS_MS < ts_ms < MAX_TS_MS)


def ms_to_iso_utc(ts_ms: int) -> str:
    """
    Convert ms since epoch to ISO 8601 UTC string, assuming value is already validated.
    """
    dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
    return dt.isoformat()


def stream_rows(pf: pq.ParquetFile, max_rows: int | None, batch_size: int) -> None:
    print("\n=== STREAMING ROWS ===")

    schema = pf.schema
    col_names = schema.names

    has_timestamp = "timestamp" in col_names

    printed = 0

    for batch in pf.iter_batches(batch_size=batch_size):
        # Work with Arrow arrays directly; avoid batch.to_pydict() to prevent
        # automatic conversion of timestamps to Python datetime.
        columns = {name: batch.column(name) for name in col_names}

        # Precompute timestamp mask if present
        if has_timestamp:
            ts_col = columns["timestamp"]
            ts_ms = ts_array_to_int_ms(ts_col)

            # Mask of valid timestamps
            valid_mask = (ts_ms > MIN_TS_MS) & (ts_ms < MAX_TS_MS)
        else:
            # If no timestamp column, treat all rows as valid
            valid_mask = np.ones(batch.num_rows, dtype=bool)
            ts_ms = None  # type: ignore[assignment]

        # Iterate row-wise within the batch
        for i in range(batch.num_rows):
            if not valid_mask[i]:
                # Skip corrupted timestamp rows
                continue

            row_dict: dict[str, object] = {}

            for name, col in columns.items():
                if name == "timestamp" and has_timestamp:
                    # Use validated int ms and convert to ISO string
                    ts_val_ms = int(ts_ms[i])
                    row_dict["timestamp_ms"] = ts_val_ms
                    row_dict["timestamp_iso"] = ms_to_iso_utc(ts_val_ms)
                else:
                    # Safe conversion for non-timestamp columns
                    scalar = col[i]
                    row_dict[name] = scalar.as_py()

            print(row_dict)
            printed += 1

            if max_rows is not None and printed >= max_rows:
                print(f"\n[info] Reached max_rows={max_rows}, stopping.")
                return

    if printed == 0:
        print("[info] No valid rows found (all timestamps invalid or file empty).")


def main() -> None:
    args = parse_args()
    path = Path(args.path)

    if not path.exists():
        print(f"[error] File not found: {path}", file=sys.stderr)
        sys.exit(1)

    try:
        pf = pq.ParquetFile(path)
    except Exception as e:
        print(f"[error] Failed to open Parquet file: {e}", file=sys.stderr)
        sys.exit(1)

    print_file_info(path, pf)
    stream_rows(pf, max_rows=args.max_rows, batch_size=args.batch_size)


if __name__ == "__main__":
    main()