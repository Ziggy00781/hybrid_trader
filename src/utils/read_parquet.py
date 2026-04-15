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
    parser.add_argument(
        "--show-raw-ts",
        action="store_true",
        help="Show raw timestamp_ms instead of ISO string.",
    )
    return parser.parse_args()


def print_file_info(path: Path, pf: pq.ParquetFile) -> None:
    print("\n=== FILE INFO ===")
    print(f"Path: {path}")
    print(f"Row groups: {pf.num_row_groups}")
    print("Schema:")
    print(pf.schema)


def is_valid_timestamp_ms(ts_ms: int | None) -> bool:
    if ts_ms is None:
        return False
    return MIN_TS_MS < ts_ms < MAX_TS_MS


def ms_to_iso_utc(ts_ms: int) -> str:
    """
    Convert ms since epoch to ISO 8601 UTC string, assuming value is already validated.
    """
    dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
    return dt.isoformat()


def stream_rows(
    pf: pq.ParquetFile,
    max_rows: int | None,
    batch_size: int,
    show_raw_ts: bool,
) -> None:
    print("\n=== STREAMING ROWS ===")

    schema = pf.schema
    col_names = schema.names

    has_timestamp = "timestamp" in col_names
    printed = 0

    for batch in pf.iter_batches(batch_size=batch_size):
        columns = {name: batch.column(name) for name in col_names}

        # Prepare timestamp values as raw int64 ms (or None) if present
        if has_timestamp:
            ts_col = columns["timestamp"]

            # Cast timestamp(ms) -> int64 ms inside Arrow (no Python datetime)
            ts_ms_list = ts_col.cast(pa.int64()).to_pylist()
        else:
            ts_ms_list = None  # type: ignore[assignment]

        num_rows = batch.num_rows

        for i in range(num_rows):
            # Validate timestamp if present
            if has_timestamp:
                ts_ms = ts_ms_list[i]
                if not is_valid_timestamp_ms(ts_ms):
                    # Skip corrupted timestamp rows
                    continue
            else:
                ts_ms = None

            row_dict: dict[str, object] = {}

            for name, col in columns.items():
                if name == "timestamp" and has_timestamp:
                    # Use validated int ms and optionally convert to ISO
                    ts_val_ms = int(ts_ms)  # type: ignore[arg-type]
                    if show_raw_ts:
                        row_dict["timestamp_ms"] = ts_val_ms
                    else:
                        row_dict["timestamp_iso"] = ms_to_iso_utc(ts_val_ms)
                else:
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
    stream_rows(
        pf,
        max_rows=args.max_rows,
        batch_size=args.batch_size,
        show_raw_ts=args.show_raw_ts,
    )


if __name__ == "__main__":
    main()