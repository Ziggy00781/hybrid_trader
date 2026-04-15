#!/usr/bin/env python

import sys
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa

def main():
    if len(sys.argv) < 2:
        print("Usage: python raw_ts.py <file.parquet>")
        sys.exit(1)

    path = Path(sys.argv[1])
    pf = pq.ParquetFile(path)

    print("=== FIRST 20 RAW TIMESTAMP VALUES (int64 ms) ===")

    count = 0
    for batch in pf.iter_batches(batch_size=5000, columns=["timestamp"]):
        col = batch.column("timestamp")

        # Cast timestamp(ms) → int64 ms (NO datetime conversion)
        ts_list = col.cast(pa.int64()).to_pylist()

        for v in ts_list:
            print(v)
            count += 1
            if count >= 20:
                return

if __name__ == "__main__":
    main()
