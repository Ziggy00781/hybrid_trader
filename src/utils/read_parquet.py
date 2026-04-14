import sys
from pathlib import Path
import pyarrow.parquet as pq

def main():
    if len(sys.argv) != 2:
        print("Usage: python read_parquet_stream.py <filename.parquet>")
        sys.exit(1)

    file_path = Path(sys.argv[1])

    if not file_path.exists():
        print(f"Error: File not found -> {file_path}")
        sys.exit(1)

    try:
        parquet_file = pq.ParquetFile(file_path)
    except Exception as e:
        print(f"Failed to open parquet file: {e}")
        sys.exit(1)

    print("\n=== FILE INFO ===")
    print(f"Path: {file_path.resolve()}")
    print(f"Row groups: {parquet_file.num_row_groups}")
    print(f"Schema:\n{parquet_file.schema}")

    print("\n=== STREAMING ROWS ===")

    # Iterate row group by row group
    for rg in range(parquet_file.num_row_groups):
        batch = parquet_file.read_row_group(rg)
        table = batch.to_pydict()

        # Convert to row-by-row iteration
        num_rows = len(next(iter(table.values())))
        columns = list(table.keys())

        for i in range(num_rows):
            row = {col: table[col][i] for col in columns}
            print(row)

        # Explicitly free memory
        del batch
        del table

if __name__ == "__main__":
    main()