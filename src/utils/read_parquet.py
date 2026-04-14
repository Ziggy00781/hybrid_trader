import sys
import pandas as pd
from pathlib import Path

def main():
    if len(sys.argv) != 2:
        print("Usage: python read_parquet.py <filename.parquet>")
        sys.exit(1)

    file_path = Path(sys.argv[1])

    if not file_path.exists():
        print(f"Error: File not found -> {file_path}")
        sys.exit(1)

    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"Failed to read parquet file: {e}")
        sys.exit(1)

    # Display basic info
    print("\n=== FILE INFO ===")
    print(f"Path: {file_path.resolve()}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    # Display the content
    print("\n=== DATA PREVIEW ===")
    print(df)

if __name__ == "__main__":
    main()