import glob
import pandas as pd

# Step 1: Find all matching files
file_paths = glob.glob("data/ticks/BTC_USDT/BTC_USDT_*.parquet")

if not file_paths:
    print("No files found. Check the directory and file names.")
else:
    # Step 2: Read the first file (or combine all files)
    df = pd.read_parquet(file_paths[0])
    print("First file loaded successfully:")
    print(df.head())
