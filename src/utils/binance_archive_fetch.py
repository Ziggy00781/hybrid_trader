import requests
import zipfile
import io
import pandas as pd
from pathlib import Path

BASE_URL = "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/5m/"
OUT_DIR = Path("data/raw/binance/archive")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def download_month(year: int, month: int):
    fname = f"BTCUSDT-5m-{year}-{month:02d}.zip"
    url = BASE_URL + fname
    # Expected CSV name inside the ZIP
    csv_name = f"BTCUSDT-5m-{year}-{month:02d}.csv"
    csv_path = OUT_DIR / csv_name

    # Skip if already downloaded
    if csv_path.exists():
        print(f"✅ Skipping {csv_name}, already exists")
        return csv_path

    print(f"⬇️ Downloading {url} ...")
    r = requests.get(url)
    if r.status_code != 200:
        print(f"❌ {year}-{month:02d} not available")
        return None

    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        z.extract(csv_name, OUT_DIR)
        return csv_path

def build_parquet(start_year=2017, end_year=2026):
    dfs = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            f = download_month(year, month)
            if f is None:
                continue
            df = pd.read_csv(
                f,
                header=None,
                names=[
                    "timestamp","open","high","low","close","volume",
                    "close_time","quote_asset_volume","trades",
                    "taker_buy_base","taker_buy_quote","ignore"
                ]
            )
            # Drop invalid timestamps
            df = df[pd.to_numeric(df["timestamp"], errors="coerce").notnull()]
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True, errors="coerce")
            df = df.dropna(subset=["timestamp"])
            dfs.append(df[["timestamp","open","high","low","close","volume"]])

    if dfs:
        full = pd.concat(dfs).sort_values("timestamp")
        out_path = Path("data/raw/binance/binance_global_btcusdt_5m.parquet")
        full.to_parquet(out_path)
        print(f"✅ Saved {len(full)} rows to {out_path}")

if __name__ == "__main__":
    build_parquet()