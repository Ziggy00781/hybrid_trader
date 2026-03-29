#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------
# CONFIG
# ---------------------------------------------
DATA_DIR="data/raw"
MERGED_DIR="data/merged"
BYBIT_URL="https://public.bybit.com/trading/BTCUSDT/"
BINANCE_URL="https://data.binance.vision/data/futures/um/daily/klines/BTCUSDT/5m/"
BINANCEUS_URL="https://data.binance.us/api/v3/klines"   # API-based, not archive
START_YEAR=2017
END_YEAR=$(date +"%Y")

# ---------------------------------------------
# Ensure directories exist
# ---------------------------------------------
mkdir -p "$DATA_DIR/bybit" "$DATA_DIR/binance" "$DATA_DIR/binanceus" "$MERGED_DIR"

echo "[*] Starting dataset preparation..."
echo "[*] Data directory: $DATA_DIR"
echo "[*] Years: $START_YEAR to $END_YEAR"

# ---------------------------------------------
# Function: download Bybit archives (resumable)
# ---------------------------------------------
download_bybit() {
    echo "[*] Downloading Bybit BTCUSDT 5m archives..."
    cd "$DATA_DIR/bybit"

    for YEAR in $(seq $START_YEAR $END_YEAR); do
        for MONTH in $(seq -w 1 12); do
            FILE="BTCUSDT_${YEAR}-${MONTH}.csv.gz"
            URL="${BYBIT_URL}${FILE}"

            if [ -f "$FILE" ]; then
                echo "    [skip] $FILE already exists"
                continue
            fi

            echo "    [dl] $FILE"
            wget -c "$URL" || echo "    [warn] Missing: $FILE"
        done
    done

    cd - >/dev/null
}

# ---------------------------------------------
# Function: download Binance Global archives
# ---------------------------------------------
download_binance() {
    echo "[*] Downloading Binance Global BTCUSDT 5m archives..."
    cd "$DATA_DIR/binance"

    for YEAR in $(seq $START_YEAR $END_YEAR); do
        for MONTH in $(seq -w 1 12); do
            FILE="BTCUSDT-5m-${YEAR}-${MONTH}.zip"
            URL="${BINANCE_URL}${FILE}"

            if [ -f "$FILE" ]; then
                echo "    [skip] $FILE already exists"
                continue
            fi

            echo "    [dl] $FILE"
            wget -c "$URL" || echo "    [warn] Missing: $FILE"
        end
    done

    cd - >/dev/null
}

# ---------------------------------------------
# Function: download BinanceUS data (API)
# ---------------------------------------------
download_binanceus() {
    echo "[*] Downloading BinanceUS BTCUSDT 5m data via API..."
    cd "$DATA_DIR/binanceus"

    # API returns JSON arrays; you will convert them in Python
    python3 - << 'EOF'
import requests, pandas as pd, datetime as dt, os

start = dt.datetime(2019,1,1)
end = dt.datetime.utcnow()
step = dt.timedelta(days=1)

os.makedirs(".", exist_ok=True)

while start < end:
    t1 = int(start.timestamp() * 1000)
    t2 = int((start + step).timestamp() * 1000)
    fname = f"BTCUSDT_5m_{start.date()}.json"

    if os.path.exists(fname):
        print("[skip]", fname)
        start += step
        continue

    print("[dl]", fname)
    url = f"https://data.binance.us/api/v3/klines?symbol=BTCUSDT&interval=5m&startTime={t1}&endTime={t2}&limit=1000"
    r = requests.get(url, timeout=10)

    if r.status_code == 200 and r.json():
        with open(fname, "w") as f:
            f.write(r.text)
    else:
        print("[warn] No data for", start.date())

    start += step
EOF

    cd - >/dev/null
}

# ---------------------------------------------
# Function: merge all exchanges into one Parquet
# ---------------------------------------------
merge_all() {
    echo "[*] Merging all exchanges into a unified Parquet..."
    python3 - << 'EOF'
import pandas as pd
import glob, os

raw = "data/raw"
merged = "data/merged/btcusdt_5m_merged.parquet"

dfs = []

# Bybit
for f in sorted(glob.glob(f"{raw}/bybit/*.csv.gz")):
    try:
        df = pd.read_csv(f, compression="gzip")
        dfs.append(df)
    except Exception as e:
        print("[warn] bad file:", f, e)

# Binance Global
for f in sorted(glob.glob(f"{raw}/binance/*.zip")):
    try:
        df = pd.read_csv(f, compression="zip")
        dfs.append(df)
    except Exception as e:
        print("[warn] bad file:", f, e)

# BinanceUS
for f in sorted(glob.glob(f"{raw}/binanceus/*.json")):
    try:
        df = pd.read_json(f)
        dfs.append(df)
    except Exception as e:
        print("[warn] bad file:", f, e)

if not dfs:
    raise RuntimeError("No data found to merge.")

df = pd.concat(dfs, ignore_index=True)

# Normalize timestamp
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
elif "open_time" in df.columns:
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)

df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])

os.makedirs("data/merged", exist_ok=True)
df.to_parquet(merged)
print("[*] Saved merged dataset:", merged)
EOF
}

# ---------------------------------------------
# Run all steps
# ---------------------------------------------
download_bybit
download_binance
download_binanceus
merge_all

echo "[*] Dataset preparation complete."
echo "[*] Next: python -m src.utils.prepare_patchtst_dataset"
