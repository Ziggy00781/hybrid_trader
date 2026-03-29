import torch
import numpy as np
import ccxt
from pathlib import Path
from src.models.patchtst import PatchTST

# Prefer GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths (mirror your Linux layout)
TRAIN_RAW_PATH = Path("data/processed/patchtst/train_raw.pt")
MODEL_PATH = Path("models/patchtst/patchtst_best.pt")

SEQ_LEN = 1024

# -----------------------------
# 1. Load model + normalization
# -----------------------------
print(f"[runtime] Using device: {DEVICE}")

_blob = torch.load(TRAIN_RAW_PATH, map_location=DEVICE)
_mean = _blob["mean"].numpy()
_std = _blob["std"].numpy()
NUM_FEATURES = len(_mean)

_model = PatchTST(
    c_in=NUM_FEATURES,
    c_out=1,
    seq_len=SEQ_LEN,
    pred_len=1,
)
_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
_model = _model.to(DEVICE)
_model.eval()

print("[runtime] Model and normalization loaded.")

# -----------------------------
# 2. Live data fetcher (Bybit)
# -----------------------------
_exchange = ccxt.bybit({"enableRateLimit": True})

def fetch_live_ohlcv(limit: int = SEQ_LEN):
    """
    Fetch last `limit` 5m candles for BTC/USDT from Bybit.
    Returns np.array of shape (limit, 6): [ts, o, h, l, c, v]
    """
    ohlcv = _exchange.fetch_ohlcv("BTC/USDT", timeframe="5m", limit=limit)
    arr = np.array(ohlcv, dtype=np.float32)
    return arr  # [ts, o, h, l, c, v]

# -----------------------------
# 3. Feature builder (MATCH TRAINING)
# -----------------------------
def build_features_from_ohlcv(ohlcv: np.ndarray) -> np.ndarray:
    """
    Build exactly 1024 rows × 10 features.
    If fewer than 1024 candles are returned, pad with zeros.
    If more, truncate.
    """
    o = ohlcv[:, 1]
    h = ohlcv[:, 2]
    l = ohlcv[:, 3]
    c = ohlcv[:, 4]
    v = ohlcv[:, 5]

    ret = np.zeros_like(c)
    ret[1:] = np.log(c[1:] / c[:-1])

    hl_spread = h - l
    oc_spread = c - o
    vol_norm = v / (v.mean() + 1e-8)
    close_rel_high = c / (h + 1e-8)

    feats = np.column_stack([
        o, h, l, c, v,
        ret, hl_spread, oc_spread,
        vol_norm, close_rel_high
    ]).astype(np.float32)

    # --- enforce 1024 rows ---
    N = feats.shape[0]
    if N < 1024:
        pad = np.zeros((1024 - N, feats.shape[1]), dtype=np.float32)
        feats = np.vstack([pad, feats])   # pad at the front
    elif N > 1024:
        feats = feats[-1024:]             # take last 1024

    print(f"[runtime] feats shape after fix: {feats.shape}")
    return feats

# -----------------------------
# 4. Normalization
# -----------------------------
def normalize_window(X: np.ndarray) -> np.ndarray:
    """
    X: (SEQ_LEN, NUM_FEATURES)
    """
    return (X - _mean) / _std

# -----------------------------
# 5. Single prediction
# -----------------------------
def predict_return(X_norm: np.ndarray) -> float:
    """
    X_norm: (SEQ_LEN, NUM_FEATURES)
    Returns scalar predicted log-return.
    """
    x = X_norm.reshape(1, SEQ_LEN, NUM_FEATURES)
    with torch.no_grad():
        pred = _model(torch.tensor(x, device=DEVICE)).item()
    return float(pred)

# -----------------------------
# 6. Signal generator
# -----------------------------
def generate_signal(pred: float) -> str:
    """
    Map predicted return to a discrete trading signal.
    """
    if pred > 0.0005:
        return "STRONG_LONG"
    elif pred > 0:
        return "LONG"
    elif pred < -0.0005:
        return "STRONG_SHORT"
    elif pred < 0:
        return "SHORT"
    else:
        return "FLAT"

# -----------------------------
# 7. High-level helper: full live step
# -----------------------------
def live_predict():
    """
    Fetch live data, build features, normalize, predict, and return (pred, signal).
    """
    ohlcv = fetch_live_ohlcv(limit=SEQ_LEN)
    feats = build_features_from_ohlcv(ohlcv)
    X_norm = normalize_window(feats)
    pred = predict_return(X_norm)
    signal = generate_signal(pred)
    return pred, signal