# src/live/runtime.py (updated version)
import torch
import numpy as np
import ccxt
import logging
from pathlib import Path
from typing import Tuple, Optional
from src.models.patchtst import PatchTST

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Global variables
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_mean = None
_std = None
NUM_FEATURES = None
_model = None
_exchange = None

# Configuration
CONFIG = {
    "train_raw_path": Path("data/processed/patchtst/train_raw.pt"),
    "model_path": Path("models/patchtst/patchtst_best.pt"),
    "seq_len": 1024,
    "symbol": "BTC/USDT",
    "timeframe": "5m",
    "signal_thresholds": {
        "strong_long": 0.0005,
        "long": 0,
        "strong_short": -0.0005,
        "short": 0,
    }
}

# -----------------------------
# Utility functions
# -----------------------------
def load_model_and_normalization():
    """Initialize model and load normalization parameters"""
    global _mean, _std, NUM_FEATURES, _model
    
    try:
        logger.info("Loading model and normalization parameters...")
        blob = torch.load(CONFIG["train_raw_path"], map_location=DEVICE)
        _mean = blob["mean"].numpy()
        _std = blob["std"].numpy()
        
        # Validate normalization parameters
        if np.any(_std == 0):
            logger.warning("Found zero standard deviation in normalization. Clipping to prevent division by zero.")
            _std = np.clip(_std, 1e-8, None)
        
        NUM_FEATURES = len(_mean)
        
        _model = PatchTST(
            c_in=NUM_FEATURES,
            c_out=1,
            seq_len=CONFIG["seq_len"],
            pred_len=1,
        )
        _model.load_state_dict(torch.load(CONFIG["model_path"], map_location=DEVICE))
        _model = _model.to(DEVICE)
        _model.eval()
        logger.info("Model and normalization loaded successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to load model or normalization parameters: {e}")
        return False

def init_exchange():
    """Initialize exchange connection"""
    global _exchange
    
    try:
        _exchange = ccxt.bybit({
            "enableRateLimit": True,
            "options": {"adjustForTimeDifference": True},
            "timeout": 10000,  # 10 seconds timeout
        })
        logger.info("Connected to exchange successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize exchange connection: {e}")
        return False

def fetch_live_ohlcv(limit: int = CONFIG["seq_len"]) -> Optional[np.ndarray]:
    """
    Fetch last `limit` candles for BTC/USDT from exchange.
    Returns np.array of shape (limit, 6): [ts, o, h, l, c, v]
    """
    if _exchange is None:
        logger.error("Exchange not initialized")
        return None
        
    try:
        ohlcv = _exchange.fetch_ohlcv(CONFIG["symbol"], timeframe=CONFIG["timeframe"], limit=limit)
        if not ohlcv:
            logger.warning("Received empty OHLCV data from exchange.")
            return None
            
        arr = np.array(ohlcv, dtype=np.float32)
        logger.debug(f"Fetched {len(arr)} OHLCV records")
        return arr
    except ccxt.NetworkError as e:
        logger.error(f"Network error while fetching data: {e}")
        return None
    except ccxt.ExchangeError as e:
        logger.error(f"Exchange error while fetching data: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error while fetching data: {e}")
        return None

def build_features_from_ohlcv(ohlcv: np.ndarray) -> Optional[np.ndarray]:
    """
    Build features from OHLCV data.
    """
    try:
        if ohlcv.shape[0] < 2:
            logger.error("Insufficient data to calculate returns.")
            return None
            
        o = ohlcv[:, 1]
        h = ohlcv[:, 2]
        l = ohlcv[:, 3]
        c = ohlcv[:, 4]
        v = ohlcv[:, 5]

        # Calculate log returns with safety check
        ret = np.zeros_like(c)
        ret[1:] = np.log(np.clip(c[1:] / c[:-1], 1e-10, None))

        hl_spread = h - l
        oc_spread = c - o
        
        # Normalize volume with safety check
        vol_mean = v.mean()
        vol_norm = v / (vol_mean + 1e-8) if vol_mean > 0 else v
        
        # Relative high with safety check
        close_rel_high = c / (np.maximum(h, 1e-8))

        feats = np.column_stack([
            o, h, l, c, v,
            ret, hl_spread, oc_spread,
            vol_norm, close_rel_high
        ]).astype(np.float32)

        # Enforce correct shape
        N = feats.shape[0]
        if N < CONFIG["seq_len"]:
            pad_rows = CONFIG["seq_len"] - N
            pad = np.zeros((pad_rows, feats.shape[1]), dtype=np.float32)
            feats = np.vstack([pad, feats])
            logger.debug(f"Padded {pad_rows} rows to match sequence length")
        elif N > CONFIG["seq_len"]:
            feats = feats[-CONFIG["seq_len"]:]
            logger.debug("Truncated data to match sequence length")

        logger.debug(f"Features shape after fix: {feats.shape}")
        
        if feats.shape != (CONFIG["seq_len"], 10):  # 10 features
            logger.error(f"Feature shape mismatch. Expected ({CONFIG['seq_len']}, 10), got {feats.shape}")
            return None
            
        return feats
    except Exception as e:
        logger.error(f"Error building features: {e}")
        return None

def normalize_window(X: np.ndarray) -> Optional[np.ndarray]:
    """
    Normalize features using loaded parameters.
    """
    try:
        if _mean is None or _std is None:
            logger.error("Normalization parameters not loaded")
            return None
            
        if X.shape[1] != len(_mean) or X.shape[1] != len(_std):
            logger.error("Feature dimension mismatch in normalization")
            return None
            
        normalized = (X - _mean) / _std
        return normalized
    except Exception as e:
        logger.error(f"Error during normalization: {e}")
        return None

def predict_return(X_norm: np.ndarray) -> Optional[float]:
    """
    Make prediction using the loaded model.
    """
    try:
        if _model is None:
            logger.error("Model not loaded")
            return None
            
        if X_norm.shape != (CONFIG["seq_len"], 10):
            logger.error(f"Input shape mismatch. Expected ({CONFIG['seq_len']}, 10), got {X_norm.shape}")
            return None
            
        x = X_norm.reshape(1, CONFIG["seq_len"], 10)
        with torch.no_grad():
            pred = _model(torch.tensor(x, device=DEVICE, dtype=torch.float32)).item()
        return float(pred)
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return None

def generate_signal(pred: float) -> str:
    """
    Map predicted return to trading signal.
    """
    thresholds = CONFIG["signal_thresholds"]
    if pred > thresholds["strong_long"]:
        return "STRONG_LONG"
    elif pred > thresholds["long"]:
        return "LONG"
    elif pred < thresholds["strong_short"]:
        return "STRONG_SHORT"
    elif pred < thresholds["short"]:
        return "SHORT"
    else:
        return "FLAT"

def live_predict(fetch_function=None) -> Tuple[Optional[float], str]:
    """
    Execute complete prediction pipeline.
    Returns (prediction, signal) tuple.
    """
    try:
        logger.info("Starting live prediction cycle...")
        
        # Use provided fetch function or default
        fetch_fn = fetch_function or fetch_live_ohlcv
        
        # Fetch data
        ohlcv = fetch_fn(limit=CONFIG["seq_len"])
        if ohlcv is None:
            return None, "ERROR_FETCHING_DATA"
            
        # Build features
        feats = build_features_from_ohlcv(ohlcv)
        if feats is None:
            return None, "ERROR_BUILDING_FEATURES"
            
        # Normalize
        X_norm = normalize_window(feats)
        if X_norm is None:
            return None, "ERROR_NORMALIZING"
            
        # Predict
        pred = predict_return(X_norm)
        if pred is None:
            return None, "ERROR_PREDICTING"
            
        # Generate signal
        signal = generate_signal(pred)
        logger.info(f"Prediction: {pred:.6f}, Signal: {signal}")
        
        return pred, signal
        
    except Exception as e:
        logger.error(f"Critical error in live prediction: {e}", exc_info=True)
        return None, "CRITICAL_ERROR"

def initialize_runtime() -> bool:
    """
    Initialize all components needed for live prediction.
    """
    try:
        # Load model and normalization
        if not load_model_and_normalization():
            return False
            
        # Initialize exchange
        if not init_exchange():
            return False
            
        logger.info("Runtime initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error during runtime initialization: {e}")
        return False