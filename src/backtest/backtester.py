import pandas as pd
import joblib
from src.features.ta_regime_features import build_features

def generate_signals(df: pd.DataFrame,
                     prob_threshold: float = 0.6,
                     model=None) -> pd.DataFrame:
    feats = build_features(df)

    if model is None:
        model = joblib.load("models/btcusdt_5m_lgbm.pkl")

    X = feats.drop(columns=["regime"])
    probs = model.predict_proba(X)[:, 1]

    signals = pd.DataFrame(index=feats.index)
    signals["prob_long"] = probs
    signals["regime"] = feats["regime"]
    signals["signal"] = 0
    signals.loc[
        (signals["prob_long"] >= prob_threshold) &
        (signals["regime"] == "trending_up"),
        "signal"
    ] = 1
    return signals