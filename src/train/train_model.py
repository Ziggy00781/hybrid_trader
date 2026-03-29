import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
import joblib

from src.features.ta_regime_features import build_features

def make_labels(df: pd.DataFrame,
                horizon: int = 12,
                tp_pct: float = 0.004,
                sl_pct: float = 0.004) -> pd.Series:
    close = df["close"]
    future = close.shift(-horizon)
    max_future = close.rolling(window=horizon).max().shift(-horizon + 1)
    min_future = close.rolling(window=horizon).min().shift(-horizon + 1)

    tp_hit = (max_future >= close * (1 + tp_pct))
    sl_hit = (min_future <= close * (1 - sl_pct))

    # long label: 1 if tp hit and sl not hit first (simplified)
    label = (tp_hit & ~sl_hit).astype(int)
    return label

def train():
    raw_path = Path("data/raw/btcusdt_5m.parquet")
    df = pd.read_parquet(raw_path)

    feats = build_features(df)
    aligned = df.loc[feats.index]
    y = make_labels(aligned)

    data = feats.join(y.rename("label")).dropna()
    X = data.drop(columns=["regime", "label"])
    y = data["label"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = LGBMClassifier(
        n_estimators=300,
        max_depth=-1,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9
    )
    model.fit(X_train, y_train)

    val_pred = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_pred)
    print(f"Validation AUC: {auc:.3f}")

    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, "models/btcusdt_5m_lgbm.pkl")

if __name__ == "__main__":
    train()