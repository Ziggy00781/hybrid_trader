import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
import joblib

from src.features.ta_regime_features import build_mathematical_features as build_features


def make_labels(df: pd.DataFrame,
                horizon: int = 144,        # 12 hours (72=6h, 144=12h, 288=24h)
                tp_pct: float = 0.009,
                sl_pct: float = 0.006) -> pd.Series:
    """Create labels for multi-hour direction prediction"""
    close = df["close"]
    future_max = close.rolling(window=horizon, min_periods=horizon//2).max().shift(-horizon + 1)
    future_min = close.rolling(window=horizon, min_periods=horizon//2).min().shift(-horizon + 1)

    tp_hit = (future_max >= close * (1 + tp_pct))
    sl_hit = (future_min <= close * (1 - sl_pct))

    label = (tp_hit & ~sl_hit).astype(int)
    return label


def train():
    raw_path = Path("data/raw/binance_btcusdt_5m.parquet")
    
    print("Loading full dataset...")
    df = pd.read_parquet(raw_path)
    print(f"✅ Loaded {len(df):,} candles | {df.index[0].date()} → {df.index[-1].date()}")

    print("\nBuilding mathematical features (Markov + Stochastic + Fourier)...")
    feats = build_features(df)

    aligned = df.loc[feats.index]
    y = make_labels(aligned, horizon=144)   # 12-hour horizon

    data = feats.join(y.rename("label")).dropna()
    
    X = data.drop(columns=["regime", "label"])
    y = data["label"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=False, random_state=42
    )

    print(f"\nTraining on {len(X_train):,} samples | Validation: {len(X_val):,}")
    print(f"Positive label ratio: {y.mean():.4f}")

    model = LGBMClassifier(
        n_estimators=1500,
        learning_rate=0.025,
        max_depth=7,
        num_leaves=96,
        subsample=0.78,
        colsample_bytree=0.78,
        reg_alpha=0.4,
        reg_lambda=0.4,
        min_child_samples=20,
        random_state=42,
        verbose=-1
    )

    print("Starting training (this should take several minutes)...")

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        callbacks=None   # Removed problematic callback
    )

    val_pred = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_pred)

    print(f"\n🎯 TRAINING COMPLETE!")
    print(f"   Validation AUC       : {auc:.4f}")
    print(f"   Features used        : {X.shape[1]}")
    print(f"   Horizon              : {144} candles (~12 hours)")
    print(f"   Positive class ratio : {y.mean():.4f}")

    # Feature Importance
    importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nTop 10 Most Important Features:")
    print(importance.head(10))

    # Save model
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, "models/btcusdt_5m_lgbm.pkl")
    print("\n✅ Model saved to models/btcusdt_5m_lgbm.pkl")


if __name__ == "__main__":
    train()