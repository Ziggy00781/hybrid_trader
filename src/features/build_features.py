from src.models.timesfm_inference import timesfm_predict

def add_timesfm_features(df, window=256, horizon=12):
    logret = np.log(df["close"] / df["close"].shift(1))

    tfm_mean = []
    tfm_std = []
    tfm_up = []
    tfm_down = []

    for i in range(len(df)):
        if i < window:
            tfm_mean.append(np.nan)
            tfm_std.append(np.nan)
            tfm_up.append(np.nan)
            tfm_down.append(np.nan)
            continue

        seq = logret.iloc[i-window:i].values
        preds = timesfm_predict(seq, horizon=horizon)

        tfm_mean.append(preds["tfm_mean"])
        tfm_std.append(preds["tfm_std"])
        tfm_up.append(preds["tfm_up_prob"])
        tfm_down.append(preds["tfm_down_prob"])

    df["tfm_mean"] = tfm_mean
    df["tfm_std"] = tfm_std
    df["tfm_up_prob"] = tfm_up
    df["tfm_down_prob"] = tfm_down

    return df