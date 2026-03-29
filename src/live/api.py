from fastapi import FastAPI
from src.live.runtime import live_predict

app = FastAPI()

@app.get("/predict")
def get_prediction():
    pred, signal = live_predict()
    return {
        "prediction": pred,
        "signal": signal,
    }