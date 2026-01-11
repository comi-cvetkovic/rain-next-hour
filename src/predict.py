from __future__ import annotations

import joblib
import pandas as pd

from src.data import fetch_hourly_weather
from src.features import FEATURE_COLS


MODEL_PATH = "models/rain_next_hour_model.joblib"


def load_bundle(model_path: str = MODEL_PATH) -> dict:
    return joblib.load(model_path)


def predict_rain_next_hour(latitude: float, longitude: float) -> dict:
    """
    Fetch latest hourly data and predict probability of rain in the next hour.
    """
    bundle = load_bundle()
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    df = fetch_hourly_weather(latitude, longitude, past_days=2)

    # Use latest available hour as "current state"
    latest = df.sort_values("time").iloc[-1]

    X = pd.DataFrame([{c: float(latest[c]) for c in feature_cols}])

    prob = float(model.predict_proba(X)[0, 1])
    pred = int(prob >= 0.5)

    return {
        "time_used": str(latest["time"]),
        "prob_rain_next_hour": prob,
        "pred_rain_next_hour": pred,
        "features_used": {c: float(latest[c]) for c in feature_cols},
        "recent_df": df.tail(24).reset_index(drop=True),  # for UI chart/table
    }
