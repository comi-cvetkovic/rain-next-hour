from __future__ import annotations

import pandas as pd


FEATURE_COLS = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "cloud_cover",
    "wind_speed_10m",
    "precipitation",  # current hour precip can be predictive too
]


def make_supervised(df: pd.DataFrame, rain_threshold_mm: float = 0.1) -> tuple[pd.DataFrame, pd.Series]:
    """
    Convert hourly time series into supervised learning:
    X at time t -> y = rain at time t+1
    """
    df = df.copy()

    # Label is based on NEXT hour precipitation
    df["precip_next_hour"] = df["precipitation"].shift(-1)
    df["rain_next_hour"] = (df["precip_next_hour"] > rain_threshold_mm).astype(int)

    # Drop last row (no t+1 available) and any missing
    df = df.dropna(subset=FEATURE_COLS + ["rain_next_hour"])

    X = df[FEATURE_COLS].astype(float)
    y = df["rain_next_hour"].astype(int)
    return X, y
