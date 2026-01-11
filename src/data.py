from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import requests
import pandas as pd


@dataclass(frozen=True)
class Location:
    name: str
    latitude: float
    longitude: float


def geocode_city(city: str, country_code: Optional[str] = None) -> Location:
    """
    Uses Open-Meteo geocoding to resolve a city name to lat/lon.
    """
    params = {"name": city, "count": 1, "language": "en", "format": "json"}
    if country_code:
        params["country_code"] = country_code

    r = requests.get("https://geocoding-api.open-meteo.com/v1/search", params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    results = data.get("results") or []
    if not results:
        raise ValueError(f"Could not find location for city='{city}'. Try a more specific name.")

    best = results[0]
    name = best.get("name", city)
    admin1 = best.get("admin1")
    country = best.get("country")
    display = ", ".join([x for x in [name, admin1, country] if x])

    return Location(
        name=display,
        latitude=float(best["latitude"]),
        longitude=float(best["longitude"]),
    )


def fetch_hourly_weather(latitude: float, longitude: float, past_days: int = 7) -> pd.DataFrame:
    """
    Fetch hourly weather data from Open-Meteo (dynamic, updates continuously).
    We fetch past_days of history + today/next hours.
    """
    hourly_vars = [
        "temperature_2m",
        "relative_humidity_2m",
        "surface_pressure",
        "cloud_cover",
        "wind_speed_10m",
        "precipitation",
    ]

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(hourly_vars),
        "past_days": past_days,
        "forecast_days": 1,
        "timezone": "auto",
    }

    r = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    hourly = data.get("hourly") or {}
    times = hourly.get("time")
    if not times:
        raise ValueError("API response missing hourly time series.")

    df = pd.DataFrame({"time": pd.to_datetime(times)})
    for v in hourly_vars:
        df[v] = hourly.get(v)

    df = df.dropna().sort_values("time").reset_index(drop=True)
    return df
