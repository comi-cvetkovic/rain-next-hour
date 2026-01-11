# RainNextHour — Will it rain in the next hour?

## 0) Project name and group members
**Project:** RainNextHour  
**Group members:** <Your names here>

## 1) Dynamic data sources (no Kaggle static datasets)
We use **dynamic live weather data** from Open-Meteo:
- **Open-Meteo Geocoding API**: converts a city name into latitude/longitude.
- **Open-Meteo Forecast API**: provides hourly weather measurements for the past days and near-future.
The data is retrieved on-demand and updates continuously.

## 2) Prediction problem
We solve: **Binary classification**
- **Question:** Will it rain in the next hour?
- **Label:** `rain_next_hour = 1` if next hour precipitation > 0.1 mm, else 0
- **Features (hour t):** temperature, humidity, pressure, cloud cover, wind speed, precipitation
- **Model:** Logistic Regression (with scaling), class_weight=balanced
- **Metric:** Classification report + ROC-AUC during training

## 3) UI you provide to show value
We provide a **Streamlit web UI**:
- User inputs a city name (and optional country code)
- App geocodes it to lat/lon
- App fetches the latest hourly data (dynamic)
- App displays:
  - probability of rain in the next hour
  - whether rain is likely/unlikely
  - the latest input features used
  - a table and line chart showing recent precipitation history

## 4) Technologies used
- Python 3.x
- Data ingestion: requests + Open-Meteo APIs
- Feature engineering: pandas
- Model training/inference: scikit-learn
- Model persistence: joblib
- UI: Streamlit

## How to run
### 1) Install
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
